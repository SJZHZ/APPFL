import time
import torch
import importlib
import copy
from torch.nn import Module
from omegaconf import DictConfig
# import yaml

from typing import Optional, Any, Tuple
from torch_geometric.loader import DataLoader

# from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer
from appfl.misc.utils import apply_model_device, parse_device_str


class GridFMTrainer(VanillaTrainer):
    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.train_configs.eval_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), (
            f"Optimizer {self.train_configs.optim} not found in torch.optim"
        )
        self.optimizer = getattr(optim_module, self.train_configs.optim)(
            self.model.parameters(), **self.train_configs.optim_args
        )

        if self.loss_fn is None and hasattr(self.model, "loss_fn"):
            self.loss_fn = self.model.loss_fn

        self.device_config, self.device = parse_device_str(self.train_configs.device)

    def train(self, **kwargs):
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        self.model = apply_model_device(self.model, self.device_config, self.device)

        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataloader is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        )

        # Set up logging title
        title = (
            ["Round", "Time", "Train Loss", "Train Accuracy"]
            if (not do_validation) and (not do_pre_validation)
            else (
                [
                    "Round",
                    "Pre Val?",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
                if do_pre_validation
                else [
                    "Round",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
            )
        )
        if self.train_configs.mode == "epoch":
            title.insert(1, "Epoch")

        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            val_loss, val_accuracy = self._validate()
            self.val_results["pre_val_loss"] = val_loss
            self.val_results["pre_val_accuracy"] = val_accuracy
            content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]
            if self.train_configs.mode == "epoch":
                content.insert(1, 0)
            self.logger.log_content(content)

        for epoch in range(self.train_configs.num_local_epochs):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            for batch in self.train_dataloader:
                loss, _ = self._train_batch(batch)
                total_loss += loss

            train_loss = total_loss / len(self.train_dataloader)
            train_accuracy = "N/A"
            # print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            if do_validation:
                val_loss, val_accuracy = self._validate()
                if "val_loss" not in self.val_results:
                    self.val_results["val_loss"] = []
                    self.val_results["val_accuracy"] = []
                    self.val_results["train_loss"] = []
                self.val_results["val_loss"].append(val_loss)
                self.val_results["val_accuracy"].append(val_accuracy)
                self.val_results["train_loss"].append(train_loss)

            per_epoch_time = time.time() - start_time

            self.logger.log_content(
                [self.round, epoch, per_epoch_time, train_loss, train_accuracy]
                if (not do_validation) and (not do_pre_validation)
                else (
                    [
                        self.round,
                        epoch,
                        per_epoch_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                    if not do_pre_validation
                    else [
                        self.round,
                        epoch,
                        "N",
                        per_epoch_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                )
            )

        self.round += 1
        self.model_state = copy.deepcopy(self.model.state_dict())

        if "cuda" in self.train_configs.device:
            for k in self.model_state:
                self.model_state[k] = self.model_state[k].cpu()

    def _train_batch(self, batch):
        batch = batch.to(self.device)
        if hasattr(self.model, "shared_step"):
            outputs, loss_dict = self.model.shared_step(batch)
        else:
            outputs = self.model(
                x=batch.x,
                pe=batch.pe,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                mask=batch.mask,
            )
            loss_dict = self.loss_fn(
                outputs,
                batch.y,
                batch.edge_index,
                batch.edge_attr,
                batch.mask,
            )
        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), outputs

    def _validate(self) -> Tuple[float, str]:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = batch.to(self.device)
                if hasattr(self.model, "shared_step"):
                    _, loss_dict = self.model.shared_step(batch)
                    loss = loss_dict["loss"]
                else:
                    outputs = self.model(
                        x=batch.x,
                        pe=batch.pe,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        mask=batch.mask,
                    )
                    loss = self.loss_fn(
                        outputs,
                        batch.y,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.mask,
                    )["loss"]
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.val_dataloader), "N/A"
