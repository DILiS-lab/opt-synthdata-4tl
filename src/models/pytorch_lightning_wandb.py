"""Provides a model that predicts next timesteps from with a\
     pytorch lightning architecture."""

import abc
import dataclasses
import time
import shutil
import os

import torch
from torch import utils
import numpy as np
import numpy.typing as npt
import pandas as pd

from simba_ml.prediction.time_series.data_loader import window_generator
from simba_ml.prediction.time_series.models import model
from simba_ml.prediction import normalizer
from simba_ml.prediction.time_series.config import (
    time_series_config,
)

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

try:  # pragma: no cover
    import pytorch_lightning as pl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch Lightning is not installed. Please install it to "
        "use the PyTorchLightningModel."
    ) from e

if tuple(int(v) for v in pl.__version__.split(".")) < (  # type: ignore[attr-defined]
    1,
    9,
    0,
):  # pragma: no cover
    raise ImportError(
        "PyTorch Lightning version 1.9.0 or higher is required \
            for the PyTorchLightningModel."
    )

torch.set_float32_matmul_precision("high")

wandb_project = os.environ["WANDB_PROJECT"] if "WANDB_PROJECT" in os.environ else None


@dataclasses.dataclass
class ArchitectureParams:
    """Defines the parameters for the architecture."""

    units: int = 32
    activation: str = "relu"


@dataclasses.dataclass
class TrainingParams:
    """Defines the parameters for the training."""

    patience: int = 5
    batch_size: int = 64
    validation_split: float = 0.2
    verbose: int = 0
    accelerator: str = "auto"
    learning_rate: float = 0.001
    finetuning_learning_rate: float | None = None
    epochs: int = 10
    finetuning_epochs: int | None = None
    show_progress_bar: bool = True


@dataclasses.dataclass
class PytorchLightningWandbModelConfig(model.ModelConfig):
    """Defines the configuration for the PytorchLightningModel."""

    name: str = "Pytorch Lightning Model"
    architecture_params: ArchitectureParams = dataclasses.field(
        default_factory=ArchitectureParams
    )
    training_params: TrainingParams = dataclasses.field(default_factory=TrainingParams)
    normalize: bool = True
    seed: int = 42
    finetuning: bool = False


class PytorchLightningWandbModel(model.Model):
    """Defines a Pytorch Lightning model to predict the next timestamps."""

    model_params: PytorchLightningWandbModelConfig

    def __init__(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: PytorchLightningWandbModelConfig,
    ) -> None:
        """Initializes the model.

        Args:
            time_series_params: configuration for the time series that
                influence the training and archicture of the model.
            model_params: configuration for the model.

        Raises:
            TypeError: if input_length or output_length is not an integer.
        """
        super().__init__(time_series_params, model_params)
        self.set_seed(self.model_params.seed)
        if self.model_params.normalize:
            self.normalizer = normalizer.Normalizer()
        self.model = self.get_model(time_series_params, model_params)
        self.time_series_params = time_series_params
        self.model_params = model_params
        self.training_checkpoint_directory = (
            "src/models/training_checkpoints-"
            + self.model_params.name
            + "-"
            + str(time.time()).replace(".", "")
        )
        self.finetuning_checkpoint_directory = (
            "src/models/finetuning_checkpoint-"
            + self.model_params.name
            + "-"
            + str(time.time()).replace(".", "")
        )
        self.training_checkpoint_callback = ModelCheckpoint(
            dirpath=self.training_checkpoint_directory,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            verbose=False,
            every_n_epochs=1,
        )
        self.finetuning_checkpoint_callback = ModelCheckpoint(
            dirpath=self.finetuning_checkpoint_directory,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            verbose=False,
            every_n_epochs=1,
        )
        self.early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=self.model_params.training_params.patience,
            mode="min",
            verbose=False,
        )
        self.start_time = time.time()

    @abc.abstractmethod
    def get_model(
        self,
        time_series_params: time_series_config.TimeSeriesConfig,
        model_params: PytorchLightningWandbModelConfig,
    ) -> pl.LightningModule:
        """Returns the model.

        Args:
            time_series_params: configuration for the time series that
                influence the training and archicture of the model.
            model_params: configuration for the model.
        """

    def set_seed(self, seed: int) -> None:
        """Sets the seed for the model.

        Args:
            seed: seed to set.
        """
        torch.manual_seed(seed)

    def train(self, train: list[pd.DataFrame]) -> None:
        """Trains the model with the given data.

        Args:
            train: training data.
        """
        train = [array.astype(np.float32) for array in train]
        if self.model_params.normalize:
            train = self.normalizer.normalize_train_data(train, self.time_series_params)
        X_train, y_train = window_generator.create_window_dataset(
            train, self.time_series_params
        )
        train_data = list(zip(X_train, y_train))
        train_loader: utils.data.DataLoader[
            torch.FloatTensor  # pylint: disable=no-member
        ] = utils.data.DataLoader(
            train_data[: int(len(train_data) * 0.9)],  # type: ignore[arg-type]
            batch_size=self.model_params.training_params.batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader: utils.data.DataLoader[
            torch.FloatTensor  # pylint: disable=no-member
        ] = utils.data.DataLoader(
            train_data[int(len(train_data) * 0.9) :],  # type: ignore[arg-type]
            batch_size=self.model_params.training_params.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.start_trainer(train_loader, val_loader=val_loader)

    def start_trainer(
        self,
        train_loader: utils.data.dataloader.DataLoader[
            torch.FloatTensor  # pylint: disable=no-member
        ],
        val_loader: utils.data.dataloader.DataLoader[
            torch.FloatTensor  # pylint: disable=no-member
        ],
    ) -> None:
        """Starts the trainer for the model.

        Args:
            train_loader: the data loader for the training data.
        """
        # TODO name needs to be changed/removed
        wandb_logger = (
            WandbLogger(
                project="thesis-1",
                entity="julianzabbarov",
                name=self.name
                + ("-Finetuning" if self.model_params.finetuning else ""),
            )
            if wandb_project
            else None
        )
        if self.model_params.finetuning:
            check_finetuning_params(self.model_params)
            print(
                f"Loading trained model from {self.training_checkpoint_callback.best_model_path}..."
            )
            ckpt = torch.load(self.training_checkpoint_callback.best_model_path)
            trainer = pl.Trainer(
                callbacks=[self.finetuning_checkpoint_callback],
                max_epochs=self.model_params.training_params.finetuning_epochs
                + ckpt["epoch"],
                log_every_n_steps=1,
                accelerator=self.model_params.training_params.accelerator,
                enable_progress_bar=self.model_params.training_params.show_progress_bar,
                logger=wandb_logger,
            )
            trainer.fit(
                self.model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=self.training_checkpoint_callback.best_model_path,
            )
        else:
            print("Starting training...")
            trainer = pl.Trainer(
                callbacks=[
                    self.training_checkpoint_callback,
                    self.early_stopping_callback,
                ],
                max_epochs=self.model_params.training_params.epochs,
                log_every_n_steps=1,
                accelerator=self.model_params.training_params.accelerator,
                enable_progress_bar=self.model_params.training_params.show_progress_bar,
                logger=wandb_logger,
            )
            trainer.fit(self.model, train_loader, val_dataloaders=val_loader)
        if wandb_logger:
            wandb_logger.experiment.finish()

    def predict(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predicts the next timestamps for every row (time series).

        Args:
            data: np.array, where each dataframe is a time series.

        Returns:
            np.array, where each value is a time series.
        """
        self.set_seed(self.model_params.seed)
        self.model.eval()
        if self.model_params.finetuning:
            print(
                f"Loading fine-tuned model from {self.finetuning_checkpoint_callback.best_model_path}..."
            )
            best_model = self.model.load_from_checkpoint(
                self.finetuning_checkpoint_callback.best_model_path,
                time_series_params=self.time_series_params,
                model_params=self.model_params,
            )
            empty_checkpoint_directories(
                [
                    self.finetuning_checkpoint_directory,
                    self.training_checkpoint_directory,
                ]
            )
        else:
            print(
                f"Loading trained model from {self.training_checkpoint_callback.best_model_path}..."
            )
            best_model = self.model.load_from_checkpoint(
                self.training_checkpoint_callback.best_model_path,
                time_series_params=self.time_series_params,
                model_params=self.model_params,
            )
            empty_checkpoint_directories([self.training_checkpoint_directory])
        if self.model_params.normalize:
            data = self.normalizer.normalize_test_data(data)
        data_to_torch = torch.from_numpy(data).to(  # pylint: disable=no-member
            torch.float32  # pylint: disable=no-member
        )
        if next(best_model.parameters()).is_cuda:
            data_to_torch = data_to_torch.to("cuda")
        prediction = best_model(data_to_torch).cpu().detach().numpy()
        if self.model_params.normalize:
            prediction = self.normalizer.denormalize_prediction_data(prediction)
        training_time = time.time() - self.start_time
        print(f"Prediction finished. Required run time: {training_time:.2f} seconds.\n")
        return prediction


def check_finetuning_params(
    model_params: PytorchLightningWandbModelConfig,
) -> None:
    """Checks whether all the required arguments for finetuning the model are set.

    Args:
        model_params: the model parameters to check.

    Raises:
        ValueError: if the model is not set to finetuning.
    """
    if model_params.finetuning and not (
        model_params.training_params.finetuning_learning_rate
        and model_params.training_params.finetuning_epochs
    ):
        raise ValueError(
            "The model is set to finetuning but the finetuning learning rate or the finetuning epochs are not set."  # pylint: disable=line-too-long
        )


def empty_checkpoint_directories(list: list) -> None:
    """Empties the checkpoint directories."""
    for item in list:
        shutil.rmtree(item)
