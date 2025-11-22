import os
import subprocess
from dataclasses import dataclass
from typing import Tuple
import datetime
import sys
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
from IPython.display import display
import optuna
import optuna.visualization as vis
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST



@dataclass
class Config:
    """Configuration options for the Lightning MNIST example.

    Args:
        data_dir : The path to the directory where the MNIST dataset is stored. Defaults to the value of
            the 'PATH_DATASETS' environment variable or '.' if not set.

        save_dir : The path to the directory where the training logs will be saved. Defaults to 'logs/'.

        batch_size : The batch size to use during training. Defaults to 256 if a GPU is available,
            or 64 otherwise.

        max_epochs : The maximum number of epochs to train the model for. Defaults to 3.

        accelerator : The accelerator to use for training. Can be one of "cpu", "gpu", "tpu", "ipu", "auto".

        devices : The number of devices to use for training. Defaults to 1.

    Examples:
        This dataclass can be used to specify the configuration options for training a PyTorch Lightning model on the
        MNIST dataset. A new instance of this dataclass can be created as follows:

        >>> config = Config()

        The default values for each argument are shown in the documentation above. If desired, any of these values can be
        overridden when creating a new instance of the dataclass:

        >>> config = Config(batch_size=128, max_epochs=5)

    """

    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "lightning_logs/"
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    num_workers: int = 4


class LitMNIST(pl.LightningModule):
    """PyTorch Lightning module for training a multi-layer perceptron (MLP) on the MNIST dataset.

    Attributes:
        data_dir : The path to the directory where the MNIST data will be downloaded.

        hidden_size : The number of units in the hidden layer of the MLP.

        learning_rate : The learning rate to use for training the MLP.

    Methods:
        forward(x):
            Performs a forward pass through the MLP.

        training_step(batch, batch_idx):
            Defines a single training step for the MLP.

        validation_step(batch, batch_idx):
            Defines a single validation step for the MLP.

        test_step(batch, batch_idx):
            Defines a single testing step for the MLP.

        configure_optimizers():
            Configures the optimizer to use for training the MLP.

        prepare_data():
            Downloads the MNIST dataset.

        setup(stage=None):
            Splits the MNIST dataset into train, validation, and test sets.

        train_dataloader():
            Returns a DataLoader for the training set.

        val_dataloader():
            Returns a DataLoader for the validation set.

        test_dataloader():
            Returns a DataLoader for the test set.

    """

    def __init__(self, config: Config, hidden_size: int = 64, learning_rate: float = 2e-4):
        """Initializes a new instance of the LitMNIST class.

        Args:
            config : Configuration object containing data_dir, save_dir, etc.

            hidden_size : The number of units in the hidden layer of the MLP (default is 64).

            learning_rate : The learning rate to use for training the MLP (default is 2e-4).

        """
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = config.data_dir
        self.save_dir = config.save_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.config = config

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x : The input data.

        Returns:
            torch.Tensor: The output of the MLP.

        """
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        """Defines a single training step for the MLP.

        Args:
            batch: A tuple containing the input data and target labels.

            batch_idx: The index of the current batch.

        Returns:
            (torch.Tensor): The training loss.

        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
        """Defines a single validation step for the MLP.

        Args:
            batch : A tuple containing the input data and target labels.
            batch_idx : The index of the current batch.

        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
        """Defines a single testing step for the MLP.

        Args:
            batch : A tuple containing the input data and target labels.
            batch_idx : The index of the current batch.

        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer to use for training the MLP.

        Returns:
            torch.optim.Optimizer: The optimizer.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    # ------------------------------------- #
    # DATA RELATED HOOKS
    # ------------------------------------- #

    def prepare_data(self) -> None:
        """Downloads the MNIST dataset."""
        MNIST(self.data_dir, train=True, download=True)

        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        """Splits the MNIST dataset into train, validation, and test sets.

        Args:
            stage : The current stage (either "fit" or "test"). Defaults to None.

        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training set.

        Returns:
            DataLoader: The training DataLoader.

        """
        return DataLoader(self.mnist_train, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation set.

        Returns:
            DataLoader: The validation DataLoader.

        """
        return DataLoader(self.mnist_val, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test set.

        Returns:
            DataLoader: The test DataLoader.

        """
        return DataLoader(self.mnist_test, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
    
    
config = Config()
# below is used for hyperparameter tuning with optuna
# if not using optuna, set to False
use_optuna = True

if not use_optuna:



    model = LitMNIST(config)

    # Start TensorBoard automatically
    print("Starting TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", config.save_dir, "--host", "0.0.0.0", "--port", "6006"])
    print("TensorBoard available at http://localhost:6006")

    # Instantiate a PyTorch Lightning trainer with the specified configuration
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        logger=TensorBoardLogger(save_dir=config.save_dir, name="mnist"),
    )

    # Train the model using the trainer
    trainer.fit(model)

    trainer.test(ckpt_path="best")
    
else:

    # optuna parameters
    num_trials = 5
    
    
    # Create a unique log directory for this Optuna run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    optuna_log_dir = os.path.join(config.save_dir, f"optuna_{timestamp}")
    os.makedirs(optuna_log_dir, exist_ok=True)

    # Optional: Hyperparameter tuning with Optuna
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        
        model = LitMNIST(config, hidden_size=hidden_size, learning_rate=learning_rate)
        
        trainer = pl.Trainer(
            max_epochs=2,  # More epochs for tuning
            logger=False,  # Disable logging during tuning
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        
        trainer.fit(model)
        return trainer.callback_metrics["val_acc"]

    #below is used for hyperparameter tuning with optuna
    # should stay static and not be changed
    study_name = f"mnist_tuning_{timestamp}"
    study = optuna.create_study(storage=f"sqlite:///{optuna_log_dir}/db.sqlite3", study_name=study_name, direction="maximize")
    study.optimize(objective, n_trials=num_trials) 
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)
    
    # Visualize results
    vis.plot_optimization_history(study).write_image(os.path.join(optuna_log_dir, "optuna_optimization_history.png"))
    vis.plot_param_importances(study).write_image(os.path.join(optuna_log_dir, "optuna_param_importances.png"))
    vis.plot_parallel_coordinate(study).write_image(os.path.join(optuna_log_dir, "optuna_parallel_coordinate.png"))
    print(f"Plots and database saved in {optuna_log_dir}")

    # add subprocess to start optuna-dashboard
    print("Starting Optuna Dashboard...")
    subprocess.Popen(["optuna-dashboard", f"sqlite:///{optuna_log_dir}/db.sqlite3", "--host", "0.0.0.0", "--port", "8080"])
    print("Optuna Dashboard available at http://localhost:8080")
