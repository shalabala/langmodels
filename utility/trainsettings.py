import torch
import os
from torch.utils.data import DataLoader
from torch.nn import Module


class TrainSettings:
    """
    Class to hold the training settings for a model.
    This class is used to configure the training process, including
    the model, dataset, optimizer, and other parameters.
    """

    def __init__(self,
                 name: str,
                 model: Module,
                 dataset_tr: torch.utils.data.Dataset,
                 dataset_val: torch.utils.data.Dataset,
                 epochs: int = 100,
                 eval_after_epoch: int = 1,
                 save_path: str = None,
                 device: str = 'cpu',
                 batch_size: int = 64,
                 optimizer_type: str = 'adam',
                 lr: float = 0e-3,
                 momentum: float = 0.9,
                 save_after_epoch: int = None,
                 print_after_steps: int = -1,
                 print_memory: bool = False,
                 ):

        if save_after_epoch is None:
            save_after_epoch = eval_after_epoch

        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"{model.name()}_saves")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        self.model = model
        self.lr = lr,
        self.device = device

        self.epochs = epochs
        self.model.to(device)

        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.optimizer_type = optimizer_type

        self.eval_after_epoch = eval_after_epoch

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_data = DataLoader(
            dataset_tr, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False)
        self.save_path = save_path
        self.save_after_epoch = save_after_epoch
        self.name = name
        self.print_memory = print_memory
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.print_after_steps = print_after_steps

    def properties(self) -> dict:
        """
        Returns a dictionary with the properties of the training settings.
        """
        return {
            'model': self.model.name(),
            'optimizer': self.optimizer_type,
            'loss_fn': "CrossEntropyLoss",
            'save_path': self.save_path,
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
            'momentum': self.momentum,
            'save_after_epoch': self.save_after_epoch,
            'name': self.name
        }

    def save_if_needed(self, epoch: int) -> None:
        """
        Save the model if the current epoch is a multiple of save_after_epoch.
        Args:
            epoch (int): The current epoch.
        """
        if (epoch) % self.save_after_epoch == 0:
            path = TrainSettings._save_model(self.model, os.path.join(
                self.save_path, f'{self.name}_epoch_{epoch}'))
            print(f"Model saved at {path} after epoch {epoch}.")

    def save_final(self, epoch: int) -> None:
        """
        Save the final model after training.

        Args:
            epoch (int): The current epoch.
        """
        path = TrainSettings._save_model(self.model, os.path.join(
            self.save_path, f'{self.name}_epoch_{epoch}_final'))
        print(f"Final model saved at {path} after epoch {epoch+1}.")
        
    def save_snapshot(self):
        """
        Save a snapshot of the model.
        """
        path = os.path.join(
            self.save_path, f'{self.name}_snapshot.pth')
        torch.save(self.model.state_dict(), path)
        print(f"Model snapshot saved at {path}.")

    @staticmethod
    def _save_model(model: torch.nn.Module, path_prefix: str, path_suffix: str = ".pth") -> str:
        """
        Save the model state dict to a file. Returns the path to the saved file.
        Args:
            model (torch.nn.Module): The model to save.
            path_prefix (str): The prefix for the file path.
            path_suffix (str): The suffix for the file path. (e.g. ".pth")
        """
        path = f"{path_prefix}{path_suffix}"
        i = 1
        while os.path.exists(path):
            path = f"{path_prefix}_{i}_{path_suffix}"
            i += 1

        torch.save(model.state_dict(), path)
        return path
