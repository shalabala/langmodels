from abc import ABC, abstractmethod
import random
import time
import math
import torch
import torch.utils
import torch.utils.data

from utility.evaluator import Evaluator


class TrainLogic(ABC):
    """
    Abstract class for training logic. It defines the interface for training and validation methods.
    """
    @abstractmethod
    def train(self, device: str, imgs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> float:
        pass

    @abstractmethod
    def validate(self, val_data: torch.utils.data.DataLoader, device: str, model: torch.nn.Module,
                 loss_fn: torch.nn.Module, evaluators: list[Evaluator]) -> tuple[float, list[tuple[str, float]]]:
        pass


class MockTrainLogic(TrainLogic):
    """
    Mock class for training logic. It simulates the training and validation process.
    """

    def __init__(self, steps_per_epoch: int, batch_size: int, delay=1.0, tr_speed: float = 10, val_rate: float = 2, overfit: float = 10):
        """
        Mock class for training logic. It simulates the training and validation process.
        Args:
            steps_per_epoch (int): Number of steps per epoch.
            tr_speed (float): Training speed. The bigger the value the longer it takes to "fit" the model.
            val_rate (float): Validation rate. The bigger the value larger the distance between the
            training and validation loss. 1 means train loss = val loss.
            overfit (float): Overfitting rate. The bigger the value the later the model starts to overfit.
        """
        super().__init__()
        self.tr_speed = tr_speed
        self.val_rate = val_rate
        self.overfit = overfit
        self.epoch = 0
        self.steps_per_epoch = steps_per_epoch
        self.delay = delay
        self.batch_size = batch_size

    def train(self, device: str, imgs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> float:
        """
        Simulates the training process. It returns a loss value that decreases over time.
        The loss value is calculated based on the training speed and the number of epochs.
        """
        # formula: tr_speed / epoch
        self.epoch += 1/self.steps_per_epoch
        tr_loss = self.tr_speed / self.epoch
        time.sleep(self.delay)
        return MockTrainLogic.randomize(tr_loss)

    def validate(self, val_data: torch.utils.data.DataLoader, device: str, model: torch.nn.Module,
                 loss_fn: torch.nn.Module, evaluators: list[Evaluator]) -> tuple[float, list[tuple[str, float]]]:
        """
        Simulates the validation process. It returns a loss value that is higher than the training loss.
        The loss value is calculated based on the validation rate and the number of epochs.
        """
        # formula: (epoch*3/overfit) +(tr_speed* val_rate)/epoch
        val_loss_raw = (self.epoch*3 / self.overfit) + \
            (self.tr_speed * self.val_rate) / self.epoch
            
        val_loss = MockTrainLogic.randomize(val_loss_raw) * len(val_data)
        return (val_loss, [(str(evaluator), math.tanh(1/val_loss)) for evaluator in evaluators])

    @staticmethod
    def randomize(value, std=.1):
        """
        Randomizes the value based on the given variance.
        Args:
            value (float): The value to be randomized.
            std (float): The variance. Default is 1.
        Returns:
            float: The randomized value.
        """
        return abs(random.gauss(value, std))


class TrainLogicImplementation(TrainLogic):
    """
    Class to hold the training logic for a model.
    """

    def train(self, device: str, imgs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> float:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def validate(self, val_data: torch.utils.data.DataLoader, device: str, model: torch.nn.Module,
                 loss_fn: torch.nn.Module, evaluators: list[Evaluator]) -> tuple[float, list[tuple[str, float]]]:
        for evaluator in evaluators:
            evaluator.clear()

        epoch_val_loss = 0.0

        for imgs, labels in val_data:
            imgs, labels = imgs.to(device), labels.to(
                device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels) if loss_fn is not None else torch.tensor(0.0)
            epoch_val_loss += loss.item()
            for evaluator in evaluators:
                evaluator.pass_predictions(outputs, labels)

        return (epoch_val_loss, [(evaluator.__str__(), evaluator.get_value()) for evaluator in evaluators])
