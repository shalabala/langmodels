import torch
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def pass_predictions(self, predictions, labels):
        pass

    @abstractmethod
    def get_value(self):
        pass

    def __str__(self):
        return self.__class__.__name__


class AccuracyEvaluator(Evaluator):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def clear(self):
        self.correct = 0
        self.total = 0

    def pass_predictions(self, predictions, labels):
        _, predicted = torch.max(predictions.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def get_value(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def __str__(self):
        return "ACC"


class ClassAccuracyEvaluator(Evaluator):
    def __init__(self, class_num, class_names: list[str] = None):
        self.correct = 0
        self.total = 0
        self.class_num = class_num
        self.class_names = class_names

    def clear(self):
        self.correct = 0
        self.total = 0

    def pass_predictions(self, outputs, labels):
        _, predictions = torch.max(outputs.data, 1)
        # TP
        ps = labels == self.class_num
        tps = (predictions[ps] == self.class_num).sum().item()
       
        self.correct += tps 
        self.total += ps.sum().item()

    def get_value(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def __str__(self):
        return f"ACC_{self._getname()}"
    
    def _getname(self):
        return self.class_names[self.class_num] if self.class_names is not None else str(self.class_num)
