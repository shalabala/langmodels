import torch.utils
from torch.utils.data import DataLoader
import torch
from torch.nn import Module
from torch.utils.data import Dataset
import torch.utils.data
from utility.evaluator import Evaluator
from utility.trainlogic import TrainLogic, TrainLogicImplementation
from matplotlib import pyplot as plt


class Tester:
    def __init__(self,
                 model: torch.nn.Module,
                 test_set: Dataset,
                 evaluators: list[Evaluator],
                 class_names: list[str] = None,
                 logic: TrainLogic = TrainLogicImplementation(),
                 loss: torch.nn.Module = None,
                 device='cpu',
                 batch_size=64):
        """
        Initialize the Tester with a model and device.

        Args:
            model (Module): The model to be tested.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.evaluators = evaluators
        self.batch_size = batch_size
        self.test_data = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=True)
        self.logic = logic
        self.loss = loss
        self.class_names = class_names

    def test(self) -> None:
        self.model.eval()
        with torch.no_grad():
            epoch_val_loss, evaluations = self.logic.validate(self.test_data,
                                                              self.device,
                                                              self.model,
                                                              self.loss,
                                                              self.evaluators)
            if self.loss is not None:
                print(f"Test Loss: {epoch_val_loss:.4f}")
            for eval in evaluations:
                print(f"{eval[0]}: {eval[1]:.4f}")
            print("Testing completed.")

    def top(self, n=5):
        i = 0
        self.model.eval() 
        with torch.no_grad():
            for imgs, labels in self.test_data:
                for img, label in zip(imgs, labels):
                    if i >= n:
                        break
                    img, label = img.unsqueeze(0).to(self.device), label
                    outputs = self.model(img)
                    _, predictions = torch.max(outputs, 1)
                    self._print(img[0], label.item(), predictions[0].item())
                    i += 1

    def _print(self, img, label, prediction):
        fig = plt.figure()
        plt.title(f"Label: {self._getname(label)}, " +
                  f"Prediction: {self._getname(prediction)}")
        img_to_show = img.permute(1, 2, 0).cpu().numpy()
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        img_to_show = (img_to_show * std + mean) * 255
        img_to_show = img_to_show.astype('uint8')
        plt.imshow(img_to_show)

    def _getname(self, label):
        return self.class_names[label] if self.class_names is not None else str(label)
