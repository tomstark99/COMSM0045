from typing import Union, Dict, Any

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        data, labels = batch
        
        logits = self.model.forward(data.to(self.device))
        preds = logits.detach().cpu().argmax(-1)

        step_results = {
            'logits': logits.cpu(),
            'loss': self.criterion(logits, labels.to(self.device)),
            'accuracy': self.compute_accuracy(preds, labels.cpu())
        }
        return step_results

    def _train_step(self) -> Dict[str, Any]:
        self.model.train()

        train_results = {
            'loss': [],
            'accuracy': []
        }

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            step_results = self._step(batch)

            loss = step_results['loss']
            loss.backward()

            self.optimizer.step()

            train_results['loss'].append(loss.detach().cpu().item())
            train_results['accuracy'].append(step_results['accuracy'])

        return train_results

    def _val_step(self) -> Dict[str, Any]:
        self.model.eval()

        validation_results = {
            'loss': [],
            'accuracy': [],
            'preds': [],
            'labels': []
        }
        
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                _, labels = batch

                step_results = self._step(batch)
                loss = step_results['loss']
                preds = step_results['logits'].argmax(-1).numpy()
                validation_results['loss'].append(loss.cpu().item())
                validation_results['preds'].extend(list(preds))
                validation_results['labels'].extend(list(labels.numpy()))

        validation_results['accuracy'] = self.compute_accuracy(
            np.array(validation_results["labels"]), 
            np.array(validation_results["preds"])
        )
        validation_results['average_loss'] = np.sum(validation_results['loss']) / len(self.val_loader)

        return validation_results

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        for epoch in tqdm(
            range(start_epoch, epochs),
            unit=" epoch",
            dynamic_ncols=True
        ):

            train_results = self._train_step()
            loss = train_results['loss']
            accuracy = train_results['accuracy']

            if ((self.step + 1) % log_frequency) == 0:
                self.log_metrics(epoch, accuracy, loss)
            if ((self.step + 1) % print_frequency) == 0:
                self.print_metrics(epoch, accuracy, loss)

            self.step += 1

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
        
        self.print_per_class_accuracy()


    def validate(self):

        validation_results = self._val_step()

        accuracy = validation_results['accuracy']
        average_loss = np.sum(validation_results['loss']) / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

    def compute_accuracy(
        preds: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Args:
            labels: ``(batch_size, class_count)`` tensor or array containing example labels
        """
        return float((labels == preds).sum()) / len(labels)

    def print_metrics(self, epoch, accuracy, loss):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
        )

    def print_per_class_accuracy(self):
        classes = {0: 'beach',
                   1: 'bus',
                   2: 'cafe/restaurant',
                   3: 'car',
                   4: 'city_center',
                   5: 'forest_path',
                   6: 'grocery_store',
                   7: 'home',
                   8: 'library',
                   9: 'metro_station',
                  10: 'office',
                  11: 'park',
                  12: 'residential_area',
                  13: 'train',
                  14: 'tram'}

        correct_pred = {classname: 0 for classname in classes.keys()}
        total_pred = {classname: 0 for classname in classes.keys()}

        self.model.eval()
        with torch.no_grad():
            for batch, labels in self.val_loader:

                logits = self.model(batch.to(self.device))
                preds = logits.argmax(-1).cpu().numpy()

                for label, pred in zip(labels.cpu().numpy(), preds):
                    if label == pred:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        
        for classname, correct in correct_pred.items():
            accuracy = 100 * float(correct) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f}%".format(classes[classname], accuracy))

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
