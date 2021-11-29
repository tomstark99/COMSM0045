from typing import Union, Dict, Any, Tuple
from pathlib import Path

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
        full_train: bool,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.full_train = full_train
        self.batch_size = train_loader.batch_size
        self.max_batch_size = 64
        self.current_accuracy = (0, 0)
        self.step = 0

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        data, labels = batch
        
        """
        FOR CHUNKING BATCHES OVER 64
        n_chunks = int(np.ceil(self.batch_size/self.max_batch_size))
        chunks = torch.chunk(data, n_chunks, dim=0)

        chunk_logits = []

        for chunk in chunks:
            chunk_out = self.model.forward(chunk.to(self.device))
            chunk_logits.append(chunk_out)

        logits = torch.cat(chunk_logits) 
        """
        logits = self.model.forward(data.to(self.device))
        preds = logits.detach().cpu().argmax(-1)

        step_results = {
            'logits': logits.cpu(),
            'loss': self.criterion(logits, labels.to(self.device)),
            'accuracy': self.compute_accuracy(preds, labels.cpu())
        }
        return step_results

    def _train_step(
        self,
        epoch: int,
        print_frequency: int = 100,
        log_frequency: int = 10
    ) -> Dict[str, Any]:
        self.model.train()

        train_results = {
            'loss': [],
            'accuracy': [],
            'preds': [],
            'labels': []
        }

        for batch in self.train_loader:# tqdm(
            # self.train_loader,
            # unit=" audio seq",
            # dynamic_ncols=True,
            # total=len(self.train_loader)
        # ):
            _, labels = batch

            self.optimizer.zero_grad()
            step_results = self._step(batch)

            loss = step_results['loss']
            loss.backward()

            self.optimizer.step()

            # print(f'loss: {loss.detach().cpu().item()}')

            preds = step_results['logits'].argmax(-1).numpy()
            train_results['loss'].append(loss.detach().cpu().item())
            train_results['preds'].extend(list(preds))
            train_results['labels'].extend(list(labels.numpy()))

            if ((self.step + 1) % log_frequency) == 0:
                self.log_metrics(
                    epoch, 
                    step_results['accuracy'], 
                    loss.detach().cpu().item()
                )
            if ((self.step + 1) % print_frequency) == 0:
                self.print_metrics(
                    epoch, 
                    step_results['accuracy'], 
                    loss.detach().cpu().item()
                )

            self.step += 1

        train_results['accuracy'] = self.compute_accuracy(
            np.array(train_results["labels"]), 
            np.array(train_results["preds"])
        )

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
        print_frequency: int = 100,
        log_frequency: int = 10,
        start_epoch: int = 0
    ):
        if self.full_train:
            for epoch in tqdm(
                range(start_epoch, epochs),
                unit=" epoch",
                dynamic_ncols=True
            ):

                train_results = self._train_step(
                    epoch,
                    print_frequency,
                    log_frequency
                )

                """
                list of losses and accuracies from 1 epoch over the whole dataset: 
                    len({loss, accuracy}) = len(dataloader)
                """
                # loss = train_results['loss']
                # accuracy = train_results['accuracy']

                self.summary_writer.add_scalar("epoch", epoch, self.step)
                if ((epoch + 1) % val_frequency) == 0:
                    self.validate(epoch)
        else:
            epoch = start_epoch
            epoch_since_last_increase = start_epoch
            max_accuracy = self.current_accuracy[0]
            while epoch_since_last_increase < 100:

                train_results = self._train_step(
                    epoch,
                    print_frequency,
                    log_frequency
                )

                self.summary_writer.add_scalar("epoch", epoch, self.step)
                if ((epoch + 1) % val_frequency) == 0:
                    self.validate(epoch)

                if self.current_accuracy[0] > max_accuracy:
                    max_accuracy = self.current_accuracy[0]
                    self.summary_writer.add_scalar("max accuracy", max_accuracy, self.step)
                    if max_accuracy >= 0.835:
                        epoch_since_last_increase = 16
                    else:
                        epoch_since_last_increase = 0
                else:
                    epoch_since_last_increase += 1
                
                epoch += 1


        # self.print_per_class_accuracy()


    def validate(self, epoch):

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

        if self.full_train:
            # save the model if the accuracy is greater than 85% # 0.862
            if accuracy > 0.862:
                self.save_model_params(f'{epoch}_{accuracy}')
        else:
            # 0.835
            if epoch % 5 == 0:
                if accuracy > self.current_accuracy[0]:
                    self.save_model_params(f'{epoch}_{accuracy}')

        if accuracy > self.current_accuracy[0]:
            self.current_accuracy = (accuracy, epoch)

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

    def compute_accuracy(
        self,
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
        incorrect_pred = {classname: {class_n: 0 for class_n in classes.keys()} for classname in classes.keys()}
        total_pred = {classname: 0 for classname in classes.keys()}

        self.model.eval()
        with torch.no_grad():
            for batch, labels in self.val_loader:

                logits = self.model(batch.to(self.device))
                preds = logits.argmax(-1).cpu().numpy()

                for label, pred in zip(labels.cpu().numpy(), preds):
                    if label == pred:
                        correct_pred[label] += 1
                    incorrect_pred[label][pred] += 1
                    total_pred[label] += 1
        
        for classname, correct in correct_pred.items():
            accuracy = 100 * float(correct) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f}%".format(classes[classname], accuracy))

    def log_metrics(self, epoch, accuracy, loss):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss)},
                self.step
        )

    def save_model_params(self, model_path: str):
        torch.save(self.model.state_dict(), f'./checkpoints/{model_path}.pth')

    def load_model_params(self, model_path: str):
        self.model.load_state_dict(torch.load(f'/checkpoints/{model_path}.pth', map_location=torch.device('cpu')))
