import os
import time
import pickle
import torch

from log_utils import log
import color.utils.progress as progress
import color.utils.utils as utils


def load_training_params(save_dir):
    full_path = os.path.join(save_dir, 'training_params.pickle')
    with open(full_path, 'rb') as x:
        return pickle.load(x)


class ModelTraining:

    def __init__(self, model, loss_fn, optimizer, dataset, **kwargs):
        self.params = {
            'num_epochs': 10,
            'curr_epoch': 1,
            'draw_plots': True,
            'show_progress': True,
            'do_cv': True,
            'use_cuda': True,
            'save_dir': None
        }
        utils.dict_update_existing(self.params, kwargs)
        if self.params['save_dir'] is None or os.path.isdir(self.params['save_dir']):
            raise Exception('Save directory is not specified or it already exists')

        # Validate
        self._validate()

        # Device
        self.device = torch.device('cuda' if self.params['use_cuda'] else 'cpu')

        # Model
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Dataset
        self.dataset = dataset

        # Training cache
        self.epoch_train_losses = []
        self.epoch_cv_losses = []
        self.epoch_durations = []

    def train(self):
        for epoch in range(self.params['curr_epoch'], self.params['curr_epoch'] + self.params['num_epochs']):

            # Progress bar
            progress_bar = (progress.ProgressBar(len(self.dataset.train_set) + len(self.dataset.cv_set),
                                                 status='Training epoch {}'.format(epoch)) if self.params[
                'show_progress'] else None)
            curr_progress = 0

            # Start epoch timer
            time_start = time.time()

            # Train
            loss_sum = 0
            for i, batch in enumerate(self.dataset.train_loader):
                self.model.zero_grad()
                loss = self.train_batch(*batch)
                loss_sum += loss
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                if progress_bar is not None:
                    curr_progress += 1
                    progress_bar.update(curr_progress)

            self.epoch_train_losses.append(
                float(loss_sum) / len(self.dataset.train_set) if len(self.dataset.train_set) > 0 else 0)

            # CV
            if self.params['do_cv']:
                with torch.no_grad():
                    loss_sum = 0
                    for i, batch in enumerate(self.dataset.cv_loader):
                        loss = self.cv_batch(*batch)
                        loss_sum += loss

                        # Update progress bar
                        if progress_bar is not None:
                            curr_progress += 1
                            progress_bar.update(curr_progress)

                    self.epoch_cv_losses.append(
                        float(loss_sum) / len(self.dataset.cv_set) if len(self.dataset.cv_set) > 0 else 0)

            # Compute epoch time
            time_end = time.time()
            self.epoch_durations.append(time_end - time_start)

            # Complete progress bar
            if progress_bar is not None:
                progress_bar.complete(status=self.epoch_results_message(epoch))

            # Plot losses
            if self.params['draw_plots']:
                self.draw_plots(epoch)

    def epoch_results_message(self, epoch):
        return 'Epoch {} | Time: {}s'.format(epoch, self.epoch_durations[epoch])

    def train_batch(self, *args):
        raise NotImplementedError()

    def cv_batch(self, *args):
        raise NotImplementedError()

    def draw_plots(self, epoch):
        raise NotImplementedError()

    def _validate(self):
        save_dir = self.params['save_dir']
        if os.path.isdir(save_dir):
            raise Exception('Save directory already exists: {}'.format(save_dir))

    def save(self):
        save_dir = self.params['save_dir']
        os.makedirs(save_dir)
        log.info('Saving params to "%s"', os.path.abspath(save_dir))

        # Save model
        if hasattr(self.model, 'save'):
            self.model.save(save_dir)
        else:
            log.warn('Model has no save method')

        self.dataset.save(save_dir)

        # Save training params
        params = {
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_cv_losses': self.epoch_cv_losses,
            'epoch_durations': self.epoch_durations,
        }
        params.update(self.params)
        full_path = os.path.join(save_dir, 'training_params.pickle')
        with open(full_path, 'wb') as x:
            pickle.dump(params, x)

        log.info('Save complete')
