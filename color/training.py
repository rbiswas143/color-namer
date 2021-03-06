import os
import time
import pickle
import torch

from log_utils import log
import color.utils.progress as progress
import color.utils.utils as utils
import color.utils.plotter as plotter
import color.data.dataset as colors_dataset


def load_training_params(save_dir):
    """Load model training parameter and losses from a model directory"""
    full_path = os.path.join(save_dir, 'training_params.pickle')
    with open(full_path, 'rb') as x:
        return pickle.load(x)


class ModelTraining:
    """Abstract base class that captures the model training and cross validation workflow"""

    def __init__(self, model, loss_fn, optimizer, dataset, **kwargs):

        # Training parameter default values
        self.params = {
            'num_epochs': 10,  # Epoch to train
            'curr_epoch': 1,  # Epoch number to start with, in case resuming (resuming not supported yet)
            'draw_plots': True,  # Draw plots or not
            'show_progress': True,  # Render command line progress bar or not
            'do_cv': True,  # Do cross-validation or not
            'seq_len_first': False,  # Sequence models expect sequence length before batch along the axes
            'use_cuda': True,  # Use GPU or not
            'save_dir': None,  # Existing directory where the model will be saved
            'plotter_env': None,  # Vizdom environment in which to render the plots
        }
        utils.dict_update_existing(self.params, kwargs)

        # Validations
        self._validate()

        # Device: CPU or GPU
        self.device = torch.device('cuda' if self.params['use_cuda'] else 'cpu')

        # Model
        self.model = model.double().to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Dataset and loaders
        self.dataset = dataset
        self.train_loader = colors_dataset.DataLoader(
            dataset.train_set,
            seq_len_first=self.params['seq_len_first'],
            use_cuda = self.params['use_cuda']
        )
        self.cv_loader = colors_dataset.DataLoader(
            dataset.cv_set,
            seq_len_first=self.params['seq_len_first'],
            use_cuda = self.params['use_cuda']
        )

        # Create a plotter
        self.plotter = plotter.MSEPlotter(
            name=None if model.params['name'] is None else model.params['name'],
            env=self.params['plotter_env']
        ) if self.params['draw_plots'] else None

        # Cache
        self.epoch_train_losses = []
        self.epoch_cv_losses = []
        self.epoch_durations = []

    def train(self):
        """Training loop for training and cross-validating the model"""

        for epoch in range(self.params['curr_epoch'], self.params['curr_epoch'] + self.params['num_epochs']):

            # Progress bar
            progress_bar = progress.ProgressBar(
                len(self.dataset.train_set) + len(self.dataset.cv_set),
                status='Training epoch {}'.format(epoch)) if self.params['show_progress'] else None
            curr_progress = 0

            # Start epoch timer
            time_start = time.time()

            # Train
            loss_sum = 0
            for i, batch in enumerate(self.train_loader):
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
                    for i, batch in enumerate(self.cv_loader):
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
        return 'Epoch {} | Train Loss: {:2f} | CV Loss: {:2f} | Time: {}s'.format(
            epoch,
            self.epoch_train_losses[epoch - 1],
            self.epoch_cv_losses[epoch - 1],
            self.epoch_durations[epoch - 1]
        )

    def train_batch(self, *args):
        """Override to define the computations for training a batch of data and return the loss"""
        raise NotImplementedError()

    def cv_batch(self, *args):
        """Override to define the computations for cross-validating a batch of data and return the loss"""
        raise NotImplementedError()

    def draw_plots(self, epoch):
        """Plot training and cross-validation losses"""
        self.plotter.plot(self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], epoch)

    def _validate(self):
        # Save directory must not exist, which otherwise implies that the model is already persisted
        if self.params['save_dir'] is None or os.path.isdir(self.params['save_dir']):
            raise Exception('Save directory is not specified or it already exists. Save directory:{}'
                            .format(self.params['save_dir']))

    def save(self):
        """Save the model, dataset and training parameters"""

        # Create save directory
        save_dir = self.params['save_dir']
        os.makedirs(save_dir)
        log.info('Saving params to "%s"', os.path.abspath(save_dir))

        # Save model
        if hasattr(self.model, 'save'):
            self.model.save(save_dir)
        else:
            log.warn('Model has no save method')

        # Save dataset
        self.dataset.save(save_dir)

        # Save training params. Losses are being added to the parameters
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
