import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import color.data.dataset as color_dataset
import color.training as training
import color.utils.plotter as plotter
import color.utils.utils as utils
from log_utils import log


def load_model_params(save_dir, include_weights=True):
    # Hyperparams
    params_path = os.path.join(save_dir, 'model_params.pickle')
    with open(params_path, 'rb') as x:
        params = pickle.load(x)

    if not include_weights:
        return params

    # Model weights
    weights_path = os.path.join(save_dir, 'model_weights.pt')
    weights = torch.load(weights_path)
    return weights, params


class NamePredictorBaseModel(nn.Module):
    """Base class for all name prediction models, encapsulating model, loss function and optimizer"""

    def __init__(self):
        super(NamePredictorBaseModel, self).__init__()

        # Default model parameters
        self.params = {
            'name': None,  # Model name
            'emb_dim': 50,  # Embedding dimensions
            'color_dim': 3,  # Color space dimensions, But don't change
            'lr': 0.1,  # Learning rate
            'momentum': 0.9,  # Momentum
            'weight_decay': 0.0001,  # L2 regularization
            'lr_decay': (2, 0.95),  # Learning rate decay
            'loss_fn': 'MSE', # One of (MSE, MSE_stop_word)
        }

    def save(self, save_dir):
        """Save the model and parameters"""

        # Model weights
        weights_path = os.path.join(save_dir, 'model_weights.pt')
        torch.save(self.state_dict(), weights_path)

        # Hyper-parameters
        params_path = os.path.join(save_dir, 'model_params.pickle')
        with open(params_path, 'wb') as x:
            pickle.dump(self.params, x)

    def get_optimizer(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.params['lr'],
            momentum=self.params['momentum'],
            weight_decay=self.params['weight_decay']
        )

    def get_loss_fn(self):
        if self.params['loss_fn'] == 'MSE':
            return nn.MSELoss()
        else:
            # As the color name sequences are small, we can prevent the model from overfitting to the stop word vector
            # by reducing the weight of the stop word vectors. Here, loss from stop words is scaled down by their frequency
            def _loss_fn(emb, emb_pred, num_stop_words):
                loss = F.mse_loss(emb, emb_pred)
                return loss + (F.mse_loss(emb[-1], emb_pred[-1]) / num_stop_words)
            return _loss_fn


class NamePredictorSequenceModel(NamePredictorBaseModel):
    """RNN model for color name predictions"""

    def __init__(self, **kwargs):
        super(NamePredictorSequenceModel, self).__init__()

        # RNN model defaults
        self.params.update({
            'name': 'rnn-name-predictor',  # Model name
            'model_type': 'RNN',  # One of (RNN, LSTM)
            'hidden_dim': 50,  # No of neurons in each hidden layer
            'num_layers': 2,  # No of hidden layers
            'dropout': 0,  # Dropout factor
            'nonlinearity': 'relu'  # Only for RNN: Activation function: tanh, relu
        })
        utils.dict_update_existing(self.params, kwargs)

        # A linear layer converts RGB value to a vector of the same length as the embeddings
        self.rgb2emb = nn.Linear(self.params['color_dim'], self.params['emb_dim'])

        # RNN Layer
        if self.params['model_type'] == 'RNN':
            self.rnn = nn.RNN(
                self.params['emb_dim'], self.params['hidden_dim'], num_layers=self.params['num_layers'],
                dropout=self.params['dropout'], nonlinearity=self.params['nonlinearity']
            )
        else:
            self.rnn = nn.LSTM(
                self.params['emb_dim'], self.params['hidden_dim'],
                num_layers=self.params['num_layers'], dropout=self.params['dropout']
            )

        # Final linear layer converts RNN output to output embeddings
        self.hidden2emb = nn.Linear(self.params['hidden_dim'], self.params['emb_dim'])

    def forward(self, rgb, emb):
        # Linear layer resizes rgb vector
        rgb2emb_out = self.rgb2emb(rgb)
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape) # Reshape to a single sequence embedding

        # Prepare LSTM inputs
        # First time step input is the resized RGB vector, wheras the last word's embedding is not fed as input
        emb = torch.cat((rgb2emb_out, emb[:-1]), dim=0)

        # Process color name with RNN
        rnn_out, _ = self.rnn(emb)

        # Convert RNN output to output embedding
        return self.hidden2emb(rnn_out)

    def gen_name(self, rgb):
        """This function can be used to generate multiple color name predictions"""

        # Linear layer resizes RGB vector
        rgb2emb_out = self.rgb2emb(rgb)
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape) # Reshape to a single sequence embedding
        
        # Run first time step and store it as single item list which will later be used to store multiple outputs at each time step
        outputs  = [self.rnn(emb)]

        # The following coroutine works towards generating color names
        # At each iteration new approximate word embeddings are predicted and returned
        # Exact word embeddings are received and the sequence processing continues
        while True:

            # Process all RNN outputs on final linear layer and return the output embeddings and RNN states
            # The received inputs is a list of tuples of word embeddings and RNN states
            inputs = yield [(self.hidden2emb(rnn_out), rnn_state) for rnn_out, rnn_state in outputs]

            # Process each embedding with RNN
            outputs = [self.rnn(emb, rnn_state) for emb, rnn_state in inputs]


class NamePredictionTraining(training.ModelTraining):
    """Training workflow for all color name prediction models"""

    def __init__(self, model, loss_fn, optimizer, dataset, **kwargs):
        super(NamePredictionTraining, self).__init__(model, loss_fn, optimizer, dataset, **kwargs)

        # Create a plotter
        self.plotter = plotter.MSEPlotter(
            name=None if model.params['name'] is None else model.params['name'],
            env=self.params['plotter_env']
        ) if self.params['draw_plots'] else None

    def epoch_results_message(self, epoch):
        return 'Epoch {} | Train Loss: {:2f} | CV Loss: {:2f} | Time: {}s'.format(
            epoch, self.epoch_train_losses[epoch - 1],
            self.epoch_cv_losses[epoch - 1],
            self.epoch_durations[epoch - 1]
        )

    def train_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.train_set))

    def cv_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.cv_set))

    def draw_plots(self, epoch):
        """Plot training and cross-validation losses"""
        self.plotter.plot(self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], epoch)


def train():
    """Test configuration"""

    emb_dim = 200
    dataset = color_dataset.Dataset(max_words=None, dataset='big', emb_len=emb_dim, normalize_rgb=True,
                                    pad_len=None, batch_size=1, use_cuda=True, var_seq_len=True)

    model_key = 'lstm'
    if model_key == 'lstm':
        model = NamePredictorLSTM(emb_dim=emb_dim, hidden_dim=emb_dim, num_layers=1, dropout=0,
                                  lr=0.1, momentum=0.9, weight_decay=0.0001,
                                  name='LSTM_namer')
        print(model)
    else:
        raise Exception('Invalid model key: {}'.format(model_key))

    log.debug('Model: %s', model)
    log.debug('Model Params: %d', utils.get_trainable_params(model))
    loss_fn = model.get_loss_fn()
    optimizer = model.get_optimizer()

    save_dir = utils.get_rel_path(__file__, '..', '..', 'trained_models',
                                  '{}_{}'.format(model.params['name'], utils.get_unique_key()))
    trainer = NamePredictionTraining(
        model, loss_fn, optimizer, dataset,
        num_epochs=20, draw_plots=True, show_progress=True,
        use_cuda=True, save_dir=save_dir
    )
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    try:
        train()
    except Exception:
        log.exception('Training failed')
        raise
