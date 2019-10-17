import torch
import torch.nn as nn

import color.data.dataset as color_dataset
import color.training as training
import color.utils.utils as utils
import color.models.model_utils as model_utils
from log_utils import log


class ColorPredictorBaseModel(model_utils.BaseModel):
    """Abstract base class for all color prediction models, encapsulating model, loss function and optimizer"""

    def __init__(self):
        super(ColorPredictorBaseModel, self).__init__()

        # Default model parameters
        self.params = {
            'name': None,  # Model name
            'emb_dim': 50,  # Embedding dimensions
            'color_dim': 3,  # Color space dimensions, But don't change
            'lr': 0.1,  # Learning rate
            'momentum': 0.9,  # Momentum
            'weight_decay': 0.0001,  # L2 regularization
            'lr_decay': (2, 0.95),  # Learning rate decay
        }

    def forward(self, *inputs):
        raise NotImplementedError


class ColorPredictorSequenceModel(ColorPredictorBaseModel):
    """RNN model for color predictions"""

    def __init__(self, **kwargs):
        super(ColorPredictorSequenceModel, self).__init__()

        # RNN model defaults
        self.params.update({
            'name': 'rnn-color-predictor',  # Model name
            'model_type': 'RNN',  # One of (RNN, LSTM)
            'hidden_dim': 50,  # No of neurons in each hidden layer
            'num_layers': 2,  # No of hidden layers
            'dropout': 0,  # Dropout factor
            'nonlinearity': 'relu'  # Only for RNN: Activation function: tanh, relu
        })
        utils.dict_update_existing(self.params, kwargs)

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

        # Final linear layer converts the last time step output of RNN to an RGB value
        self.linear = nn.Linear(self.params['hidden_dim'], self.params['color_dim'])

    def forward(self, emb):
        # Process color name embeddings with RNN
        rnn_out, _ = self.rnn(emb)

        # Extract last time step output
        final_out = rnn_out[-1]

        # Convert last time step output of RNN to an RGB value using linear layer
        linear_out = self.linear(final_out)

        # Scale output between 0 and 1 with sigmoid
        return torch.sigmoid(linear_out)


class ColorPredictorCNNModel(ColorPredictorBaseModel):
    """CNN model for color predictions"""

    def __init__(self, **kwargs):
        super(ColorPredictorCNNModel, self).__init__()

        # RNN model defaults
        self.params.update({
            'name': 'cnn-color-predictor',  # Model name
            'max_words': 3,  # Fixed size of each word embedding sequence
            'num_conv_layers': 2,  # No of convolution layers
            'conv_kernel_size': 3,  # Convolution kernel size
            'conv_stride': 1,  # Convolution stride
            'pool_kernel_size': 2,  # Pooling kernel size
            'pool_stride': 2,  # Pooling stride
            'num_linear_layers': 1,  # No of trailing linear layers
            'linear_size_reduce': 2,  # Factor by which size of trailing linear layers are successively reduced
        })
        utils.dict_update_existing(self.params, kwargs)

        # Define 1D Convolution + Pooling blocks

        self.conv_layers = []
        self.pool_layers = []

        inp_len = self.params['emb_dim']  # Size of input (here embedding length)
        inp_channels = self.params['max_words']  # Channels (here words) in previous (here first) layer
        out_channels = self.params['max_words'] * 2  # Channels in current layer

        for l in range(self.params['num_conv_layers']):
            # 1D Convolution Layer
            conv = nn.Conv1d(
                inp_channels, out_channels,
                kernel_size=self.params['conv_kernel_size'],
                stride=self.params['conv_stride']
            )

            # Layer needs to be explicitly registered as it is not a direct member of the model instance
            self.add_module('cnn{}'.format(l + 1), conv)
            self.conv_layers.append(conv)

            # Update input dimensions and channels for pooling layer
            inp_len = int((inp_len - self.params['conv_kernel_size']) / self.params['conv_stride']) + 1
            inp_channels, out_channels = inp_channels * 2, out_channels * 2
            assert inp_len > 1  # Invalid architecture, input dimension too small

            # Pooling Layer
            pool = nn.MaxPool1d(self.params['pool_kernel_size'], stride=self.params['pool_stride'])

            # Layer needs to be explicitly registered as it is not a direct member of the model instance
            self.add_module('pool{}'.format(l + 1), pool)
            self.pool_layers.append(pool)

            # Update input dimensions for next convolution layer
            inp_len = int((inp_len - self.params['pool_kernel_size']) / self.params['pool_stride']) + 1
            assert inp_len > 1  # Invalid architecture, input dimension too small

        # Define Linear Layers

        self.linear_layers = []

        in_feature = inp_len * inp_channels  # Input size of current layer
        out_feature = round(in_feature / self.params['linear_size_reduce'])  # Output size of current layer

        for l in range(self.params['num_linear_layers']):
            assert out_feature >= 3  # Invalid architecture, input dimension too small

            # Linear Layer
            linear = nn.Linear(in_feature, out_feature)

            # Layer needs to be explicitly registered as it is not a direct member of the model instance
            self.add_module('linear{}'.format(l + 1), linear)
            self.linear_layers.append(linear)

            # Update input/output dimensions for next linear layer
            in_feature, out_feature = out_feature, round(out_feature / self.params['linear_size_reduce'])
        out_feature = in_feature

        # Final linear layer for converting linear output to RGB values
        self.final_linear = nn.Linear(out_feature, self.params['color_dim'])

    def forward(self, emb):

        # Execute all convolution and pooling layers
        pool_out = emb
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = conv(pool_out)
            pool_out = pool(conv_out)

        # Execute all linear layers
        linear_in = pool_out.view(pool_out.shape[0], -1)
        for linear in self.linear_layers:
            linear_in = linear(linear_in)

        # Convert linear layer output to RGB value and scale with sigmoid
        linear_out = self.final_linear(linear_in)
        return torch.sigmoid(linear_out)


class ColorPredictionTraining(training.ModelTraining):
    """Training workflow for all color prediction models"""

    def train_batch(self, rgb, embedding, _):
        rgb_preds = self.model(embedding)
        return self.loss_fn(rgb, rgb_preds)

    def cv_batch(self, rgb, embedding, _):
        rgb_preds = self.model(embedding)
        return self.loss_fn(rgb, rgb_preds)


def train_rnn():
    """Sample configurations"""

    # Create dataset
    emb_dim = 50
    dataset = color_dataset.Dataset(dataset='big', emb_len=emb_dim)

    # Initialize model
    model = ColorPredictorSequenceModel(name='sample-lstm-model', model_type='LSTM')
    log.debug('Model: %s', model)
    log.debug('Model Params: %d', utils.get_trainable_params(model))
    loss_fn = model.get_loss_fn()
    optimizer = model.get_optimizer()

    # Save directory
    save_dir = utils.get_rel_path(
        __file__, '..', '..', 'trained_models',  # Base directory
        '{}_{}'.format(model.params['name'], utils.get_unique_key())  # Model directory
    )

    # Initialize model training
    trainer = ColorPredictionTraining(
        model, loss_fn, optimizer, dataset,
        num_epochs=10,
        seq_len_first=True,
        use_cuda=True,
        save_dir=save_dir
    )

    # Train and save
    trainer.train()
    trainer.save()


def train_cnn():
    """Sample configurations"""

    # Create dataset
    emb_dim = 50
    dataset = color_dataset.Dataset(
        dataset='big',
        emb_len=emb_dim,
        normalize_rgb=True,
        max_words=3,
        pad_len=3,
    )

    # Initialize model
    model = ColorPredictorCNNModel(max_words=3)
    log.debug('Model: %s', model)
    log.debug('Model Params: %d', utils.get_trainable_params(model))
    loss_fn = model.get_loss_fn()
    optimizer = model.get_optimizer()

    # Save directory
    save_dir = utils.get_rel_path(
        __file__, '..', '..', 'trained_models',  # Base directory
        '{}_{}'.format(model.params['name'], utils.get_unique_key())  # Model directory
    )

    # Initialize model training
    trainer = ColorPredictionTraining(
        model, loss_fn, optimizer, dataset,
        num_epochs=10,
        use_cuda=True,
        save_dir=save_dir
    )

    # Train and save
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    try:
        train_rnn()
        # train_cnn()
    except Exception:
        log.exception('Training failed')
        raise
