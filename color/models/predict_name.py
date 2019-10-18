import torch
import torch.nn as nn
import torch.nn.functional as F

import color.data.dataset as color_dataset
import color.training as training
import color.utils.utils as utils
import color.models.model_utils as model_utils
from log_utils import log


class NamePredictorBaseModel(model_utils.BaseModel):
    """Abstract base class for all name prediction models, encapsulating model, loss function and optimizer"""

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
            'loss_fn': 'MSE',  # One of (MSE, MSE_stop_word)
        }

    def get_loss_fn(self):
        if self.params['loss_fn'] == 'MSE':
            return nn.MSELoss()
        else:
            # Since color name sequences are small, we can prevent the model from overfitting to the stop word vector by
            # reducing the weight of the stop word vectors. Here, loss from stop words is scaled down by their frequency
            def _loss_fn(emb, emb_pred, num_stop_words):
                loss = F.mse_loss(emb, emb_pred)
                return loss + (F.mse_loss(emb[-1], emb_pred[-1]) / num_stop_words)

            return _loss_fn

    def forward(self, *inputs):
        raise NotImplementedError


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
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape)  # Reshape to a single sequence embedding

        # Prepare LSTM inputs
        # First time step input is the resized RGB vector, wheras the last word's embedding is not fed as input
        emb = torch.cat((rgb2emb_out, emb[:-1]), dim=0)

        # Process color name with RNN
        rnn_out, _ = self.rnn(emb)

        # Convert RNN output to output embedding
        return self.hidden2emb(rnn_out)

    def gen_name(self, rgb):
        """This function can be used as a co-routine to generate color name predictions"""

        # Linear layer resizes RGB vector
        rgb2emb_out = self.rgb2emb(rgb)
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape)  # Reshape to a single sequence embedding

        # First time step - process transformed RGB vector
        rnn_out, rnn_state = self.rnn(rgb2emb_out)

        # The following co-routine works towards generating color names
        # At each iteration a new approximate word embedding is predicted and returned
        # An exact word embedding is received and the sequence processing continues
        while True:
            # Process RNN output with final linear layer and return the output embedding and RNN state
            # The received input is a tuples of computed word embedding and RNN state
            emb, rnn_state = yield self.hidden2emb(rnn_out), rnn_state

            # Process embedding with RNN
            rnn_out, rnn_state = self.rnn(emb, rnn_state)


class NamePredictionTraining(training.ModelTraining):
    """Training workflow for all color name prediction models"""

    def train_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.train_set))

    def cv_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.cv_set))


def train():
    """Sample configurations"""

    # Create dataset
    emb_dim = 50
    dataset = color_dataset.Dataset(dataset='big', emb_len=emb_dim)

    # Initialize model
    model = NamePredictorSequenceModel(name='sample-lstm-model', model_type='LSTM')
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
    trainer = NamePredictionTraining(
        model, loss_fn, optimizer, dataset,
        num_epochs=10,
        seq_len_first=True,
        use_cuda=True,
        save_dir=save_dir
    )

    # Train and save
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    try:
        train()
    except Exception:
        log.exception('Training failed')
        raise
