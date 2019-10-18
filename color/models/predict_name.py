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
                loss = F.mse_loss(emb[:-1], emb_pred[:-1])
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

        # Check the loss function type and pass stop word weight
        args = [embedding, embedding_preds]
        if self.model.params['loss_fn'] == 'MSE_stop_word':
            args.append(len(self.dataset.train_set))

        return self.loss_fn(*args)

    def cv_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)

        # Check the loss function type and pass stop word weight
        args = [embedding, embedding_preds]
        if self.model.params['loss_fn'] == 'MSE_stop_word':
            args.append(len(self.dataset.cv_set))

        return self.loss_fn(*args)


class NamePrediction:

    def __init__(self, words, last_emb, similarity):
        self.words = words
        self.last_emb = last_emb
        self.similarity = similarity
        self.emb_pred = None
        self.nn_state = None


def predict_names(model, dataset, rgb, num_names=3, max_len=6, stop_word=False, normalize_rgb=True):
    """Predicts a number of names for the specified color RGB value"""

    # Covert RGB value to to input tensor and optionally normalize it
    rgb = torch.DoubleTensor(rgb).view(1, 3)
    if normalize_rgb:
        rgb = rgb / 256

    with torch.no_grad():

        # Create a generator instance
        name_generator = model.gen_name(rgb)

        # Compute norm of each embedding vector
        embs = torch.DoubleTensor(dataset.embeddings)
        embs_mag = torch.sqrt(torch.sum(embs * embs, dim=1)).reshape(-1)

        # A list of selected word embeddings that will be sent to the generator
        # Before the first time step, no predictions exist
        predictions = [NamePrediction([], None, 0)]
        predictions[0].emb_pred, predictions[0].nn_state = next(name_generator)

        for i in range(max_len):

            pred_candidates = []
            for pred in predictions:

                emb_pred_mag = torch.sqrt(torch.sum(pred.emb_pred * pred.emb_pred))
                emb_dot = torch.mm(embs, pred.emb_pred.view(-1, 1)).view(-1)
                embs_sim = emb_dot / (embs_mag * emb_pred_mag)

                _, top_idx = torch.topk(embs_sim, num_names)
                for idx in top_idx.tolist():
                    curr_words = pred.words[:]
                    curr_words.append(dataset.vocab[idx])

                    last_emb = embs[idx].view(1, 1, -1)
                    sim = (embs_sim[idx] + (len(pred.words) * pred.similarity)) / len(curr_words)
                    pred_candidates.append(NamePrediction(curr_words, last_emb, sim))

            pred_candidates.sort(key=lambda pred: pred.similarity, reverse=True)
            predictions = pred_candidates[:num_names]

            for pred in predictions:
                pred.emb_pred, pred.nn_state = name_generator.send((pred.last_emb, pred.nn_state))

        return predictions


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
