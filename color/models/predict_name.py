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


class NamePredictorBase(nn.Module):

    def __init__(self):
        super(NamePredictorBase, self).__init__()
        self.params = {
            'name': None,
            'emb_dim': 50,
            'color_dim': 3,  # But don't change
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'lr_decay': (2, 0.95),
        }

    def save(self, save_dir):
        # Model weights
        weights_path = os.path.join(save_dir, 'model_weights.pt')
        torch.save(self.state_dict(), weights_path)

        # Hyperparams
        params_path = os.path.join(save_dir, 'model_params.pickle')
        with open(params_path, 'wb') as x:
            pickle.dump(self.params, x)

    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.params['lr'], momentum=self.params['momentum'],
                               weight_decay=self.params['weight_decay'])

    def get_loss_fn(self):
        def _loss_fn(emb, emb_pred, stop_weight):
            loss = F.mse_loss(emb, emb_pred)
            return loss + (F.mse_loss(emb[-1], emb_pred[-1]) / stop_weight)

        return _loss_fn

    def forward(self, rgb, emb):
        rgb2emb_out = self.rgb2emb(rgb)
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape)
        emb = torch.cat((rgb2emb_out, emb[:-1]), dim=0)

        seq_nn = self.lstm if hasattr(self, 'lstm') else self.rnn
        seq_nn_out = seq_nn(emb)

        return self.hidden2emb(seq_nn_out[0])

    def gen_name(self, rgb):
        rgb2emb_out = self.rgb2emb(rgb)
        rgb2emb_out = rgb2emb_out.reshape(1, *rgb2emb_out.shape)
        seq_nn = self.lstm if hasattr(self, 'lstm') else self.rnn
        seq_nn_out, seq_nn_state = seq_nn(rgb2emb_out)

        while True:
            emb = yield self.hidden2emb(seq_nn_out[0])
            seq_nn_out, seq_nn_state = seq_nn(emb, seq_nn_state)


class NamePredictorLSTM(NamePredictorBase):

    def __init__(self, **kwargs):
        super(NamePredictorLSTM, self).__init__()
        self.params.update({
            'name': 'LSTM',
            'hidden_dim': 50,
            'num_layers': 2,
            'dropout': 0,
        })
        utils.dict_update_existing(self.params, kwargs)

        self.linear = nn.Linear(self.params['color_dim'], self.params['emb_dim'])
        self.lstm = nn.LSTM(self.params['emb_dim'], self.params['hidden_dim'],
                            num_layers=self.params['num_layers'], dropout=self.params['dropout'])


class NamePredictorRNN(NamePredictorBase):

    def __init__(self, **kwargs):
        super(NamePredictorRNN, self).__init__()
        self.params.update({
            'name': 'RNN',
            'hidden_dim': 50,
            'num_layers': 2,
            'dropout': 0,
            'nonlinearity': 'relu'
        })
        utils.dict_update_existing(self.params, kwargs)

        self.rgb2emb = nn.Linear(self.params['color_dim'], self.params['emb_dim'])
        self.rnn = nn.RNN(self.params['emb_dim'], self.params['hidden_dim'],
                          num_layers=self.params['num_layers'], dropout=self.params['dropout'])
        self.hidden2emb = nn.Linear(self.params['hidden_dim'], self.params['emb_dim'])


class NamePredictionTraining(training.ModelTraining):

    def __init__(self, model, loss_fn, optimizer, dataset, **kwargs):
        super(NamePredictionTraining, self).__init__(model, loss_fn, optimizer, dataset, **kwargs)
        self.plotter = plotter.MSEPlotter(
            name=None if model.params['name'] is None else model.params['name'],
            env=self.params['plotter_env']
        ) if self.params['draw_plots'] else None

    def epoch_results_message(self, epoch):
        return 'Epoch {} | Train Loss: {:2f} | CV Loss: {:2f} | Time: {}s'.format(
            epoch, self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], self.epoch_durations[epoch - 1])

    def train_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.train_set))

    def cv_batch(self, rgb, embedding, _):
        embedding_preds = self.model(rgb, embedding)
        return self.loss_fn(embedding, embedding_preds, len(self.dataset.cv_set))

    def draw_plots(self, epoch):
        self.plotter.plot(self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], epoch)


def train():
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
