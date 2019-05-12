import os
import pickle
import torch
import torch.nn as nn

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


class ColorPredictorBase(nn.Module):

    def __init__(self):
        super(ColorPredictorBase, self).__init__()
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
        return nn.MSELoss()


class ColorPredictorLSTM(ColorPredictorBase):

    def __init__(self, **kwargs):
        super(ColorPredictorLSTM, self).__init__()
        self.params.update({
            'name': 'LSTM',
            'hidden_dim': 50,
            'num_layers': 2,
            'dropout': 0,
        })
        utils.dict_update_existing(self.params, kwargs)

        self.lstm = nn.LSTM(self.params['emb_dim'], self.params['hidden_dim'],
                            num_layers=self.params['num_layers'], dropout=self.params['dropout'])
        self.linear = nn.Linear(self.params['hidden_dim'], self.params['color_dim'])

    def forward(self, emb):
        emb = emb.view(emb.shape[1], 1, -1)
        lstm_out, _ = self.lstm(emb)
        final_out = lstm_out[-1]
        linear_out = self.linear(final_out)
        return torch.sigmoid(linear_out)


class ColorPredictorRNN(ColorPredictorBase):

    def __init__(self, **kwargs):
        super(ColorPredictorRNN, self).__init__()
        self.params.update({
            'name': 'RNN',
            'hidden_dim': 50,
            'num_layers': 2,
            'dropout': 0,
            'nonlinearity': 'relu'
        })
        utils.dict_update_existing(self.params, kwargs)

        self.rnn = nn.RNN(self.params['emb_dim'], self.params['hidden_dim'], num_layers=self.params['num_layers'],
                          dropout=self.params['dropout'], nonlinearity=self.params['nonlinearity'])
        self.linear = nn.Linear(self.params['hidden_dim'], self.params['color_dim'])

    def forward(self, emb):
        emb = emb.view(emb.shape[1], emb.shape[0], -1)
        rnn_out, _ = self.rnn(emb)
        final_out = rnn_out[-1]
        linear_out = self.linear(final_out)
        return torch.sigmoid(linear_out)


class ColorPredictorCNN(ColorPredictorBase):

    def __init__(self, **kwargs):
        super(ColorPredictorCNN, self).__init__()
        self.params.update({
            'name': 'CNN',
            'max_words': 3,
            'num_conv_layers': 2,
            'conv_kernel_size': 3,
            'conv_stride': 1,
            'pool_kernel_size': 2,
            'pool_stride': 2,
            'num_linear_layers': 1,
            'linear_size_reduce': 2
        })
        utils.dict_update_existing(self.params, kwargs)

        # Define Conv + Pool blocks
        self.conv_layers = []
        self.pool_layers = []
        inp_len, inp_channels, out_channels = (
            self.params['emb_dim'], self.params['max_words'], self.params['max_words'] * 2)
        for l in range(self.params['num_conv_layers']):
            conv = nn.Conv1d(inp_channels, out_channels,
                             kernel_size=self.params['conv_kernel_size'],
                             stride=self.params['conv_stride'])
            self.add_module('cnn{}'.format(l + 1), conv)
            self.conv_layers.append(conv)
            inp_len = int((inp_len - self.params['conv_kernel_size']) / self.params['conv_stride']) + 1
            inp_channels, out_channels = inp_channels * 2, out_channels * 2
            assert inp_len > 1

            pool = nn.MaxPool1d(self.params['pool_kernel_size'], stride=self.params['pool_stride'])
            self.add_module('pool{}'.format(l + 1), pool)
            self.pool_layers.append(pool)
            inp_len = int((inp_len - self.params['pool_kernel_size']) / self.params['pool_stride']) + 1
            assert inp_len > 1

        # Define linear layers
        self.linear_layers = []
        in_feature = inp_len * inp_channels
        out_feature = round(in_feature / self.params['linear_size_reduce'])
        for l in range(self.params['num_linear_layers']):
            assert out_feature >= 3
            linear = nn.Linear(in_feature, out_feature)
            self.add_module('linear{}'.format(l + 1), linear)
            self.linear_layers.append(linear)
            in_feature, out_feature = out_feature, round(out_feature / self.params['linear_size_reduce'])
        out_feature = in_feature

        # Final linear layer
        self.final_linear = nn.Linear(out_feature, self.params['color_dim'])

    def forward(self, emb):
        pool_out = emb
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = conv(pool_out)
            pool_out = pool(conv_out)
        linear_in = pool_out.view(pool_out.shape[0], -1)
        for linear in self.linear_layers:
            linear_in = linear(linear_in)
        linear_out = self.final_linear(linear_in)
        return torch.sigmoid(linear_out)


class ColorPredictionTraining(training.ModelTraining):

    def __init__(self, model, loss_fn, optimizer, dataset, **kwargs):
        super(ColorPredictionTraining, self).__init__(model, loss_fn, optimizer, dataset, **kwargs)
        self.plotter = (plotter.MSEPlotter(name=None if model.params['name'] is None else model.params['name'])
                        if self.params['draw_plots'] else None)

    def epoch_results_message(self, epoch):
        return 'Epoch {} | Train Loss: {:2f} | CV Loss: {:2f} | Time: {}s'.format(
            epoch, self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], self.epoch_durations[epoch - 1])

    def train_batch(self, rgb, embedding, _):
        rgb_preds = self.model(embedding)
        return self.loss_fn(rgb, rgb_preds)

    def cv_batch(self, rgb, embedding, _):
        rgb_preds = self.model(embedding)
        return self.loss_fn(rgb, rgb_preds)

    def draw_plots(self, epoch):
        self.plotter.plot(self.epoch_train_losses[epoch - 1], self.epoch_cv_losses[epoch - 1], epoch)


def train():
    emb_dim = 200
    dataset = color_dataset.Dataset(max_words=None, dataset='big', emb_len=emb_dim, normalize_rgb=True,
                                    pad_len=None, batch_size=1, use_cuda=True)

    model_key = 'lstm'
    if model_key == 'lstm':
        model = ColorPredictorLSTM(emb_dim=emb_dim, hidden_dim=emb_dim, num_layers=1, dropout=0,
                                   lr=0.1, momentum=0.9, weight_decay=0.0001)
    elif model_key == 'rnn':
        model = ColorPredictorRNN(emb_dim=emb_dim, hidden_dim=emb_dim, num_layers=2, dropout=0, nonlinearity='relu',
                                  lr=0.1, momentum=0.9, weight_decay=0.0001)
    elif model_key == 'cnn':
        model = ColorPredictorCNN(name='cnn_test', max_words=3,
                                  num_conv_layers=1, conv_kernel_size=7, conv_stride=1,
                                  pool_kernel_size=9, pool_stride=1,
                                  num_linear_layers=1, linear_size_reduce=1,
                                  lr=0.5, weight_decay=1e-05, lr_decay=(1, 0.9337531782245498), momentum=0.8)
        print(model)
    else:
        raise Exception('Invalid model key: {}'.format(model_key))

    log.debug('Model: %s', model)
    log.debug('Model Params: %d', utils.get_trainable_params(model))
    loss_fn = model.get_loss_fn()
    optimizer = model.get_optimizer()

    save_dir = utils.get_rel_path(__file__, '..', '..', 'trained_models',
                                  '{}_{}'.format(model.params['name'], utils.get_unique_key()))
    trainer = ColorPredictionTraining(
        model, loss_fn, optimizer, dataset,
        num_epochs=20, draw_plots=False, show_progress=True,
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
