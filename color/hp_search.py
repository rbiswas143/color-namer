import os
import pickle

import color.models as models
import color.data.dataset as color_dataset
import color.utils.utils as utils
from log_utils import log

default_config = {
    'hp_name': 'test_hp',
    'model_key': 'predict_color_lstm',
    'model_params': [],
    'training_params': {},
    'dataset_params': {},
}

hp_dir_base = utils.get_rel_path(__file__, '..', 'trained_models', 'hp')


def run_models(hp_config=default_config):
    dataset = color_dataset.Dataset(**hp_config['dataset_params'])
    Model, Trainer = models.get_model(hp_config['model_key'])
    for i, model_params in enumerate(hp_config['model_params']):
        log.info('Training model %d of %d', i + 1, len(hp_config['model_params']))
        model = Model(**model_params)
        loss_fn = model.get_loss_fn()
        optimizer = model.get_optimizer()
        log.debug('Model: %s', model)
        log.debug('Model Params: %d', utils.get_trainable_params(model))

        hp_dir = os.path.join(hp_dir_base, hp_config['hp_name'])
        hp_config['training_params']['save_dir'] = os.path.join(hp_dir, model_params['name'])

        trainer = Trainer(model, loss_fn, optimizer, dataset, **hp_config['training_params'])

        try:
            trainer.train()
            log.info('Model training complete')
        except:
            log.exception('Error executing model no %d', i + 1)
            log.debug('Config: %s', model_params)
        finally:
            trainer.save()


def load_model_configs(config_path):
    with open(config_path, 'rb') as x:
        return pickle.load(x)


if __name__ == '__main__':
    hp_name = 'predict_color_lstm_r2'
    config_path = os.path.join(hp_dir_base, hp_name, 'hp_config.pickle')
    hp_config = load_model_configs(config_path)
    run_models(hp_config)
