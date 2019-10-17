import os
import pickle

import color.models as models
import color.data.dataset as color_dataset
import color.utils.utils as utils
from log_utils import log


def run_models(hp_config, hp_dir_base):
    """
    Trains all models as specified in the hyper-parameter tuning configuration
    Tip: Use Ctrl+C to terminate a poorly performing model and continue to the next
    """

    # Dataset and partitions will be common for all the models
    dataset = color_dataset.Dataset(**hp_config.params['dataset_params'])

    # Fetch the model and training classes
    model_class, training_class = models.get_model(**hp_config.params['model_key'])

    # Train models
    for i, model_params in enumerate(hp_config.params['model_params']):

        # Initialize model
        log.info('Training model %d of %d', i + 1, len(hp_config.params['model_params']))
        model = model_class(**model_params)
        loss_fn = model.get_loss_fn()
        optimizer = model.get_optimizer()
        log.debug('Model: %s', model)
        log.debug('Model Params: %d', utils.get_trainable_params(model))

        # Compute save directory for current model
        hp_dir = os.path.join(hp_dir_base, hp_config.params['hp_name'])
        hp_config.params['training_params']['save_dir'] = os.path.join(hp_dir, model_params['name'])

        # Initialize model training
        # A separate Visdom environment is created for the current run
        model_training = training_class(
            model, loss_fn, optimizer, dataset,
            plotter_env=hp_config.params['hp_name'],
            **hp_config.params['training_params']
        )

        try:
            # Train current model
            model_training.train()
            log.info('Model training complete')
        except:
            # Log exception and proceed to train next model
            log.exception('Error executing model no %d', i + 1)
            log.debug('Config: %s', model_params)
        finally:
            # Save model anyway. Finished, terminated, failed models will all be persisted
            model_training.save()

    log.info('Hyper-parameter search complete')


def _load_model_configs(config_path):
    with open(config_path, 'rb') as x:
        return pickle.load(x)


class HPConfig:
    """Configuration class for Hyper-parameter Search"""

    def __init__(self, **params):
        # Default configuration
        self.params = {
            'hp_name': '',
            'model_key': '',
            'model_params': [],
            'training_params': {},
            'dataset_params': {},
        }
        utils.dict_update_existing(self.params, params)


if __name__ == '__main__':
    # Hyper-parameter search data is stored under trained_models/hp
    # Supply a unique name to the current process
    hp_dir_base = utils.get_rel_path(__file__, '..', 'trained_models', 'hp')
    run_name = 'predict_name_rnn'

    # Load configuration (generated using notebook)
    config_path = os.path.join(hp_dir_base, run_name, 'hp_config.pickle')
    hp_config = HPConfig(**_load_model_configs(config_path))

    # Run all models
    run_models(hp_config, hp_dir_base)
