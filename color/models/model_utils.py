import os
import pickle
import torch
import torch.nn as nn


def load_model_params(save_dir, include_weights=True):
    """Load model weights and hyper-parameters"""

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


class BaseModel(nn.Module):
    """Abstract base class for all models"""

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
        return nn.MSELoss()

    def forward(self, *inputs):
        raise NotImplementedError
