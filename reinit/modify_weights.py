import torch
import numpy as np


def resample_weights(pruning_hparams, model, mask):
    '''Resample weights that are pruned from the winning ticket'''

    # Determine which layers can be pruned.
    prunable_tensors = set(model.prunable_layer_names)
    if pruning_hparams.pruning_layers_to_ignore:
        prunable_tensors -= set(
            pruning_hparams.pruning_layers_to_ignore.split(','))

    # Get the model weights.
    weights = {
        k: v.clone().cpu().detach().numpy()[mask[k] == 1]
        for k, v in model.state_dict().items() if k in prunable_tensors
    }
    print(model['conv.weight'])
    with torch.no_grad():
        for k, v in weights.items():
            model[k][mask[k] == 0] = np.random.choice(
                v, len(model[k][mask[k] == 0]))

    return model
