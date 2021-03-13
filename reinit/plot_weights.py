import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_weigtht_dist(mask, model, level):
    # Determine which layers can be pruned.
    prunable_tensors = set(model.prunable_layer_names)

    # Get the model weights.
    all_weights = {
        k: v.clone().cpu().detach().numpy()
        for k, v in model.state_dict().items() if k in prunable_tensors
    }

    # Get unpruned weights.
    unpruned_weights = {k: v[mask[k] == 1] for k, v in all_weights.items()}

    base_dir = os.path.join("figures/", "level-" + str(level))
    os.mkdir(base_dir)

    for layer_name, layer_weights in unpruned_weights.items():
        sns.distplot(layer_weights)
        plt.title(layer_name)
        save_layer_name = layer_name.replace(".", "-")
        save_path = os.path.join(base_dir, save_layer_name)
        plt.savefig(save_path)
        plt.close()
