"""
## This file contains the code necessary to compute uncertainties
## via Monte Carlo sampling and Mutual Information.
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Batuhan Yildirim, 2020, BPartIS
-----
"""

import torch
from cluster import Cluster


def enable_eval_dropout(model):
    for module in model.modules():
        if 'Dropout' in type(module).__name__:
            module.train()

def entropy(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0-eps)
    return -1.0*((p*torch.log(p)) + ((1.0-p)*(torch.log(1.0-p))))

def expected_entropy(mc_preds):
    return torch.mean(entropy(mc_preds), dim=0)

def predictive_entropy(mc_preds):
    return entropy(torch.mean(mc_preds, dim=0))

def monte_carlo_predict(model, image, n_samples=30, device='cuda'):
    h, w = image.shape[-2:]
    cluster = Cluster(n_sigma=2, h=h, w=w, device=device)
    model.eval()
    enable_eval_dropout(model)

    # get monte carlo model samples
    mc_outputs = []
    for i in range(n_samples):
        output = model(image).detach()
        mc_outputs.append(output)
    mc_outputs = torch.cat(mc_outputs, dim=0)

    # get semantic segmentations of monte carlo samples
    semantic_predictions = []
    for i in range(n_samples):
        prediction, mc_sem_map, _ = cluster.monte_carlo_cluster(mc_outputs[i])
        semantic_predictions.append((prediction > 0.0).float())
        # semantic_predictions.append(mc_sem_map)

    semantic_predictions = torch.stack(semantic_predictions, dim=0)
    total = predictive_entropy(semantic_predictions)
    aleatoric = expected_entropy(semantic_predictions)
    epistemic = total - aleatoric
    epistemic = torch.clamp(epistemic, 0.0, 0.7)

    # get instance prediction of mean mc_outputs
    mean_mc_output = torch.mean(mc_outputs, dim=0)
    mc_instance_prediction, _ = cluster.cluster(mean_mc_output)

    return mc_instance_prediction, epistemic

def uncertainty_filtering(prediction, uncertainty, t=0.15):

    filtered_pred = torch.zeros_like(prediction)

    for inst_id in torch.unique(prediction):
        if inst_id == 0:
            continue
        inst_mask = prediction == inst_id
        inst_uncertainty = torch.mean(uncertainty[inst_mask])
        if inst_uncertainty < t:
            filtered_pred[inst_mask] = torch.max(filtered_pred) + 1

    return filtered_pred