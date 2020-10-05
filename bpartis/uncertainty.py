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

def monte_carlo_predict(model, image, n_samples=50, device='cuda'):
    h, w = image.shape[-2:]
    cluster = Cluster(n_sigma=2, h=h, w=w, device=device)
    model.eval()
    enable_eval_dropout(model)

    # get monte carlo model samples
    mc_outputs = []
    mc_seed_maps = []
    for i in range(n_samples):
        output = model(image).detach()
        seed_map = torch.sigmoid(output[0, -1]).unsqueeze(0)
        mc_outputs.append(output)
        mc_seed_maps.append(seed_map)

    mc_outputs = torch.cat(mc_outputs, dim=0)
    mc_seed_maps = torch.cat(mc_seed_maps, dim=0)

    # MC prediction
    mc_prediction, _ = cluster.cluster(mc_outputs.mean(dim=0))

    # Uncertainty
    total = predictive_entropy(mc_seed_maps)
    aleatoric = expected_entropy(mc_seed_maps)
    epistemic = total - aleatoric

    return mc_prediction, epistemic

def uncertainty_filtering(prediction, uncertainty, t=0.0125):

    filtered_pred = torch.zeros_like(prediction)

    for inst_id in torch.unique(prediction):
        if inst_id == 0:
            continue
        inst_mask = prediction == inst_id
        inst_uncertainty = torch.mean(uncertainty[inst_mask])
        if inst_uncertainty < t:
            filtered_pred[inst_mask] = torch.max(filtered_pred) + 1

    return filtered_pred