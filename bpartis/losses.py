"""
## Spatial Embeddings loss function
## (much of this code was adapted from https://github.com/davyneven/SpatialEmbeddings credit to Davy Neven).
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
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from utils.losses import lovasz_hinge


class SpatialEmbLoss(nn.Module):

    def __init__(self, to_center=False, n_sigma=2, foreground_weight=1):
        super().__init__()
        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))

        # coordinate map
        xm = torch.linspace(0, 1, 512).view(
            1, 1, -1).expand(1, 512, 512)
        ym = torch.linspace(0, 1, 512).view(
            1, -1, 1).expand(1, 512, 512)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, height, width = prediction.size(0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
        mean_iou = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(
                prediction[b, 2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)   # 1 x h x w

                # calculate center of attraction
                center = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # calculate var loss before exp (smoothness loss I think)
                var_loss = var_loss + \
                    torch.mean(
                        torch.pow(sigma_in - s.detach(), 2))

                s = torch.exp(s*10)

                # calculate gaussian
                dist = torch.exp(-1*torch.sum(
                    torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                    lovasz_hinge(dist*2-1, in_mask)

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou:
                    mean_iou += calculate_iou(dist > 0.5, in_mask) / len(instance_ids)

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b+1)
        mean_iou = mean_iou / (b+1)

        return loss + prediction.sum()*0, mean_iou


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou

class ReconstructionLoss(nn.Module):

    def __init__(self, alpha=0.84):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True)

    def forward(self, x, x_prime):
        return (1-self.alpha)*self.l1(x, x_prime) + self.alpha*self.ms_ssim(x, x_prime) 