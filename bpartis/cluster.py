"""
## Class to cluster embeddings and produce instance segmentation maps.
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


class Cluster:

    def __init__(self, n_sigma=1, device='cuda'):
        self.n_sigma = n_sigma
        self.device = device
        xm = torch.linspace(0, 1, 512).view(1, 1, -1).expand(1, 512, 512)
        ym = torch.linspace(0, 1, 512).view(1, -1, 1).expand(1, 512, 512)
        xym = torch.cat((xm, ym), 0)
        self.xym = xym.to(self.device)

    def cluster(self, prediction, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+self.n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w
       
        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1
        mask = (seed_map > 0.5).byte()

        if mask.sum() > 128:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).byte().to(self.device)

            while(unclustered.sum() > 128):

                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed+1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed+1]*10)
                dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
                                                        center, 2)*s, 0, keepdim=True))

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append(
                            {'mask': instance_mask.squeeze()*255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances

    def monte_carlo_cluster(self, prediction, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+self.n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w
       
        instance_map = torch.zeros(height, width).byte()
        instances = []

        monte_carlo_semantic_map = torch.zeros(height, width).to(self.device)

        count = 1
        mask = (seed_map > 0.5).byte()

        if mask.sum() > 128:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).byte().to(self.device)

            while(unclustered.sum() > 128):

                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed+1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed+1]*10)
                dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
                                                        center, 2)*s, 0, keepdim=True))
                dist_image = torch.exp(-1*torch.sum(torch.pow(spatial_emb - center.view(2, 1, 1), 2)*s.view(2, 1, 1), 0, keepdim=True)).squeeze()
                dist_image = dist_image * (dist_image > 0.5)
                monte_carlo_semantic_map += dist_image

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append(
                            {'mask': instance_mask.squeeze()*255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        
        return instance_map, monte_carlo_semantic_map, instances