from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from roma.utils.utils import get_gt_warp, get_gt_warp_amd, get_gt_warp_symmetric
import wandb
import roma
import math
import numpy as np

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]

        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)

        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        
        return losses

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps["flow_pre_delta"],
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
            b, h, w, d = flow_pre_delta.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
        return tot_loss

class RobustLossesMy(RobustLosses):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        merge = 'False',
        depth_err = 0.05,
        ua_weight = 1.,
        kl_weight = 0,
        annealing_step = 1,
        lamb = 1.,
        var_weight = 1.,
    ):
        super().__init__(robust, center_coords, scale_normalize, ce_weight, local_loss,
                         local_dist, local_largest_scale, smooth_mask, depth_interpolation_mode,
                          mask_depth_loss, relative_depth_error_threshold, alpha, c)
        self.merge = merge
        self.depth_err = depth_err
        self.ua_weight = ua_weight
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step
        self.lamb = lamb
        self.var_weight = var_weight
    
    def kl_divergence(self, alpha, num_classes):
        #ones = torch.ones([1, num_classes], dtype=torch.float32).cuda()
        ones = torch.ones_like(alpha)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl


    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + self.var_weight * loglikelihood_var
        return loglikelihood


    def mse_loss(self, y, alpha, num_classes):
        loglikelihood = self.loglikelihood_loss(y, alpha)

        annealing_coef = torch.min(
                        torch.tensor(1.0, dtype=torch.float32),
                        torch.tensor(roma.GLOBAL_STEP / self.annealing_step, dtype=torch.float32),
                        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)

        
        return loglikelihood + self.kl_weight * kl_div

    def edl_mse_loss(self, output, target, num_classes):
        evidence = F.relu(output)
        alpha = evidence + self.lamb
        loss = torch.mean(
            self.mse_loss(target, alpha, num_classes)
        )
        return loss

    def ua_loss(self, output, target, scale):
        #print('x2 shape is', x2.shape)
        #print('la shape is', la.shape)
        
        target = target[:, None]
        target = torch.cat((target, 1 - target), dim = 1)
        
        loss = self.edl_mse_loss(output, target, 2)

        losses = {
            f"ua_loss_{scale}": loss,
        }

        
        return losses

    def forward(self, corresps, batch):
        #print(corresps.keys())
        scales = list(corresps.keys())
        scales.sort()
        scales = scales[::-1]

        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        
        sizes = {scale: corresps[scale]["certainty"].shape[-2:] for scale in scales}
        ests = []
        for scale in scales:
            #print('scale is', scale)
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, \
            scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow, \
            scale_la, scale_alpha, scale_beta = (
                scale_corresps["certainty"],
                scale_corresps["flow_pre_delta"],
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                scale_corresps.get("la"),
                scale_corresps.get("alpha"),
                scale_corresps.get("beta"),
            )

            flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
            b, h, w, d = flow_pre_delta.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                #print('scale_gm_cls is not none') this one!
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                #print('scale_gm_flow is not none')
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                #print('delta_cls is not none')
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                #print('delta_cls is none, using regression_loss') this one!
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
            
            # use different depth error threshold for el
            gt_warp, gt_mask = get_gt_warp(batch["im_A_depth"], batch["im_B_depth"], batch["T_1to2"],
                                  batch["K1"], batch["K2"], H=h, W=w, 
                                  relative_depth_error_threshold = self.depth_err)

            if self.merge == 'False':
               ua_loss = self.ua_loss(scale_certainty, gt_mask, scale)
               tot_loss += self.ua_weight * ua_loss[f"ua_loss_{scale}"]

               if scale_gm_certainty is not None:
                  #print('scale_gm_certainty is not None for scale', scale)
                  scale_gm = str(scale) + '_gm'
                  ua_loss = self.ua_loss(scale_gm_certainty, gt_mask, scale = scale_gm)
                  tot_loss += self.ua_weight * ua_loss[f"ua_loss_{scale_gm}"]
            elif self.merge == 'True':
                if scale != 1:
                    scale_certainty = F.interpolate(
                            scale_certainty,
                            size=sizes[1],
                            align_corners=False,
                            mode="bilinear",
                        )
                    scale_la = F.interpolate(
                            scale_la,
                            size=sizes[1],
                            align_corners=False,
                            mode="bilinear",
                        )
                    
                    scale_alpha = F.interpolate(
                            scale_alpha,
                            size=sizes[1],
                            align_corners=False,
                            mode="bilinear",
                        )

                    scale_beta = F.interpolate(
                            scale_beta,
                            size=sizes[1],
                            align_corners=False,
                            mode="bilinear",
                        )
                ests.append([scale_certainty, scale_la, scale_alpha, scale_beta])
        
        if self.merge == 'True':
            flow, la, alpha, beta = self.combine_uncertainty(ests)
            scale = 'all'
            ua_loss = self.ua_loss(scale_certainty[:, 0], gt_mask, scale, la, alpha, beta)
            tot_loss += self.ua_weight * ua_loss[f"ua_loss_{scale}"]
        return tot_loss

class RobustLossesAMD(RobustLosses):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__(robust, center_coords, scale_normalize, ce_weight, local_loss,
                         local_dist, local_largest_scale, smooth_mask, depth_interpolation_mode,
                          mask_depth_loss, relative_depth_error_threshold, alpha, c)
        
    
    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps["flow_pre_delta"],
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
            b, h, w, d = flow_pre_delta.shape
            gt_warp, gt_prob = get_gt_warp_amd(                
            batch["im_A"],
            batch["im_B"],
            batch["H"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()

        return tot_loss

class RobustLossesSymmetric(RobustLosses):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__(robust, center_coords, scale_normalize, ce_weight, local_loss,
                         local_dist, local_largest_scale, smooth_mask, depth_interpolation_mode,
                          mask_depth_loss, relative_depth_error_threshold, alpha, c)
        
    
    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps["flow_pre_delta"],
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )

            #print('scale_certainty.shape is', scale_certainty.shape)
            #print('scale_gm_cls.shape is', scale_gm_cls.shape)
            #print('flow.shape is', flow.shape)
            flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
            b, h, w, d = flow_pre_delta.shape
            gt_warp, gt_prob = get_gt_warp_symmetric(                
            batch["im_A"],
            batch["im_B"],
            batch["H"],
            batch["H_inv"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()

        return tot_loss

class SelfSupLossL2(nn.Module):
    def __init__(self
        
    ):
        super().__init__()
        return  
    
    def forward(self, im_ori, im_pred, weight):
        loss = ((im_ori - im_pred).norm(dim = 1, keepdim = True) * weight).mean()

        losses = {
            f"ssl_loss": loss,
        }

        
        return loss
        