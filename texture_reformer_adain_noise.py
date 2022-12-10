# borrow from https://github.com/EndyWon/Texture-Reformer/blob/main/transfer.py
import torch
import torch.nn as nn

# original vgg-19 model
from model import Encoder1, Encoder2, Encoder3, Encoder4, Encoder5
from model import Decoder1, Decoder2, Decoder3, Decoder4, Decoder5


class Reformer(nn.Module):
    def __init__(self, args):
        super(Reformer, self).__init__()
        self.args = args

        # load pre-trained models
        self.e1 = Encoder1(args.e1); self.d1 = Decoder1(args.d1)
        self.e2 = Encoder2(args.e2); self.d2 = Decoder2(args.d2)
        self.e3 = Encoder3(args.e3); self.d3 = Decoder3(args.d3)
        self.e4 = Encoder4(args.e4); self.d4 = Decoder4(args.d4)
        self.e5 = Encoder5(args.e5); self.d5 = Decoder5(args.d5)
    
    # calculate channel-wise mean and standard deviation
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    # project to standardize the data and dispel the domain gap
    def project(self, feat):
        size = feat.size()
        mean, std = self.calc_mean_std(feat)
        projected_feat = (feat - mean.expand(size)) / std.expand(size)
        return projected_feat

    # dAdaIN
    def distorted_adaptive_instance_normalization(self, cF, sF, noise_mu=1., noise_sigma=1.):
        assert (cF.size()[:2] == sF.size()[:2])
        size = cF.size()
        style_mean, style_std = self.calc_mean_std(sF)
 
        style_mean = style_mean * noise_mu
        style_std = style_std * noise_sigma
        
        content_mean, content_std = self.calc_mean_std(cF)
        
        normalized_feat = (cF - content_mean.expand(
            size)) / content_std.expand(size)
        
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


    # Semantic-Guided Texture Warping (SGTW) module
    def SGTW(self, sF, sF_fused, cF_fused, patch_size):
        # if patch_size = 0, set global view
        if patch_size == 0:
            patch_size = min([cF_fused.shape[2], cF_fused.shape[3], sF_fused.shape[2], sF_fused.shape[3]]) - 1

        # extract original style patches
        style_patches = sF.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        style_patches = style_patches.permute(0, 2, 3, 1, 4, 5)
        style_patches = style_patches.reshape(-1, *style_patches.shape[-3:])

        # extract fused style patches
        fused_style_patches = sF_fused.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        fused_style_patches = fused_style_patches.permute(0, 2, 3, 1, 4, 5)
        fused_style_patches = fused_style_patches.reshape(-1, *fused_style_patches.shape[-3:])

        # normalize fused style patches
        norm = torch.norm(fused_style_patches.reshape(fused_style_patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
        normalized_fused_style_patches = fused_style_patches/(norm + 1e-7) 
    
        # determine the closest-matching fused style patch for each fused content patch
        coordinate = torch.nn.functional.conv2d(cF_fused, normalized_fused_style_patches)
        
        # binarize the scores
        one_hots = torch.zeros_like(coordinate)
        one_hots.scatter_(1, coordinate.argmax(dim=1, keepdim=True), 1)

        # use the original style patches to reconstruct transformed feature
        deconv_out = torch.nn.functional.conv_transpose2d(one_hots, style_patches)

        # average the overlapped patches
        overlap = torch.nn.functional.conv_transpose2d(one_hots, torch.ones_like(style_patches))
        deconv_out = deconv_out / overlap

        return deconv_out

    # View-Specific Texture Reformation (VSTR) operation with *concatenated* semantic guidance
    def VSTR_concat(self, cF, sF, cF_sem, sF_sem, patch_size, alpha, semantic_weight):
        # project style feature and content feature
        sF1 = self.project(sF)
        cF1 = self.project(cF)

        # fuse with semantic maps
        sF_fused = torch.cat((sF1, semantic_weight*sF_sem), 1)
        cF_fused = torch.cat((cF1, semantic_weight*cF_sem), 1)
        
        # Semantic-Guided Texture Warping (SGTW) module
        targetFeature = self.SGTW(sF, sF_fused, cF_fused, patch_size)

        # blend transformed feature with the content feature
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF

        return csF
 

    # dAdaIN transformation
    def dadain(self, cF, sF, alpha, noise_mu, noise_sigma):
        targetFeature = self.distorted_adaptive_instance_normalization(cF, sF, noise_mu, noise_sigma)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        return csF
