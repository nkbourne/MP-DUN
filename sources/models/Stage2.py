import torch
import pytorch_lightning as pl
from einops import rearrange
from sources.util import instantiate_from_config
from sources.modules.ldm.ddpm import DDPM
from sources.modules.compressedsensing.S2modules import denoise_cnn, ENCODES2

class DCSNet(pl.LightningModule):
    """main class"""
    def __init__(self,
                 s1_config,
                 dim,
                 n_denoise_res = 5,
                 timesteps = 3,
                 in_channels = 1,
                 block_size = 32,
                 patch_size = 64,
                 device = 'cuda',
                 linear_start= 0.1,
                 linear_end= 0.99):
        super().__init__()
        self.block_size = block_size
        self.patch_size = patch_size
        condition = ENCODES2(ch_in=in_channels, n_feats=dim//2)
        denoise = denoise_cnn(n_feats=dim, n_denoise_res=n_denoise_res, timesteps=timesteps)
        self.diffusion = DDPM(denoise=denoise, condition=condition, linear_start= linear_start,
                            linear_end= linear_end, timesteps = timesteps)
        self.instantiate_S1_stage(s1_config)

    def instantiate_S1_stage(self, config):
        model = instantiate_from_config(config)
        self.s1_model = model
        for param in self.s1_model.gplm.parameters():
            param.requires_grad = False
    
    def apply_model(self, img, IPR_target):
        ir = self.s1_model.init_cs(img)
        IPRS2, _ = self.diffusion(ir.detach(), IPR_target)
        out = self.s1_model.recon(ir, IPRS2)
        return out, IPRS2

    @torch.no_grad()
    def sampling(self, img):
        y = self.s1_model.cs_sample(img)
        return y
    
    @torch.no_grad()  
    def implement(self, img, stride=1):
        bs = img.size()[0]
        kernel_size = self.patch_size//self.block_size
        y = self.sampling(img)
        fold, unfold, normalization, weighting = self.get_fold_unfold(y,stride=stride)
        y = unfold(y)
        y = y.view((bs, -1, kernel_size,  kernel_size, y.shape[-1]))
        ir_list = [self.s1_model.get_ir(y[:, :, :, :, i]) for i in range(y.shape[-1])]
        IPRS2_list = [self.diffusion(ir_list[i]) for i in range(len(ir_list))]
        reconout_list = []
        for i in range(len(ir_list)):
            rec = self.s1_model.recon(ir_list[i], IPRS2_list[i])
            reconout_list.append(rec)

        out = torch.stack(reconout_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
        out = out * weighting
        out = out.view((out.shape[0], -1, out.shape[-1]))
        out = fold(out)
        out = out / normalization 
        return out


    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, 0.01, 0.5)
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
        return weighting

    def get_fold_unfold(self, y, stride=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = y.shape
        kernel_size = self.patch_size//self.block_size

        Ly = (h - kernel_size) // stride + 1
        Lx = (w - kernel_size) // stride + 1

        fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        unfold = torch.nn.Unfold(**fold_params)

        fold_params2 = dict(kernel_size=kernel_size * self.block_size, dilation=1, padding=0, stride=stride * self.block_size)
        fold = torch.nn.Fold(output_size=(y.shape[2] * self.block_size, y.shape[3] * self.block_size), **fold_params2)

        weighting = self.get_weighting(kernel_size * self.block_size, kernel_size * self.block_size, Ly, Lx, y.device).to(y.dtype)
        
        normalization = fold(weighting).view(1, 1, h * self.block_size, w * self.block_size)  # normalizes the overlap
        weighting = weighting.view((1, 1, kernel_size * self.block_size, kernel_size * self.block_size, Ly * Lx))
        return fold, unfold, normalization, weighting