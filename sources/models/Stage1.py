import torch
import pytorch_lightning as pl
from sources.modules.compressedsensing.S1modules import CS_DUN,GPLM
from sources.modules.distributions.distributions import DiagonalGaussianDistribution

class DCSNet(pl.LightningModule):
    """main class"""
    def __init__(self,
                 sr, 
                 hidden_dim = 32,
                 block_size = 32,
                 in_channels = 1,
                 stages = 18,
                 kl_weight = 0,
                 ckpt_path=None):
        super().__init__()
        self.kl_weight = kl_weight
        self.gplm = GPLM(ch_in=in_channels, n_feats=hidden_dim//2)
        self.deeprecon = CS_DUN(sr=sr,
                                dim=hidden_dim,
                                block_size=block_size,
                                in_channels=in_channels,
                                stages=stages)

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            self.restarted_from_ckpt = True

    @torch.no_grad()
    def cs_sample(self, x):
        return self.deeprecon.sampling(x)
    
    @torch.no_grad()
    def get_ir(self, y):
        return self.deeprecon.initial(y)

    @torch.no_grad()
    def init_cs(self, x):
        return self.deeprecon.cs_init(x)
    
    @torch.no_grad()
    def get_encode(self,inputs):
        moments = self.gplm.encode(inputs)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    @torch.no_grad()
    def get_decode(self,inputs):
        return self.gplm.decode(inputs)
    
    def recon(self,ir, pri):
        pri_f = self.gplm.decode(pri)
        return self.deeprecon(ir, pri_f)
    
    def apply_model(self, img, train = True):
        ir = self.deeprecon.cs_init(img)
        moments = self.gplm.encode(img)
        posterior = DiagonalGaussianDistribution(moments)
        if train:
            pri = posterior.sample()
        else:
            pri = posterior.mode()
        out = self.recon(ir,pri)
        return out, posterior
