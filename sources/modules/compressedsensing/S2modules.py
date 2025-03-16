# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
from sources.modules.compressedsensing.S1modules import ResBlk, Down

class ENCODES2(nn.Module):
    def __init__(self, ch_in = 1, n_feats = 32):
        super(ENCODES2, self).__init__()
        self.n_feats=n_feats

        self.en_in = nn.Conv2d(ch_in, n_feats, kernel_size=3, padding=1, bias = False)
        self.prior_dim = 1
        self.chs = [1,2,4]
        self.down = nn.ModuleList()
        for i in range(len(self.chs)-1):
            block = [
                ResBlk(in_channels=self.chs[i]*n_feats, out_channels=self.chs[i]*n_feats),
                ResBlk(in_channels=self.chs[i]*n_feats, out_channels=self.chs[i+1]*n_feats),
                Down()
            ]
            self.down.append(nn.Sequential(*block))
        self.en_out = nn.Sequential(
            ResBlk(in_channels = self.chs[-1]*n_feats, out_channels = self.chs[-1]*n_feats),
            nn.Conv2d(self.chs[-1]*n_feats, self.prior_dim, kernel_size=3, padding=1, padding_mode = 'reflect')
        )

    def forward(self,input):
        out = self.en_in(input)
        for i in range(len(self.chs)-1):
            out = self.down[i](out)
        out = self.en_out(out)
        return out

class denoise_cnn(nn.Module):
    def __init__(self, n_feats = 32, n_denoise_res = 5,timesteps=5):
        super(denoise_cnn, self).__init__()
        prior_dim = 1
        chs = [1,2,4]
        dim = (int)(n_feats * chs[-1])
        self.size = 64//(2**(len(chs)-1))
        self.max_period=timesteps*10
        mlp = [
            nn.Conv2d(prior_dim*2+1, dim, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect')
        ]
        for _ in range(n_denoise_res):
            mlp.append(ResBlk(in_channels = dim, out_channels= dim))
        self.resmlp = nn.Sequential(*mlp)
        self.cout = nn.Conv2d(dim, prior_dim, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect')

    def forward(self,x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1,1,1,1).repeat(1,1,self.size,self.size)
        out = torch.cat([x,c,t],dim=1)
        out = self.resmlp(out)
        out = self.cout(out)
        return out