import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class Prior_Embed(nn.Module):
    def __init__(self, 
                 dim, 
                 ch_in,
                 ):
        super(Prior_Embed, self).__init__()

        self.x_in = nn.Conv2d(ch_in, dim-1, kernel_size=3, stride=1, padding=1)
        self.h_in = nn.Identity()

        self.rcab = RCAB_Block(dim)
        self.rb = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, hx, prior):
        x = self.x_in(x)
        x = torch.cat([x, prior], 1)
        x = x + self.rb(x)
        
        hx = self.h_in(hx)
        hx = self.rcab(hx)
        return x, hx

class Conv_FFN(nn.Module):
    def __init__(self, 
                 dim, 
                 expansion_factor, 
                 LayerNorm_type = 'WithBias'):
        super(Conv_FFN, self).__init__()

        hidden_features = int(dim*expansion_factor)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.conv_forward = nn.Sequential(
                nn.Conv2d(dim, hidden_features, kernel_size = 1),
                nn.GELU(),
                nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,groups=hidden_features),
                nn.GELU(),
                nn.Conv2d(hidden_features, dim, kernel_size = 1)
        )

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv_backward = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,groups=hidden_features),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, kernel_size = 1)
        )

    def forward(self, x, hx):
        x = torch.cat([x, hx], 1)
        out_forward = self.norm1(x)
        out_forward = x + self.conv_forward(out_forward)

        out_backward = self.norm2(out_forward)
        out_backward = out_forward +  self.conv_backward(out_backward)
        return out_backward

class CA_Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 act = 'sigmoid', 
                 reduction=4):
        super(CA_Block,self).__init__()
        self.se_module=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size=1),
        )
        if act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Sigmoid()
    def forward(self,x):
        x = x*self.act(self.se_module(x))
        return x

class RCAB_Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 act = 'sigmoid', 
                 reduction=4):
        super(RCAB_Block,self).__init__()
        self.rcab=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            CA_Block(in_channels, act, reduction)
        )
    def forward(self,x):
        return x + self.rcab(x)

class Cross_Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 ch_in, 
                 num_heads, 
                 LayerNorm_type = 'WithBias'):
        super(Cross_Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)
        self.atten_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
        )

    def forward(self, x, hx):
        q = self.norm1(x)
        kv = self.norm2(hx)
        b,c,h,w = hx.shape

        q = self.q_dwconv(self.q(q)) 
        kv = self.kv_dwconv(self.kv(kv))
        k,v = kv.chunk(2, dim=1)

        p_emb = self.pos_emb(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) # b c c

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fin = self.atten_out(out) + p_emb + hx
        return fin
    
class BasicBlock(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 ch_in, 
                 num_heads,
                 expansion_factor, 
                 LayerNorm_type = 'WithBias'):
        super(BasicBlock, self).__init__()

        self.emebed = Prior_Embed(dim = feature_dim, 
                                      ch_in=ch_in)
        self.ca = Cross_Attention(dim = feature_dim,
                                  ch_in=ch_in,
                                  num_heads = num_heads,
                                  LayerNorm_type=LayerNorm_type)
        self.ffn = Conv_FFN(dim = feature_dim + ch_in,
                            expansion_factor=expansion_factor,
                            LayerNorm_type=LayerNorm_type)
    def forward(self, x, hidden_x, prior=None):
        xp, hidden_x = self.emebed(x, hidden_x, prior)
        hidden_x = self.ca(xp, hidden_x)
        out = self.ffn(x, hidden_x)
        return out

class ResBlk(nn.Module):
    def __init__(self,
                 *, 
                 in_channels, 
                 out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1, padding_mode = 'reflect')

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1, padding_mode = 'reflect')
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x+h

class Down(nn.Module):
    def __init__(self):
        super(Down,self).__init__()

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Up(nn.Module):
    def __init__(self):
        super(Up,self).__init__()
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return x        

class GPLM(nn.Module):
    def __init__(self, 
                 ch_in = 1, 
                 n_feats = 32):
        super(GPLM, self).__init__()
        self.n_feats=n_feats
        self.prior_dim = 1
        self.chs = [1,2,4]
        self.en_in = nn.Conv2d(ch_in, n_feats, kernel_size=3, padding=1, bias = False)
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
            nn.Conv2d(self.chs[-1]*n_feats, self.prior_dim*2, kernel_size=3, padding=1, padding_mode = 'reflect')
        )

        self.de_in = nn.Sequential(
            nn.Conv2d(self.prior_dim, self.chs[-1]*n_feats, kernel_size=3, padding=1, padding_mode = 'reflect'),
            ResBlk(in_channels = self.chs[-1]*n_feats, out_channels = self.chs[-1]*n_feats),
        )

        self.up = nn.ModuleList()
        for i in reversed(range(1, len(self.chs))):
            block = [
                Up(),
                ResBlk(in_channels=self.chs[i]*n_feats, out_channels=self.chs[i-1]*n_feats),
                ResBlk(in_channels=self.chs[i-1]*n_feats, out_channels=self.chs[i-1]*n_feats)
            ]
            self.up.append(nn.Sequential(*block))

        self.de_out = nn.Conv2d(n_feats, ch_in, kernel_size=3, padding=1, bias = False)

    def encode(self,input):
        out = self.en_in(input)
        for i in range(len(self.chs)-1):
            out = self.down[i](out)
        out = self.en_out(out)
        return out
    
    def decode(self,input):
        out = self.de_in(input)
        for i in range(len(self.chs)-1):
            out = self.up[i](out)
        out = self.de_out(out)
        return out
    
class CS_DUN(nn.Module):
    def __init__(self,
                 sr,
                 dim = 32,
                 block_size = 32,
                 in_channels = 1,
                 num_heads=1,
                 stages = 18,
                 expansion_factor = 4.0, 
                 LayerNorm_type = 'WithBias'):
        super(CS_DUN,self).__init__()
        self.block_size = block_size
        self.stages = stages


        xdim = int(block_size*block_size*in_channels)
        ydim = int(sr*xdim)

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(ydim, in_channels, block_size, block_size)), requires_grad=True)

        self.fe = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.transblocks = nn.ModuleList()
        self.weights = []
        for i in range(stages):
            self.weights.append(nn.Parameter(torch.tensor(0.5), requires_grad=True))
            self.transblocks.append(BasicBlock(feature_dim=dim,
                                               ch_in=in_channels,
                                               num_heads=num_heads,
                                               expansion_factor=expansion_factor,
                                               LayerNorm_type=LayerNorm_type))
        
    def sampling(self, inputs):
        y = F.conv2d(inputs, self.Phi, stride = self.block_size, padding = 0, bias=None)
        return y

    def initial(self, y):
        x = F.conv_transpose2d(y, self.Phi, stride=self.block_size) 
        return x
    
    def cs_init(self, inputs):
        return self.initial(self.sampling(inputs))
            
    def forward(self, ir, pri = None):
        x = ir
        hidden_x = self.fe(x)
        for i in range(self.stages):
            r = x - self.weights[i]*(self.initial(self.sampling(x))-ir)
            output = self.transblocks[i](r, hidden_x, pri)
            x = output[:, :1, :, :]
            hidden_x = output[:, 1:, :, :]
        final_x = x
        return final_x