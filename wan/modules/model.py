# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, b_, s_, n_, d_, mask_token=None, ids_restore=None, ids_keep=None, rand_num_img=None):
    if rand_num_img!=None and rand_num_img<0.4:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)
            #print(freqs_i.shape,freqs[0][:f].view(f, 1, 1, -1).shape,"ejd0qhd0ah0dhwd0qwah")
            #zzzz
            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        
        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()
    else:
        if ids_restore!=None:
            x = x.view(b_,s_,n_*d_)

            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x.view(b_,ids_restore.shape[1],n_,d_)

        n, c = x.size(2), x.size(3) // 2

        # split freqs
        #freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        seq_len = x.shape[1]
        x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[0, seq_len:]])
        output.append(x_i)

        # for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        #     seq_len = f * h * w

        #     # precompute multipliers
        #     x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
        #         seq_len, n, -1, 2))
        #     freqs_i = torch.cat([
        #         freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        #         freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        #         freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        #     ],
        #                         dim=-1).reshape(seq_len, 1, -1)
        #     #print(freqs_i.shape,x_i.shape,x[i, seq_len:].shape,"0qh0q09sqj0jhq09jh2weq0jh")    
        #     #torch.Size([28560, 1, 64]) torch.Size([28560, 40, 64]) torch.Size([0, 40, 128]) 0qh0q09sqj0jhq09jh2weq0jh

        #     # apply rotary embedding
        #     x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        #     x_i = torch.cat([x_i, x[i, seq_len:]])

        #     # append to collection
        #     output.append(x_i)
        if ids_keep!=None:
            output = torch.stack(output).float()
            output = output.view(b_,ids_restore.shape[1],n_*d_)
            output = torch.gather(
                output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, n_*d_))
            output = output.view(b_,s_,n_,d_)
            return output

        return torch.stack(output).float()



import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_stride, h_stride, w_stride):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv3d(in_ch, 256, kernel_size=(t_stride,h_stride,w_stride), stride=(t_stride,h_stride,w_stride) )
        self.norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv3d(256, out_ch, kernel_size=(1,1,1))
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.norm2 = nn.LayerNorm(out_ch)

        # 捷径路径
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or t_stride !=1 or h_stride !=1 or w_stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=(t_stride, h_stride, w_stride), stride=(t_stride,h_stride,w_stride), bias=False),
                #nn.LayerNorm(out_ch)
            )
            nn.init.xavier_uniform_(self.shortcut[0].weight)

    def forward(self, x):
        def apply_norm(norm, x):
            B, C, T, H, W = x.shape
            return norm(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        identity = x
        if len(self.shortcut) > 0:
            identity = self.shortcut[0](identity)
            #if len(self.shortcut) > 1:
            #    identity = apply_norm(self.shortcut[1], identity)

        # 主路径
        x = self.conv1(x)
        x = apply_norm(self.norm1, x)
        x = F.gelu(x)
        
        x = self.conv2(x)
        x = apply_norm(self.norm2, x)
        #print(x.shape,identity.shape)
        x = x + identity
        return x

class ConvNext3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # 下采样策略 (t_stride, h_stride, w_stride)
            ResBlock(6,    256,  t_stride=4, h_stride=8, w_stride=8),  # 块1：全维度下采样2倍
            #ResBlock(192,  768,  t_stride=2, h_stride=2, w_stride=2),  # 块2：全维度再下采样2倍
            #ResBlock(768,  3072, t_stride=1, h_stride=2, w_stride=2),  # 块3：仅空间下采样
        )
        
    def forward(self, x):
        return self.blocks(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

from copy import deepcopy

def upsample_conv3d_weights(conv_small,size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                      # 输入张量
        size=size,                  # 目标尺寸（时间维度不变）
        mode='trilinear',                # 3D插值
        align_corners=False              # 避免边缘对齐伪影
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    # 如果有偏置项，直接复制（无需修改）
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, ids_restore=None, ids_keep=None, mask_token=None, rand_num_img=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep,rand_num_img),
            k=rope_apply(k, grid_sizes, freqs, b, s, n, d, mask_token, ids_restore, ids_keep, rand_num_img),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

    

import torch.fft
    
class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        #self.mlp_e = DoubleFCWithNorm(96, dim)


        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        rand_num_img=None,
        ids_keep=None,
        ids_restore=None,
        mask_token=None,
        seq_lens1=None,
        cnt_blocks=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        if ids_keep!=None:
            #seq_lens[0] = [x.shape[1]]
            seq_lens[0] = torch.tensor(x.shape[1], dtype=torch.long, device=x.device)

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs,ids_restore=ids_restore,ids_keep=ids_keep,mask_token=mask_token,rand_num_img=rand_num_img)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        B,L,C = x.shape
        
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        model_type = 'i2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
    

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs1 = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)


        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.mask_ratio = 0.3
        self.mask_token=None

        # initialize weights
        self.init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward_side_interpolater(self, x, mask, ids_restore, kwagrs):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_before = x
        x = self.sideblock(x, **kwagrs)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        rand_num_img=None,
        enable_mask=False,
        latent_frame_zero=9,
        cache_sample=False,
        cache=None,
        return_cache=False,
        cache_list=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if rand_num_img!=None and rand_num_img>=0.4:
            x_pack = []
            # According to Eq. 3, compress historical frames at different ratios.
            for u in x:
                u = u.unsqueeze(0)
                f_num = u.shape[2]
                u1 = u[:,:,:-latent_frame_zero]
                u2 = u[:,:,-latent_frame_zero:]
                f_1 = rope_params(1024, self.d - 4 * (self.d // 6)).to(u.device)
                f_2 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                f_3 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
                if f_num - 9 <= 2 + 4:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))

                    if f_zero - 2 <= 0:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,-1].unsqueeze(2),4))
                    else:
                        u_2 = self.patch_embedding_2x(convpadd(u1[:,:,1:-1],4))

                    u_3 = self.patch_embedding(u1[:,:,-1].unsqueeze(2))
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    # Generate the corresponding RoPE encoding based on the compressed historical frames.
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2)],dim=0)
                    f_z = f1+f2+u_3.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - 9 <= 2 + 4 + 16:
                    
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    
                    if f_zero-6<=0:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,-5].unsqueeze(2),8))
                    else:
                        u_2 = self.patch_embedding_4x(convpadd(u1[:,:,1:-5],8))


                    u_3 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_4 = self.patch_embedding(u1[:,:,-3:])
                    
                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3)],dim=0)
                    f_z = f1+f2+f3+u_4.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - 9 <= 2 + 4 + 16 + 64:
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding(u1[:,:,0].unsqueeze(2))
                    

                    if f_zero-22<=0:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,-21].unsqueeze(2),16))
                    else:
                        u_2 = self.patch_embedding_8x(convpadd(u1[:,:,1:-21],16))

                    u_3 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_4 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4)) 
                    u_5 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4)],dim=0)
                    f_z = f1+f2+f3+f4+u_5.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2),u_5.flatten(2).transpose(1, 2)],dim=1)


                elif f_num - 9 <= 2 + 4 + 16 + 64 + 256: 
                    f_zero = u1.shape[2]
                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero-86<=0:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,-85].unsqueeze(2),32))
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(u1[:,:,1:-85],32))


                    u_3 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_4 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_5 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_6 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5)],dim=0)
                    f_z = f1+f2+f3+f4+f5+u_6.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2)],dim=1)

                elif f_num - 9 <= 2 + 4 + 16 + 64 + 256 + 1024:
                    f_zero = u1.shape[2]

                    u_1 = self.patch_embedding_2x(convpadd(u1[:,:,0].unsqueeze(2),4))

                    if f_zero - 342 <= 0:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,-341].unsqueeze(2),4)), 32) )
                    else:
                        u_2 = self.patch_embedding_16x(convpadd(self.patch_embedding_2x_f(convpadd(u1[:,:,1:-341],4)), 32) )


                    u_3 = self.patch_embedding_16x(convpadd(u1[:,:,-341:-85],32) )
                    u_4 = self.patch_embedding_8x(convpadd(u1[:,:,-85:-21],16))
                    u_5 = self.patch_embedding_4x(convpadd(u1[:,:,-21:-5],8))
                    u_6 = self.patch_embedding_2x(convpadd(u1[:,:,-5:-3],4))
                    u_7 = self.patch_embedding(u1[:,:,-3:])

                    f1 = u_1.shape[2]
                    f2 = u_2.shape[2]
                    f3 = u_3.shape[2]
                    f4 = u_4.shape[2]
                    f5 = u_5.shape[2]
                    f6 = u_6.shape[2]
                    freqs_i = torch.cat([up_fre(f_1,f_2,f_3,u_1,0), up_fre(f_1,f_2,f_3,u_2,f1,True), up_fre(f_1,f_2,f_3,u_3,f1+f2,True), up_fre(f_1,f_2,f_3,u_4,f1+f2+f3,True), \
                                          up_fre(f_1,f_2,f_3,u_5,f1+f2+f3+f4,True),up_fre(f_1,f_2,f_3,u_6,f1+f2+f3+f4+f5,True),\
                                            up_fre(f_1,f_2,f_3,u_7,f1+f2+f3+f4+f5+f6)],dim=0)
                    f_z = f1+f2+f3+f4+f5+f6+u_7.shape[2]
                    u1 = torch.cat([u_1.flatten(2).transpose(1, 2),u_2.flatten(2).transpose(1, 2),u_3.flatten(2).transpose(1, 2),u_4.flatten(2).transpose(1, 2), \
                                    u_5.flatten(2).transpose(1, 2),u_6.flatten(2).transpose(1, 2),u_7.flatten(2).transpose(1, 2)],dim=1)

                u2 = self.patch_embedding(u2)
                freqs_i = torch.cat([freqs_i, up_fre(f_1,f_2,f_3,u2,f_z)],dim=0)
                seq_lens1 = u1.shape[1]
                grid_sizes = torch.stack([torch.tensor(u2.shape[2:], dtype=torch.long)])
                u2 = u2.flatten(2).transpose(1, 2)
                seq_lens = torch.tensor([u1.shape[1]+u2.shape[1]], dtype=torch.long)
                u = torch.cat([u1,u2],dim=1)
                x_pack.append(u)

            x = x_pack
            x = torch.cat(x)
        else:
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                        dim=1) for u in x
            ])
            seq_lens1=None

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)


        if rand_num_img!=None and rand_num_img>=0.4:
            self.freqs = freqs_i
        else:  
            self.freqs = self.freqs1.to(x.device)

        # We referenced the implementation at https://github.com/sail-sg/MDT
        ids_keep = None
        ids_restore = None
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
            rand_num_img=rand_num_img,
            seq_lens1=seq_lens1,
            mask_token=self.mask_token, )

        len_blocks = (len(self.blocks)+1)//2
        cnt_blocks = 0

        if cache_sample and cache==None:
            cache = []

        len_blocks = (len(self.blocks)+1)//2
        cnt_blocks = 0
        for block in self.blocks:
            cnt_blocks += 1
            if cnt_blocks==len_blocks and enable_mask:
                kwargs["ids_keep"]=None
                kwargs["ids_restore"]=None
                kwargs["mask_token"]=None
                x = self.forward_side_interpolater(x, mask, ids_restore, kwargs)
                x = block(x, **kwargs, cnt_blocks=cnt_blocks)
            else:
                if cache_sample and not return_cache and (cnt_blocks-1) in cache_list:
                    index = cache_list.index(cnt_blocks - 1)
                    x = x + cache[index]
                else:
                    x_in = x
                    x = block(x, **kwargs, cnt_blocks=cnt_blocks)
                    if cache_sample and return_cache and (cnt_blocks-1) in cache_list:
                        cache.append((x-x_in).to(torch.bfloat16).detach())

        # head
        x = self.head(x, e)

        if rand_num_img!=None and rand_num_img>=0.4:
            # unpatchify
            x = self.unpatchify(x[:,seq_lens1:,:], grid_sizes)
        else:
            # unpatchify
            x = self.unpatchify(x, grid_sizes)
            
        if cache_sample and return_cache:
            return [u.float() for u in x][0], cache
        else:
            return [u.float() for u in x][0], None

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


def convpadd(tensor,pad_num):
    dim = tensor.dim()
    if dim==4:
        tensor = tensor.unsqueeze(2)
    if dim==6:
        tensor = tensor.squeeze(2)
        
    b,c,f,h,w = tensor.shape

    pad_h = (pad_num - h % pad_num) % pad_num  
    pad_w = (pad_num - w % pad_num) % pad_num  
    tensor = torch.cat([tensor,torch.zeros(b,c,f,pad_h,w).to(tensor.device)],dim=3)
    tensor = torch.cat([tensor,torch.zeros(b,c,f,h+pad_h,pad_w).to(tensor.device)],dim=4)
    return tensor

def up_fre(f_1,f_2,f_3,u,f_z,scale=False):
    b1, c1, f1, h1, w1 = u.shape
    freqs_i = torch.cat([
        f_1[f_z:f_z+f1].view(f1, 1, 1, -1).expand(f1, h1, w1, -1),
        f_2[:h1].view(1, h1, 1, -1).expand(f1, h1, w1, -1),
        f_3[:w1].view(1, 1, w1, -1).expand(f1, h1, w1, -1)],
    dim=-1).reshape(f1*h1*w1, 1, -1)
    return freqs_i
