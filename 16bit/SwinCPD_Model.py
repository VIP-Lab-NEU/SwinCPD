import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ===================================================================================
# PART 1: BUILDING BLOCKS
# ===================================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA scales by 1/sqrt(head_dim) internally.
        # If self.scale is different from default, we pre-scale Q.
        # factor = self.scale / (1 / sqrt(head_dim)) = self.scale * sqrt(head_dim)
        factor = self.scale * math.sqrt(C // self.num_heads)
        if factor != 1.0:
            q = q * factor

        # Prepare relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # Base attention mask (bias)
        attn_bias = relative_position_bias.unsqueeze(0)  # 1, nH, N, N

        if mask is not None:
            # Shifted Window Mask Handling
            nW = mask.shape[0]

            # Reshape q, k, v to decouple batch and window dimensions for broadcasting
            # (B_ // nW, nW, nH, N, dim)
            q = q.view(B_ // nW, nW, self.num_heads, N, -1)
            k = k.view(B_ // nW, nW, self.num_heads, N, -1)
            v = v.view(B_ // nW, nW, self.num_heads, N, -1)

            # Combine relative position bias with shifted window mask
            # mask: (nW, N, N) -> (1, nW, 1, N, N)
            # attn_bias: (1, nH, N, N) -> (1, 1, nH, N, N)
            combined_mask = attn_bias.unsqueeze(1) + mask.unsqueeze(1).unsqueeze(0)

            # Run SDPA
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=combined_mask,
                dropout_p=self.attn_drop.p if self.training else 0.
            )

            # Reshape back to flattened batch
            x = x.view(B_, self.num_heads, N, -1)

        else:
            # Standard W-MSA
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.
            )

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PolarizationFusionModulator(nn.Module):
    """
    Generates spatially-aware FiLM parameters (gamma, beta) for each polarization stream.
    """

    def __init__(self, dim, mlp_ratio=2.):
        super().__init__()
        input_dim = dim * 4
        hidden_dim = int(dim * mlp_ratio)
        output_dim = dim * 8  # 4 gammas, 4 betas

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.param_head = nn.Conv2d(hidden_dim, output_dim, 1, 1, 0)
        self.param_head.weight.data.zero_()
        self.param_head.bias.data.zero_()

    def forward(self, img0, img45, img90, img135):
        x = torch.cat([img0, img45, img90, img135], dim=1)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        params_logits = self.param_head(x)
        g0, g45, g90, g135, b0, b45, b90, b135 = torch.chunk(params_logits, 8, dim=1)

        # Apply stabilizing activation functions (optional)
        g0 = 2.0 * torch.sigmoid(g0)
        g45 = 2.0 * torch.sigmoid(g45)
        g90 = 2.0 * torch.sigmoid(g90)
        g135 = 2.0 * torch.sigmoid(g135)

        b0 = torch.tanh(b0)
        b45 = torch.tanh(b45)
        b90 = torch.tanh(b90)
        b135 = torch.tanh(b135)

        return g0, g45, g90, g135, b0, b45, b90, b135


# ===================================================================================
# PART 2: POLARIZATION-AWARE TRANSFORMER BLOCK (PATB)
# ===================================================================================

class PATB(nn.Module):
    """
    PATB using a dynamic, spatially-aware FiLM mechanism for inter-polarization fusion.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size,
                 depth, mlp_ratio=2., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.input_resolution = input_resolution

        self.spatial_transformer_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        self.polar_modulator = PolarizationFusionModulator(dim, mlp_ratio)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(img_size=input_resolution, patch_size=1, embed_dim=dim)

    def _apply_conv_and_residual(self, modulated_img, res_token, x_size):
        """Helper to apply the shared conv and add the initial residual."""
        conv_out = self.conv(modulated_img)
        token_out = self.patch_embed(conv_out)
        return res_token + token_out

    def forward(self, x0, x45, x90, x135, x_size):

        res_x0, res_x45, res_x90, res_x135 = x0, x45, x90, x135

        x_stacked = torch.cat([x0, x45, x90, x135], dim=0)
        y_stacked = self.spatial_transformer_group(x_stacked, x_size)
        y0, y45, y90, y135 = torch.chunk(y_stacked, 4, dim=0)

        img_y0 = self.patch_unembed(y0, x_size)
        img_y45 = self.patch_unembed(y45, x_size)
        img_y90 = self.patch_unembed(y90, x_size)
        img_y135 = self.patch_unembed(y135, x_size)

        if self.use_checkpoint:
            gammas_betas = checkpoint.checkpoint(self.polar_modulator, img_y0, img_y45, img_y90, img_y135)
        else:
            gammas_betas = self.polar_modulator(img_y0, img_y45, img_y90, img_y135)

        g0, g45, g90, g135, b0, b45, b90, b135 = gammas_betas

        modulated_img0 = img_y0 * g0 + b0
        modulated_img45 = img_y45 * g45 + b45
        modulated_img90 = img_y90 * g90 + b90
        modulated_img135 = img_y135 * g135 + b135

        out0 = self._apply_conv_and_residual(modulated_img0, res_x0, x_size)
        out45 = self._apply_conv_and_residual(modulated_img45, res_x45, x_size)
        out90 = self._apply_conv_and_residual(modulated_img90, res_x90, x_size)
        out135 = self._apply_conv_and_residual(modulated_img135, res_x135, x_size)

        return out0, out45, out90, out135


# ===================================================================================
# PART 3: THE FINAL SWINCPD MODEL
# ===================================================================================

class SwinCPD(nn.Module):
    def __init__(self, img_size=64, in_chans=12, out_chans=12,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., img_range=1., use_checkpoint=False, **kwargs):
        super().__init__()

        self.img_range = img_range
        self.upscale = 1
        self.window_size = window_size
        self.mean = torch.zeros(1, in_chans, 1, 1)
        img_size_tuple = to_2tuple(img_size)

        stream_embed_dim = embed_dim

        deep_feature_dim = stream_embed_dim * 4

        self.conv_first_shared = nn.Conv2d(3, stream_embed_dim, 3, 1, 1)

        self.body = nn.ModuleList()
        self.img_to_token = PatchEmbed(img_size=img_size_tuple, patch_size=1, in_chans=stream_embed_dim,
                                       embed_dim=stream_embed_dim)
        self.token_to_img = PatchUnEmbed(img_size=img_size_tuple, patch_size=1, embed_dim=stream_embed_dim)
        self.norm = nn.LayerNorm(stream_embed_dim)

        for i in range(len(depths)):
            self.body.append(PATB(
                dim=stream_embed_dim, input_resolution=img_size_tuple,
                depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, use_checkpoint=use_checkpoint
            ))

        self.conv_after_body = nn.Conv2d(deep_feature_dim, deep_feature_dim, 3, 1, 1)

        self.conv_last = nn.Conv2d(deep_feature_dim, out_chans, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if mod_pad_h == 0 and mod_pad_w == 0:
            return x
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x):
        H_in, W_in = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x_minus_mean = (x - self.mean) * self.img_range

        x_stream_0, x_stream_45, x_stream_90, x_stream_135 = torch.chunk(x_minus_mean, 4, dim=1)

        f_s0 = self.conv_first_shared(x_stream_0)
        f_s45 = self.conv_first_shared(x_stream_45)
        f_s90 = self.conv_first_shared(x_stream_90)
        f_s135 = self.conv_first_shared(x_stream_135)

        x_first = torch.cat([f_s0, f_s45, f_s90, f_s135], dim=1)

        f0 = self.img_to_token(f_s0);
        f45 = self.img_to_token(f_s45);
        f90 = self.img_to_token(f_s90);
        f135 = self.img_to_token(f_s135);

        x_size = (x.shape[2], x.shape[3])
        for block in self.body:
            f0, f45, f90, f135 = block(f0, f45, f90, f135, x_size)

        img0 = self.token_to_img(self.norm(f0), x_size);
        img45 = self.token_to_img(self.norm(f45), x_size);
        img90 = self.token_to_img(self.norm(f90), x_size);
        img135 = self.token_to_img(self.norm(f135), x_size);
        f_body_out = torch.cat([img0, img45, img90, img135], dim=1)

        res = self.conv_after_body(f_body_out) + x_first

        out = x_minus_mean + self.conv_last(res)

        out = out / self.img_range + self.mean
        return out[:, :, :H_in, :W_in]


if __name__ == '__main__':
    B, H, W = 2, 96, 96
    in_channels = 12
    out_channels = 12

    model = SwinCPD(
        img_size=H,
        in_chans=in_channels,
        out_chans=out_channels,
        embed_dim=48,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        use_checkpoint=False
    )

    print("--- Testing SwinCPD ---")
    input_tensor = torch.randn(B, in_channels, H, W)
    output_tensor = model(input_tensor)

    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    assert output_tensor.shape == (B, out_channels, H, W)
    print("\nShape verification successful!")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")
#    print(model)