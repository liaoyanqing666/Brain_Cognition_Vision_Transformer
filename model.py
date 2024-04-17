# vit model
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class MultiHeadAttention(nn.Module):
    """
    Input dim:
        q: [B, len_q, D]
        k: [B, len_kv, D]
        v: [B, len_kv, D]

    Output dim:
        output: [B, len_q, dim_v]
        attn: [B, num_heads, len_q, len_kv]
    """

    def __init__(self, dim: int, dim_qk: int = None, dim_v: int = None, num_heads: int = 1, dropout: float = 0.):
        """
        :param dim: input dimension
        :param dim_qk: query and key dimension, default to dim
        :param dim_v: value dimension, also the output dim of input token, default to dim
        :param num_heads: number of heads
        :param dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        dim_qk = dim if dim_qk is None else dim_qk
        dim_v = dim if dim_v is None else dim_v

        assert dim % num_heads == 0 and dim_v % num_heads == 0 and dim_qk % num_heads == 0, 'dim must be divisible by num_heads'

        self.dim = dim
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(dim, dim_qk)
        self.w_k = nn.Linear(dim, dim_qk)
        self.w_v = nn.Linear(dim, dim_v)

    def forward(self, q, k, v, mask=None):
        # q: [B, len_q, D]
        # k: [B, len_kv, D]
        # v: [B, len_kv, D]
        assert q.ndim == k.ndim == v.ndim == 3, 'input must be 3-dimensional'

        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        assert q.size(-1) == k.size(-1) == v.size(-1) == self.dim, 'dimension mismatch'
        assert len_k == len_v, 'len_k and len_v must be equal'
        len_kv = len_v

        q = self.w_q(q).view(-1, len_q, self.num_heads, self.dim_qk // self.num_heads)
        k = self.w_k(k).view(-1, len_kv, self.num_heads, self.dim_qk // self.num_heads)
        v = self.w_v(v).view(-1, len_kv, self.num_heads, self.dim_v // self.num_heads)
        # q: [B, len_q, num_heads, dim_qk//num_heads]
        # k: [B, len_kv, num_heads, dim_qk//num_heads]
        # v: [B, len_kv, num_heads, dim_v//num_heads]
        # The following 'dim_(qk)//num_heads' is writen as d_(qk)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q: [B, num_heads, len_q, d_qk]
        # k: [B, num_heads, len_kv, d_qk]
        # v: [B, num_heads, len_kv, d_v]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_qk ** 0.5)
        # attn: [B, num_heads, len_q, len_kv]

        if mask is not None:
            attn = attn.transpose(0, 1).masked_fill(mask, float('-1e20')).transpose(0, 1)
        attn = torch.softmax(attn, dim=-1)
        attn_with_drop = self.dropout(attn)

        output = torch.matmul(attn_with_drop, v)
        # output: [B, num_heads, len_q, d_v]
        output = output.transpose(1, 2)
        # output: [B, len_q, num_heads, d_v]
        output = output.contiguous().view(-1, len_q, self.dim_v)
        # output: [B, len_q, num_heads * d_v] = [B, len_q, dim_v]
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0., activation=nn.GELU()):
        super(FeedForward, self).__init__()
        hidden_dim = mlp_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_qk, dim_v, mlp_dim, dropout = 0.):

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim, num_heads = heads, dim_qk = dim_qk, dim_v = dim_v, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x, _ = attn(x, x, x)
            x = x + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim_head , mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()


if __name__ == '__main__':
    model = MultiHeadAttention(512, num_heads=8)

    # _ = FeedForward(512, 1024, 0.2) # for test
    # _ = Transformer(512, 6, 8, 64, 64, 1024, 0.2) # for test
    
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        dim_head = 1024,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)