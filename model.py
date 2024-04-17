# vit model
import torch
from torch import nn
from einops import rearrange

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
    def __init__(self, dim, hidden_dim, dropout=0., activation=nn.GELU()):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Multi-head self-attention layer.

    Implements self-attention with multiple attention heads.

    Parameters
    ----------
    dim : int
        Input dimension.
    heads : int, optional
        Number of attention heads, by default 8
    dim_head : int, optional
        Dimension of each attention head, by default 64
    dropout : float, optional
        Dropout rate, by default 0.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Layer normalization for the input
        self.norm = nn.LayerNorm(dim)

        # Softmax activation for attention weights
        self.attend = nn.Softmax(dim=-1)

        # Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)

        # Linear layer to transform input to (heads * dim_head) features
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Projection layer if number of heads is greater than 1 or dim_head is
        # different from dim.
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, dim)
        """

        x = self.norm(x)

        # Split the last dimension of `x` into 3 separate tensors of shape
        # (batch_size, sequence_length, heads * dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Rearrange the dimensions of the query, key, and value tensors
        # so that the head dimension is moved to the front.
        # The resulting tensors have shape
        # (batch_size, heads, sequence_length, dim_head)
        q, k, v = map(
            lambda t: t.rearrange('b n (h d) -> b h n d', h=self.heads), qkv
        )

        # Compute dot product attention
        # The resulting tensor has shape (batch_size, heads, sequence_length, sequence_length)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply softmax to the attention weights
        attn = self.attend(dots)

        # Apply dropout to the attention weights
        attn = self.dropout(attn)

        # Compute the final output of the layer by multiplying the attention
        # weights with the value tensor
        out = torch.matmul(attn, v)

        # Rearrange the dimensions of the output tensor so that the head
        # dimension is moved to the end
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply the final projection layer to the output
        return self.to_out(out)


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
    _ = Attention(512)
    
    _ = FeedForward(512, 1024, 0.2) # for test
