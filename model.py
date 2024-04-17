# vit model
import torch
from torch import nn


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
    def __init__(self):
        super(FeedForward, self).__init__()

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
