import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, heads_num, context_length, dropout, qkv_bias=False):
        super().__init__()

        assert(dim_out % heads_num == 0)
        self.dim_out = dim_out
        self.head_dim = dim_out // heads_num
        self.heads_num = heads_num
        self.query_w = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.key_w = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.value_w = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.proj_out = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        b, tokens_num, dim = x.shape
        # print(f"input shape: {x.shape}")
        # 把最后dim_out维度打散成head_num x head_dim，然后交换tokens_num和head_num位置便于后面的attention计算
        query = self.query_w(x).view(b, tokens_num, self.heads_num, self.head_dim).transpose(1, 2)
        key = self.key_w(x)
        # print(f"key shape: {key.shape}")
        key = key.view(b, tokens_num, self.heads_num, self.head_dim)
        # print(f"key shape after unsqueeze: {key.shape}")
        key = key.transpose(1, 2)
        # print(f"key shape after transpose: {key.shape}")
        value = self.value_w(x).view(b, tokens_num, self.heads_num, self.head_dim).transpose(1, 2)

        attention_scores = query @ key.transpose(2, 3)
        # print(f"attention_scores shape: {attention_scores.shape}")
        # 后面带下划线_的method会改变本身的值
        attention_scores.masked_fill_(self.mask.bool()[:tokens_num, :tokens_num], -torch.inf)

        attention_weight = torch.softmax(attention_scores / key.shape[-1] ** 0.5, dim=-1)

        attention_weight = self.dropout(attention_weight)

        context_vecs = attention_weight @ value
        # print(f"context_vecs shape: {context_vecs.shape}")
        context_vecs = context_vecs.transpose(1, 2)
        # print(f"context_vecs shape after transpose: {context_vecs.shape}")
        context_vecs = context_vecs.contiguous().view(b, tokens_num, self.dim_out)
        # print(f"context_vecs shape after squeeze: {context_vecs.shape}")
        context_vecs = self.proj_out(context_vecs)
        return context_vecs