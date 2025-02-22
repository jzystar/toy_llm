import torch
from torch import nn
from mha import MultiHeadAttention


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in = cfg["emb_dim"],
            dim_out = cfg["emb_dim"],
            heads_num = cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut x -> norm -> mha -> dropout + x -> shortcut x-> norm -> feedforward -> dropout + x
        norm1 = self.norm1(x)
        att = self.att(norm1)
        dropout = self.dropout_shortcut(att)
        x = dropout + x

        norm2 = self.norm2(x)
        ff = self.ff(norm2)
        dropout = self.dropout_shortcut(ff)
        x = dropout + x
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # toekn embedding 是对词库大小嵌入计算
        self.tok_emb = nn.Embedding(cfg["vocab_size"],  cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出是词库大小的维度，因为是每个词的概率
        self.out_head = nn.Linear( cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, token_length = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(token_length, device=x.device))

        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.,]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        # print(f"name = {name}, params = {param}")
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        tokens = idx[:, -context_size:]
        print(f"token: {tokens.shape}")
        print(tokens)

        with torch.no_grad():
            logits = model(tokens)
        print(f"logits: {logits.shape}")

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        print(f"probas: {probas.shape}")

        next_token = torch.argmax(probas, dim=-1, keepdim=True)
        print(f"next_token: {next_token.shape}")
        print(next_token)

        idx = torch.cat((idx, next_token), dim=1)
    
    return idx