import torch.nn as nn


class ConsistencyModule(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.subgraph_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, subgraph):
        attention_output, _ = self.subgraph_self_attention(
            subgraph,
            subgraph,
            subgraph,
            need_weights=False,
        )
        return attention_output
