import torch
import torch.nn as nn
from collections import OrderedDict


class Mapnet(nn.Module):
    def __init__(self, word_embeddings, hidden_size):
        super().__init__()
        self.word_embeddings = word_embeddings
        map_hidden_size = max(hidden_size // 16, 1)
        self.subgraph_map_module = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_size, map_hidden_size)),
            ("gelu", nn.GELU()),
            ("linear2", nn.Linear(map_hidden_size, hidden_size))
        ]))

    def forward(self, subgraphs, token_mask=None):
        subgraph_embs = self.word_embeddings(subgraphs)
        mean_subgraph_embs = subgraph_embs.mean(dim=2)
        # mapped = self.subgraph_map_module(mean_subgraph_embs)
        mapped = self.subgraph_map_module(mean_subgraph_embs)
        return mapped
