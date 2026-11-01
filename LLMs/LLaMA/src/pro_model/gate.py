import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(input_dim // 2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = MLP(embedding_dim, embedding_dim)

    def forward(self, question, retrieval):
        question = self.mlp(question)
        retrieval = self.mlp(retrieval)
        simi = torch.bmm(retrieval, question.permute(0, 2, 1))
        simi = torch.mean(simi, dim=-1, keepdim=True)
        gate = torch.sigmoid(simi)
        return gate


class GatingModule(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.dense_question_mlp = MLP(token_dim, token_dim)
        self.siamese_subgraph = SiameseNetwork(token_dim)

    def forward(self, question, subgraph_token):
        dense_question = self.dense_question_mlp(question)
        return self.siamese_subgraph(dense_question, subgraph_token) * subgraph_token

