"""
    fuse node-level and graph-level patient embedding 
"""
import torch
import torch.nn.functional as F
from torch import nn


class MergePatient(torch.nn.Module):
    def __init__(self, agg_type, **kwargs):
        """
        agg_type
        input_dim
        hidden_dim
        """
        super().__init__()
        self.agg_type = agg_type
        self.input_dim = kwargs["input_dim"]
        self.input_num = kwargs["input_num"]
        if agg_type == "attn":
            self.ws = nn.ModuleList([nn.Linear(self.input_dim, 1) for i in range(self.input_num)])
        elif agg_type == "concat":
            self.w = nn.Linear(self.input_dim * 2, self.input_dim)

    def forward(self, *outs):
        if self.agg_type == "attn":
            a = torch.stack([self.ws[i](outs[i]) for i in range(self.input_num)], dim=1).squeeze()
            a = F.softmax(a, dim=1).unsqueeze(1)
            out = a @ torch.stack(outs, dim=1)
            return out.squeeze()
        elif self.agg_type == "concat":
            return self.w(torch.cat(outs, dim=-1))
        elif self.agg_type == "mean":
            return torch.mean(torch.stack(outs), dim=0)


class Patient2Diag(torch.nn.Module):
    def __init__(self, agg_type, **kwargs):
        """
        agg_type
        input_dim
        hidden_dim
        """
        super().__init__()
        self.agg_type = agg_type
        if agg_type == "attn":
            self.input_dim = kwargs["input_dim"]
            self.hidden_dim = kwargs["hidden_dim"]
            self.W2 = nn.Linear(self.hidden_dim, 1)
            self.W1 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, out, mask):
        """
        mask: (batch_size, length), padding mask, length is the maximum length of the current batch
        out: (batch_size, length, dim)
        """
        m = (mask == False).type(torch.int).unsqueeze(-1)
        out = out * m

        if self.agg_type == "mean":
            out = out.sum(axis=1)
            return out / m.sum(axis=1)
        elif self.agg_type == "attn":
            a = self.W2(F.tanh(self.W1(out))).squeeze()
            a = a.masked_fill(mask, -1e9)
            a = F.softmax(a, dim=1)
            out = a.unsqueeze(1) @ out
            return out.squeeze()


class Diag2Patient(torch.nn.Module):
    def __init__(self, agg_type, **kwargs):
        """
        agg_type
        input_dim
        hidden_dim
        """
        super().__init__()
        self.agg_type = agg_type
        if agg_type == "attn":
            self.input_dim = kwargs["input_dim"]
            self.hidden_dim = kwargs["hidden_dim"]
            self.W = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 1)
            )

    def forward(self, out, mask):
        """
        mask: (batch_size, length), padding mask, length is the maximum length of the current batch
        out: (batch_size, length, dim)
        """
        m = (mask == False).type(torch.int).unsqueeze(-1)
        out = out * m

        if self.agg_type == "mean":
            out = out.sum(axis=1)
            return out / m.sum(axis=1)
        elif self.agg_type == "attn":
            a = self.W(out).squeeze()
            a = a.masked_fill(mask, -1e9)
            a = F.softmax(a, dim=1)
            out = a.unsqueeze(1) @ out
            return out.squeeze()
