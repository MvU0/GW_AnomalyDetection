import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math, sys

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)



    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """
        Modernized forward that is robust to whether propagate returns
        [N, H, C] or [N, H*C].  Ensures bias addition works.
        """
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))


        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                    num_nodes=x[1].size(self.node_dim))   # Not really sure what this num_nodes part actually does, seems optional
        
        # propagate -> usually returns [N, H*C] in your current message() impl
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                            return_attention_weights=return_attention_weights)
    
        
        # Case B: propagate returned [N, H*C] (2D) — this is your current situation
        if not self.concat:
            # reshape to [N, H, C] then average across heads to [N, C]
            out = out.view(-1, self.heads, self.out_channels).mean(dim=1)
            # if concat == True, out already is [N, H*C], nothing to do
        
        # Bias add (safe broadcast)
        if self.bias is not None:
            out = out + self.bias  # bias shape should be (H*C,) if concat True else (C,)
        
        
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out
        

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):
        """
        Modernized message:
        - Ensure key_i/key_j exist even if embedding is None.
        - Use modern softmax API.
        - Return messages flattened [E, H*C] so aggregate produces [N, H*C].
        """
        # x_i, x_j arrive as [E, H*C] (because lin produced H*C features)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        # If embedding provided, concatenate; otherwise use x as keys
        if embedding is not None:
            embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)
        else:
            # IMPORTANT: define keys even when no embedding
            key_i, key_j = x_i, x_j
        
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # compute raw attention scores -> shape [E, H]
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = alpha.view(-1, self.heads, 1)     # [E, H, 1]

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, dim=0)   

    
        
        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_j * alpha
        msg = msg.view(msg.size(0), -1)
        return msg




    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
