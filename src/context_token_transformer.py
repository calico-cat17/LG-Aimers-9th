"""Small reliability-aware Transformer for bounded residual correction."""
from __future__ import annotations
import torch
from torch import nn

class ContextTokenTransformer(nn.Module):
    def __init__(self, token_dims, width=32, heads=4, layers=2, dropout=.12,
                 direct_probability=False):
        super().__init__();self.projectors=nn.ModuleList([nn.Sequential(nn.Linear(d,width),nn.LayerNorm(width),nn.GELU()) for d in token_dims]);self.cls=nn.Parameter(torch.zeros(1,1,width));self.type_emb=nn.Parameter(torch.randn(1,len(token_dims)+1,width)*.02)
        self.direct_probability=direct_probability
        layer=nn.TransformerEncoderLayer(width,heads,width*3,dropout,batch_first=True,norm_first=True,activation='gelu');self.encoder=nn.TransformerEncoder(layer,layers);self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,1))
    def forward(self,tokens,reliability):
        z=torch.stack([p(t) for p,t in zip(self.projectors,tokens)],1);z=z*(.20+.80*reliability.unsqueeze(-1));cls=self.cls.expand(z.size(0),-1,-1);h=torch.cat([cls,z],1)+self.type_emb;mask=torch.cat([torch.ones((z.size(0),1),device=z.device),reliability.clamp_min(.05)],1);h=h+mask.log().unsqueeze(-1);h=self.encoder(h);raw=self.head(h[:,0]).squeeze(1)
        return torch.sigmoid(raw) if self.direct_probability else .06*torch.tanh(raw)
