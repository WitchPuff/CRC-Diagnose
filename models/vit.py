import torch
import torch.nn as nn


    


class PatchEmbedded(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 input_size=224, 
                 patch_size=16, 
                 drop_rate=0.):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2 # 196
        self.embed_dim = 3 * (patch_size ** 2) # 768
        # 用卷积运算划分patches
        self.proj = nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        # 追加class token，并使用该向量进行分类预测
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        x = nn.LayerNorm([C, H, W]).to(x.device)(x)
        x = self.proj(x).transpose(1,3)
        x = nn.Flatten(1,2)(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = torch.cat((cls_tokens, x), dim=1)
        # 将编码向量中加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x
        
        
class Attention(nn.Module):
    def __init__(self,
                scale,
                attn_drop=0.):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        self.attn_drop = nn.Dropout(p=attn_drop)
        
    def forward(self, q, k, v, mask=None):
        # assert q.shape[2] == k.shape[1]
        output = torch.matmul(q, k.transpose(1,2)) * self.scale
        if mask is not None:
            pass
        attn = self.softmax(output)
        output = torch.matmul(attn, v)
        output = self.attn_drop(output)
        return attn, output
        
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d):
        super().__init__()

        self.n_head = n_head
        self.d = d
        self.d_per_head = d // n_head

        self.fc_q = nn.Linear(d, n_head * self.d_per_head)
        self.fc_k = nn.Linear(d, n_head * self.d_per_head)
        self.fc_v = nn.Linear(d, n_head * self.d_per_head)

        self.attention = Attention(scale=self.d_per_head**-0.5)

        self.fc_o = nn.Linear(n_head * self.d_per_head, self.d)

    def forward(self, q, k, v, mask=None):
        
        batch, n_q, _ = q.size()
        batch, n_k, _ = k.size()
        batch, n_v, _ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, self.n_head, self.d_per_head).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_per_head)
        k = k.view(batch, n_k, self.n_head, self.d_per_head).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_per_head)
        v = v.view(batch, n_v, self.n_head, self.d_per_head).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_per_head)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(self.n_head, batch, n_q, self.d_per_head).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output
        
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, n_head=12, ffn_radio=4, dropout=0.):
        super().__init__()
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(n_head, embed_dim)
        self.ffn = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim*ffn_radio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(embed_dim*ffn_radio), embed_dim),
                nn.Dropout(dropout)
        )
    def forward(self, x):
        u = self.norm(x)
        q = self.wq(u)
        k = self.wk(u)
        v = self.wv(u)
        attn, u = self.mha(u, q, k, v)
        u += x # residual connection
        v = self.norm(u)
        v = self.ffn(v)
        v += u
        return v



class ViT(nn.Module):
    def __init__(self, embed_dim=768, n_head=12, num_classes=9, depth=6,
                 in_chans=3, input_size=224, patch_size=16, drop_rate=0.2,
                 ffn_radio=4) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(embed_dim=embed_dim, n_head=n_head, ffn_radio=ffn_radio, dropout=drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls = nn.Linear(embed_dim, num_classes)
        self.patch = PatchEmbedded(in_chans, input_size, patch_size, drop_rate)
    
    def forward(self, x):
        x = self.patch(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        x = self.cls(x[:,0])
        return x
