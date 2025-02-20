import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed

class CrossAttentionAggregation(nn.Module):

    def __init__(self, emb_dim, num_heads):
        super(CrossAttentionAggregation, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.var_query = nn.Parameter(torch.randn(1, 1, emb_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

    def forward(self, x):
        b, n, c = x.shape
        var_query_expanded = self.var_query.expand(
            b, n, -1
        )  # Adjusting query to match the input dimensions
        attn_output, _ = self.var_agg(var_query_expanded, x, x)
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = CrossAttentionAggregation(emb_dim, num_heads)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DataFusion(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        emb_dim,
        num_heads,
        mlp_dim,
        depth,
        decoder_depth,
        in_channels=9,
        default_vars=None,
    ):
        super(DataFusion, self).__init__()
        self.emb_dim = emb_dim
        self.default_vars = default_vars or [f"var_{i}" for i in range(in_channels)]
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, emb_dim) for _ in range(in_channels)]
        )
        self.num_patches = self.token_embeds[0].num_patches
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)

        # Variable embedding to denote which variable each token belongs to
        self.var_embed, self.var_map = self.create_var_embedding(emb_dim)

        # Variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.randn(1, 1, emb_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        # Prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(emb_dim, emb_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(emb_dim, emb_dim))
        self.head = nn.Sequential(*self.head)

        # Separate channel-wise encoder to get embeddings
        self.channel_encoders = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(in_channels)]
        )
         # Downsampling layer to achieve [1, 1, 448, 448]
       # self.downsample = nn.AdaptiveAvgPool2d(img_size)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        var_map = {var: idx for idx, var in enumerate(self.default_vars)}
        return var_embed, var_map

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, v, l, d = x.shape
        x = torch.einsum("bvld->blvd", x)  # Reorder dimensions to (B, L, V, D)
        x = x.flatten(0, 1)  # (BxL, V, D)

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D

        # x = x.view(b, 1, 448, 448)  
        # x = x.view(b, 1, int(l ** 0.5), int(l ** 0.5))  # Reshape based on dimensions

        # Downsample to target size [1, 1, 448, 448]
        #x = self.downsample(x)
        # Aggregate across first dimension to get [1, 448, 448]
        #x = x.mean(dim=0, keepdim=True)  # Taking mean across the first dimension
        #x=x.unsqueeze(0)
        return x

    def forward(self, x):
        # if x.shape[1] > 1:  # If the input has a time dimension
        #     batch_size, channels, height, width = x.shape
        #     x = x.mean(dim=1)  # Average pooling over the time dimension
        # else:
        #     batch_size, channels, hxeight, width = x.shape
        #    # x = x.squeeze(1)
        # shape [B,bands,H,W]
        batch_size, channels, height, width = x.shape
        token_embeddings = [
            self.token_embeds[i](x[:, i : i + 1, :, :]) for i in range(channels)
        ]

        # Reshape and concatenate embeddings
        x = torch.stack(token_embeddings, dim=1)  # B, V, L, D
        x = self.aggregate_variables(x)  # Aggregate variables

        # for blk in self.transformer_blocks:
        #     x = blk(x)
        # x = self.norm(x)
        # x = self.head(x)
        x=x.clone().detach() 

        return x  # Final fused embedding

        # same resolution, same bit (no transformation, min max consistent range), dataloader, object file -> torch tensor
