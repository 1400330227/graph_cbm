import torch
from torch import nn

from graph_cbm.modeling.relation.transformer import PositionEmbeddingRandom, Block


class VisionTransformerExtractor(nn.Module):
    """
          一个基于 Transformer 的特征提取器，用于替换 TwoMLPHead。

          Args:
              in_channels (int): 输入特征图的通道数 (e.g., 256 for ResNet-50 FPN)。
              representation_dim (int): 输出特征向量的维度 (e.g., 1024)。
              num_heads (int): Transformer 中多头注意力的头数。
              num_layers (int): Transformer Encoder Block 的层数。
              dropout (float): Dropout 的概率。
          """
    def __init__(self, in_channels, representation_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.representation_dim = representation_dim

        # 1. 位置编码：为特征图的每个像素点提供空间位置信息
        #    输入特征图大小通常是 7x7，所以位置编码维度是 hidden_dim
        #    我们将输入通道投影到 hidden_dim
        hidden_dim = representation_dim
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.pos_encoder = PositionEmbeddingRandom(num_pos_feats=hidden_dim // 2)

        # 2. [CLS] Token：类似于 ViT，用于聚合整个特征图的信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 3. Transformer Encoder 模块
        self.blocks = nn.ModuleList([
            Block(embedding_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4. LayerNorm 和最终的线性层
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, representation_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入的特征图，形状为 (N, C, H, W)。

        Returns:
            Tensor: 输出的特征向量，形状为 (N, representation_dim)。
        """
        # (N, C, H, W) -> (N, D, H, W)
        x = self.input_proj(x)

        # 计算并添加位置编码 (N, D, H, W)
        pos_embedding = self.pos_encoder(x.shape[2:])
        x = x + pos_embedding.unsqueeze(0)

        # (N, D, H, W) -> (N, H*W, D)
        bs, _, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # 准备并拼接 [CLS] token
        # (N, 1, D)
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        # (N, 1+H*W, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # 通过 Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 只取出 [CLS] token 对应的输出，并通过 LayerNorm
        # (N, D)
        x = self.norm(x[:, 0])

        # 通过最后的线性头
        x = self.output_head(x)

        return x