import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from mmcv.ops import MSDeformAttn
import math

class LSTA(nn.Module):
    def __init__(
        self,
        num_classes=7,      # 分类任务的类别数
        num_heads=8,        # 多头注意力的头数，通常选择2的幂次，需要能整除embed_dim
        embed_dim=768       # 特征维度，需要能被num_heads整除
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.num_classes = num_classes
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        
        # EDL head for uncertainty estimation
        self.edl_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, num_classes, kernel_size=1)
        )
        
        # 特征维度
        self.appearance_dim = embed_dim  # appearance features: (B, embed_dim, H, W)
        self.temporal_dim = embed_dim    # temporal features: (B, KT+1, embed_dim)
        self.fusion_dim = embed_dim
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.fusion_dim // 2, num_classes)
        )

        # region token处理层
        self.region_proj = nn.Linear(embed_dim, embed_dim)
        self.region_norm = nn.LayerNorm(embed_dim)
        
        # self-attention模块
        self.num_heads = num_heads
        self.region_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 使用MSDeformAttn替换原来的cross-attention
        self.deform_attn = MSDeformAttn(
            d_model=embed_dim,
            n_levels=1,  # 只有一个特征层级
            n_heads=8,
            n_points=4   # 每个注意力头采样4个点
        )
        
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.deform_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(3072, embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # 添加temporal cross-attention模块
        self.temporal_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_cross_norm = nn.LayerNorm(embed_dim)

        # self attention for region tokens
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # deformable cross attention
        self.cross_attn = MSDeformAttn(d_model=embed_dim, n_levels=1, n_heads=num_heads, n_points=4)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # position embeddings
        self.pos_embed = PositionEmbeddingLearned(2, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def compute_uncertainty(self, evidence):
        """计算uncertainty map"""
        alpha = evidence + 1
        uncertainty = self.num_classes / torch.sum(alpha, dim=1)  # (B, H, W)
        return uncertainty

    def get_top_regions(self, uncertainty_map, num_regions=5):
        """获取uncertainty最高的前N个区域的坐标
        Args:
            uncertainty_map: (B, H, W)
            num_regions: 需要返回的区域数量
        Returns:
            coords: (B, num_regions, 4) 每个区域的坐标 [x1, y1, x2, y2]
        """
        B, H, W = uncertainty_map.shape
        window_h, window_w = H // 6, W // 6  # 区域大小为原图的1/6
        
        coords_list = []
        for b in range(B):
            # 对每个batch的uncertainty map进行处理
            curr_map = uncertainty_map[b]  # (H, W)
            regions = []
            
            # 滑动窗口计算每个区域的平均uncertainty
            for i in range(0, H - window_h + 1, window_h // 2):  # 步长为window_size的一半，允许重叠
                for j in range(0, W - window_w + 1, window_w // 2):
                    region_uncertainty = curr_map[i:i+window_h, j:j+window_w].mean()
                    regions.append(([j, i, j+window_w, i+window_h], region_uncertainty))
            
            # 按uncertainty排序并获取前num_regions个区域
            regions.sort(key=lambda x: x[1], reverse=True)  # uncertainty越大越靠前
            top_coords = torch.tensor([region[0] for region in regions[:num_regions]], 
                                    device=uncertainty_map.device)
            
            # 如果区域数量不足，用最后一个区域填充
            if len(regions) < num_regions:
                last_coord = top_coords[-1]
                top_coords = torch.cat([top_coords] + [last_coord.unsqueeze(0)] * (num_regions - len(regions)))
            
            coords_list.append(top_coords)
        
        return torch.stack(coords_list)  # (B, num_regions, 4)

    def extract_region_tokens(self, features, coords):
        """将不确定性区域及其上下文分别转换为tokens
        Args:
            features: (B, C, H, W) 原始特征图
            coords: (B, num_regions, 4) 区域坐标 [x1, y1, x2, y2]
        Returns:
            region_tokens: (B, num_regions, C) 区域tokens
            context_tokens: (B, num_regions, C) 上下文tokens
        """
        B, C, H, W = features.shape
        num_regions = coords.size(1)
        region_tokens = []
        context_tokens = []

        for b in range(B):
            batch_region_tokens = []
            batch_context_tokens = []
            for i in range(num_regions):
                x1, y1, x2, y2 = coords[b, i]
                
                # 扩大context范围（在原区域基础上向外扩展1/3的区域大小）
                ctx_size = ((y2 - y1) // 3, (x2 - x1) // 3)
                ctx_x1 = max(0, x1 - ctx_size[1])
                ctx_y1 = max(0, y1 - ctx_size[0])
                ctx_x2 = min(W, x2 + ctx_size[1])
                ctx_y2 = min(H, y2 + ctx_size[0])
                
                # 提取区域特征和context特征
                region_feat = features[b, :, y1:y2, x1:x2]  # 中心区域
                context_feat = features[b, :, ctx_y1:ctx_y2, ctx_x1:ctx_x2]  # 包含context的大区域
                
                # 分别对region和context进行池化
                region_feat = F.adaptive_avg_pool2d(region_feat.unsqueeze(0), (1, 1))  # (1, C, 1, 1)
                context_feat = F.adaptive_avg_pool2d(context_feat.unsqueeze(0), (1, 1))  # (1, C, 1, 1)
                
                # 提取特征
                region_feat = region_feat.squeeze(-1).squeeze(-1)  # (1, C)
                context_feat = context_feat.squeeze(-1).squeeze(-1)  # (1, C)
                
                batch_region_tokens.append(region_feat)
                batch_context_tokens.append(context_feat)
            
            batch_region_tokens = torch.cat(batch_region_tokens, dim=0)  # (num_regions, C)
            batch_context_tokens = torch.cat(batch_context_tokens, dim=0)  # (num_regions, C)
            region_tokens.append(batch_region_tokens)
            context_tokens.append(batch_context_tokens)

        region_tokens = torch.stack(region_tokens)  # (B, num_regions, C)
        context_tokens = torch.stack(context_tokens)  # (B, num_regions, C)
        
        # 通过投影层和归一化层
        region_tokens = self.region_proj(region_tokens)
        region_tokens = self.region_norm(region_tokens)
        context_tokens = self.region_proj(context_tokens)  # 使用相同的投影层
        context_tokens = self.region_norm(context_tokens)  # 使用相同的归一化层
        
        return region_tokens, context_tokens

    def process_region_tokens(self, region_tokens, region_coords, context_tokens):
        """
        使用可变形注意力处理region tokens
        Args:
            region_tokens: shape (B, num_regions, C)
            region_coords: shape (B, num_regions, 4) - [x1,y1,x2,y2]
            context_tokens: shape (B, H*W, C)
        Returns:
            processed_tokens: shape (B, num_regions, C)
        """
        B = region_tokens.shape[0]
        
        # 1. Self attention on region tokens
        residual = region_tokens
        region_tokens = self.norm1(region_tokens)
        region_tokens = self.self_attn(region_tokens, region_tokens, region_tokens)[0]
        region_tokens = residual + region_tokens
        
        # 2. Convert box coordinates to reference points
        reference_points = torch.zeros_like(region_coords[...,:2])
        reference_points[...,0] = (region_coords[...,0] + region_coords[...,2])/2  # center x
        reference_points[...,1] = (region_coords[...,1] + region_coords[...,3])/2  # center y
        reference_points = reference_points.unsqueeze(2)  # (B, num_regions, 1, 2)
        
        # 3. Prepare spatial shapes for deformable attention
        H = W = int(math.sqrt(context_tokens.shape[1]))
        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=region_tokens.device)
        level_start_index = torch.as_tensor([0], device=region_tokens.device)
        
        # 4. Deformable cross attention
        residual = region_tokens
        region_tokens = self.norm2(region_tokens)
        region_tokens = self.cross_attn(
            query=region_tokens,
            reference_points=reference_points,
            input_flatten=context_tokens,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index
        )[0]
        region_tokens = residual + region_tokens
        
        # 5. FFN
        residual = region_tokens
        region_tokens = self.norm3(region_tokens)
        region_tokens = self.linear2(F.relu(self.linear1(region_tokens)))
        region_tokens = residual + region_tokens
        
        return region_tokens

    def process_temporal_tokens(self, region_tokens, temp_features):
        """使用region tokens作为query，temporal features作为key和value进行cross-attention
        Args:
            region_tokens: (B, num_regions, C) region tokens作为query
            temp_features: (B, KT, C) temporal features作为key和value
        Returns:
            temporal_tokens: (B, num_regions, C) 处理后的temporal tokens
        """
        B, num_regions, C = region_tokens.shape
        KT = temp_features.shape[1]
        
        # 1. 准备reference points (使用固定的参考点,因为temporal features是1D序列)
        reference_points = torch.linspace(0, 1, KT, device=region_tokens.device)
        reference_points = reference_points.view(1, KT, 1, 1).repeat(B, 1, 1, 1)  # (B, KT, 1, 1)
        
        # 2. 准备spatial shapes (对于temporal sequence,我们将其视为1D序列)
        spatial_shapes = torch.as_tensor([(KT, 1)], dtype=torch.long, device=region_tokens.device)
        level_start_index = torch.as_tensor([0], device=region_tokens.device)
        
        # 3. 使用可变形注意力
        temporal_tokens = self.cross_attn(
            query=region_tokens,  # (B, num_regions, C)
            reference_points=reference_points,  # (B, KT, 1, 1)
            input_flatten=temp_features,  # (B, KT, C)
            input_spatial_shapes=spatial_shapes,  # (n_levels, 2)
            input_level_start_index=level_start_index  # (n_levels, )
        )[0]
        
        # 4. 残差连接和归一化
        temporal_tokens = region_tokens + temporal_tokens
        temporal_tokens = self.norm2(temporal_tokens)
        
        # 5. FFN
        residual = temporal_tokens
        temporal_tokens = self.norm3(temporal_tokens)
        temporal_tokens = self.linear2(F.relu(self.linear1(temporal_tokens)))
        temporal_tokens = residual + temporal_tokens
        
        return temporal_tokens

    def forward(self, app_feats, temp_feats, region_coords=None):
        """
        前向传播函数
        Args:
            app_feats: appearance特征 (B, T, C, H, W)
            temp_feats: temporal特征 (B, KT, C)
            region_coords: 区域坐标 (B, num_regions, 4) [x1,y1,x2,y2]
        Returns:
            output: (B, num_classes) 分类输出
            evidence: EDL head的输出
            uncertainty: 不确定性图
            region_coords: 区域坐标
        """
        B, T, C, H, W = app_feats.shape
        
        # 1. 计算uncertainty map
        evidence = self.compute_uncertainty(temp_feats)  # (B, T, H, W)
        uncertainty = self.compute_uncertainty(evidence)  # (B, T, H, W)
        
        # 2. 获取top uncertain regions
        if region_coords is None:
            region_coords = self.get_top_regions(uncertainty)  # (B, num_regions, 4)
        
        # 3. 提取region tokens和context tokens
        region_tokens, context_tokens = self.extract_region_tokens(app_feats, region_coords)
        
        # 4. 添加位置编码
        region_centers = torch.zeros_like(region_coords[...,:2])
        region_centers[...,0] = (region_coords[...,0] + region_coords[...,2])/2
        region_centers[...,1] = (region_coords[...,1] + region_coords[...,3])/2
        pos_embed = self.pos_embed(region_centers)
        region_tokens = region_tokens + pos_embed
        
        # 5. 处理region tokens和context tokens
        region_tokens = self.process_region_tokens(region_tokens, region_coords, context_tokens)
        
        # 6. 使用region tokens和temporal features进行cross attention
        temporal_tokens = self.process_temporal_tokens(region_tokens, temp_feats)
        
        # 7. 对temporal_tokens进行平均池化得到最终特征
        fusion_feats = temporal_tokens.mean(dim=1)  # (B, C)
        
        # 8. 将output初始化为0向量
        output = torch.zeros(B, self.num_classes, device=fusion_feats.device)
        
        return output

@register_model
def build_lsta(num_classes=7, appearance_model=None, temporal_model=None, **kwargs):
    model = LSTA(
        num_classes=num_classes,
        appearance_model=appearance_model,
        temporal_model=temporal_model,
        **kwargs
    )
    return model
