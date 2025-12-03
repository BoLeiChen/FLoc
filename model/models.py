import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden=128, num_heads=8, dim_B=142):
        super().__init__()

        
        self.proj_k = nn.Linear(hidden, hidden)
        self.proj_v = nn.Linear(hidden, hidden)
        self.proj_q = nn.Linear(hidden, hidden)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True
        )

        # FeedForward
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, A, B):
        
        # B: (B,81,142)
        
        Q = self.proj_q(A) # [B 40 128]
        K = self.proj_k(B) # [B 81X8 128]
        V = self.proj_v(B)

        # Cross Attention
        attn_out, _ = self.attn(Q, K, V)  # (B,40,128)
        A = self.norm1(A + attn_out)

        # FFN
        ff_out = self.ff(A)
        A = self.norm2(A + ff_out)

        return A


class CrossAttentionStack(nn.Module):
    def __init__(self, num_layers=4, hidden=128, dim_B=128):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden=hidden, dim_B=dim_B)
            for _ in range(num_layers)
        ])

    def forward(self, A, B):
        for layer in self.layers:
            A = layer(A, B)  # (B,40,128)
        return A
    
# class RMDSlidingWindowEncoder(nn.Module):
#     def __init__(self, 
#                  total_rays=140,      # RMD 原始射线数 (360度)
#                  fov_rays=40,         # RGB 对应多少条 RMD 射线 (窗口大小)
#                  num_views=8,         # 把每个点拆成多少个方向
#                  output_dim=128):     # 最终映射到的特征维度 (要和 RGB 维度一致)
#         super().__init__()
        
#         self.total_rays = total_rays
#         self.fov_rays = fov_rays
#         self.num_views = num_views
        
#         # 计算每个视角之间错开多少个索引
#         # 比如 140条线分8个方向，每转一个方向就错开 140/8 ≈ 17 条线
#         self.shift_step = int(total_rays / num_views)
        
#         input_feat_dim = fov_rays + 2 + 2
#         self.local_encoder = nn.Sequential(
#             nn.Linear(input_feat_dim, 256), # +2 是为了放入当前视角的 Sin/Cos 编码
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, rmd_matrix):
#         """
#         输入 rmd_matrix: (B, 81, 142) - 原始深度数据
#         输出 keys: (B, 81 * 8, 128) - 准备好做 Attention 的 Key
#         """
#         B, N, D = rmd_matrix.shape
#         device = rmd_matrix.device
        
#         rays_data = rmd_matrix[..., :-2]
#         pos_data = rmd_matrix[..., -2:]
#         all_views = []
        
#         # === 1. 循环生成 8 个视角的切片 ===
#         for k in range(self.num_views):
#             # 计算偏移量
#             shift = k * self.shift_step
            
#             # 核心操作：只对射线数据进行循环移位
#             rolled_rays = torch.roll(rays_data, shifts=-shift, dims=-1)
            
#             # 截取窗口
#             local_rays = rolled_rays[..., :self.fov_rays] # (B, 81, 40)
            
#             # === 3. 注入角度信息 ===
#             # 计算当前视角的全局角度
#             angle_val = (2 * np.pi * k) / self.num_views
            
#             sin_val = torch.sin(torch.tensor(angle_val, device=device))
#             cos_val = torch.cos(torch.tensor(angle_val, device=device))
            
#             # 扩展角度编码: (B, 81, 1)
#             sin_enc = sin_val.expand(B, N, 1)
#             cos_enc = cos_val.expand(B, N, 1)
            
#             # === 4. 拼接所有信息 ===
#             #  [局部射线(40), 视角角度(2), 地点坐标(2)] 
#             inp = torch.cat([local_rays, sin_enc, cos_enc, pos_data], dim=-1) 
#             # shape: (B, 81, 140 + 2 + 2) = (B, 81, 144)
            
#             # === 5. 编码 ===
#             encoded_view = self.local_encoder(inp) # (B, 81, 128)
            
#             all_views.append(encoded_view)

#         # === 4. 合并所有视角 ===
#         # 现在的形状是 list of 8 个 (B, 81, 128)
#         # 我们要在 N 的维度上堆叠，变成 (B, 81*8, 128)
#         keys = torch.cat(all_views, dim=1) 
        
#         return keys


class RMDSlidingWindowEncoder(nn.Module):
    def __init__(self, 
                 total_rays=140,      # 纯射线数量 (不含坐标)
                 fov_rays=40,         # 局部视窗大小
                 num_views=8,         # 方向数
                 output_dim=128):     # 输出特征维度
        super().__init__()
        
        self.total_rays = total_rays
        self.fov_rays = fov_rays
        self.num_views = num_views
        
        self.shift_step = int(total_rays / num_views)
        
        # 输入维度 = 局部射线(40) + 角度编码(2)
        # 注意：这里我们不放入坐标，让 MLP 专注于“几何形状”
        input_feat_dim = fov_rays + 2
        
        self.local_encoder = nn.Sequential(
            nn.Linear(input_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, rmd_matrix):
        """
        输入 rmd_matrix: (B, 81, 142) -> 前140是射线，后2是坐标
        输出 features: (B, 81*8, 128)
        输出 coords:   (B, 81*8, 2)   -> 同步扩展好的坐标，方便后面用
        """
        B, N, D = rmd_matrix.shape
        device = rmd_matrix.device
        
        # === 1. 关键修改：拆分射线和坐标 ===
        rays_data = rmd_matrix[..., :self.total_rays] # (B, 81, 140)
        pos_data  = rmd_matrix[..., self.total_rays:] # (B, 81, 2)
        
        all_views_feat = []
        
        # === 2. 循环生成 8 个视角 ===
        for k in range(self.num_views):
            # 计算偏移
            shift = k * self.shift_step
            
            # A. 只对【射线】进行旋转
            rolled_rays = torch.roll(rays_data, shifts=-shift, dims=-1)
            
            # B. 截取窗口
            local_rays = rolled_rays[..., :self.fov_rays] # (B, 81, 40)
            
            # C. 注入角度信息
            angle_val = (2 * math.pi * k) / self.num_views
            sin_enc = torch.full((B, N, 1), math.sin(angle_val), device=device)
            cos_enc = torch.full((B, N, 1), math.cos(angle_val), device=device)
            
            # D. 拼接 (只拼射线和角度)
            inp = torch.cat([local_rays, sin_enc, cos_enc], dim=-1) # (B, 81, 42)
            
            # E. 编码
            encoded_view = self.local_encoder(inp) # (B, 81, 128)
            all_views_feat.append(encoded_view)

        # === 3. 合并特征 ===
        # (B, 648, 128)
        features = torch.cat(all_views_feat, dim=1)
        
        # === 4. 同步扩展坐标 (为了后面加位置编码用) ===
        # 坐标不需要旋转，但需要复制 8 份来对齐特征
        # (B, 81, 2) -> (B, 81, 8, 2) -> (B, 648, 2)
        coords_expanded = pos_data.unsqueeze(2).expand(-1, -1, self.num_views, -1)
        coords_expanded = coords_expanded.reshape(B, -1, 2)
        
        return features, coords_expanded

class RMDmapGlobalEncoder(nn.Module):
    """
    整合了 Sliding Window 和 Self-Attention 的全局编码器
    """
    def __init__(self, total_rays=140, fov_rays=40, num_views=8, output_dim=128):
        super().__init__()
        
        # 1. 几何特征提取器
        self.window_encoder = RMDSlidingWindowEncoder(
            total_rays=total_rays, 
            fov_rays=fov_rays, 
            num_views=num_views, 
            output_dim=output_dim
        )
        
        # 2. 位置编码器 (把 x,y 映射成 embedding)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # 3. Self-Attention (全局交互)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=4, 
            dim_feedforward=output_dim*4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, rmd_matrix):
        # 1. 提取几何特征 & 对齐后的坐标
        # features: [B, 648, 128]
        # coords:   [B, 648, 2]
        features, coords = self.window_encoder(rmd_matrix)
        
        # 2. 生成位置编码
        pos_emb = self.pos_mlp(coords)
        
        # 3. 注入位置信息 (Add)
        # 现在特征 = 几何形状(含朝向) + 全局位置
        features = features + pos_emb
        
        # 4. 全局交互 (让点与点之间建立联系)
        features = self.transformer(features)
        
        return features # [B, 648, 128]
    
class F3MlpDecoder(nn.Module):
    def __init__(self, d_min=0.1, d_max=20, d_hyp=-0.2, D=128, input_dim=128):
        super().__init__()
        
        self.head = nn.Linear(input_dim, D)
        self.d_min, self.d_max, self.d_hyp, self.D, self.input_dim = d_min, d_max, d_hyp, D, input_dim
        
    def forward(self, features): # [B, 40, input_dim] -> [B, 40, 1]
        d_vals = torch.linspace(
            self.d_min ** self.d_hyp, self.d_max ** self.d_hyp, self.D,
            device=features.device
        ) ** (1 / self.d_hyp)
        logits = self.head(features)
        prob = F.softmax(logits, dim=-1)
        pred_d = torch.sum(prob * d_vals, dim=-1)
        return pred_d
    
