import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    把 Int 类型的 timestep 转成 Float 类型的 Embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.ndim > 1:
            x = x.squeeze(-1)
            
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :] 
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PointWiseDiffusionNet(nn.Module):
    def __init__(self, action_dim=1, cond_dim=128, time_emb_dim=256):
        super().__init__()
        
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        input_dim = action_dim + cond_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim) 
        )

    def forward(self, sample, timestep, global_cond=None):
        """
        sample:   [B, 40] 或 [B, 40, 1]
        timestep: [B] 或 [B, 1] (整数类型)
        global_cond: [B, 40, 128]
        """
        
        sample = sample.unsqueeze(-1) 

        B, H, A = sample.shape       
        _, _, C = global_cond.shape  
        
        sample_flat = sample.reshape(B * H, A)     # [B*40, 1]
        cond_flat = global_cond.reshape(B * H, C)  # [B*40, 128]
        

        t_emb = self.time_emb(timestep) 
        
        # 然后再过 MLP
        t_emb = self.time_mlp(t_emb)
        
        # 扩展到每个时间步 [B, 1, 256] -> [B, 40, 256] -> [B*40, 256]
        t_emb = t_emb.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
        
        # 3. 拼接
        input_feat = torch.cat([sample_flat, cond_flat, t_emb], dim=-1)
        
        # 4. 预测
        out = self.net(input_feat) 
        
        # 5. 还原形状
        out = out.reshape(B, H, A)

        out = out.squeeze(-1) 
            
        return out