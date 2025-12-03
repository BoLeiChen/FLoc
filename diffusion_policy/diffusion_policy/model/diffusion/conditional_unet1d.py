from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim, # (!!!) 这是 time_feature 的维度 (dsed)
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True, # (!!!) 这个模块明确使用了 scale/bias
                 per_ray_cond_dim=None): # (!!!) 新增参数: 逐射线条件的维度 (128)
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        # --- 修改 FiLM 部分 ---
        # 1. 时间条件编码器 (保持不变)
        time_cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.time_cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, time_cond_channels),
            # (!!!) 输出 B x C 或 B x 2C，不再需要 Rearrange('batch t -> batch t 1')
        )

        # 2. 新增：逐射线条件编码器 (如果存在)
        self.per_ray_cond_dim = per_ray_cond_dim
        if per_ray_cond_dim is not None:
            ray_cond_channels = out_channels * 2 if cond_predict_scale else out_channels
            self.ray_cond_encoder = nn.Sequential(
                nn.Mish(),
                # (!!!) 输入维度是 per_ray_cond_dim (128)
                nn.Linear(per_ray_cond_dim, ray_cond_channels),
                # (!!!) 输入将是 B x T x C_cond, 输出需要 B x T x C' 或 B x T x 2C'
            )
        else:
            self.ray_cond_encoder = None
        # --- FiLM 部分修改结束 ---

        # 残差连接 (保持不变)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
            
    def forward(self, x, time_feature, global_cond=None):
            '''
            x : [ B x in_channels x T ] # T 可能是 40, 20, ...
            time_feature : [ B x cond_dim=dsed ]
            global_cond : [ B x per_ray_cond_dim=128 x 40 ] # 始终是原始长度 40
            returns:
            out : [ B x out_channels x T ]
            '''
            out = self.blocks[0](x) # [B, out_channels, T]
            current_horizon = out.shape[-1] # 获取当前特征图的序列长度 T

            # --- 融合时间条件 (全局) ---
            time_embed = self.time_cond_encoder(time_feature)
            time_embed = time_embed.unsqueeze(-1) # -> [B, C' or 2C', 1]

            # --- 融合逐射线条件 (局部) ---
            ray_embed = None
            # (!!!) 新增检查: 只有当 global_cond 存在且序列长度匹配时才计算和融合
            if global_cond is not None and self.ray_cond_encoder is not None \
            and global_cond.shape[-1] == current_horizon: # <--- 检查长度是否为 T
                
                ray_embed_input = global_cond.permute(0, 2, 1) # -> [B, T, 128]
                ray_embed = self.ray_cond_encoder(ray_embed_input) # -> [B, T, C' or 2C']
                ray_embed = ray_embed.permute(0, 2, 1) # -> [B, C' or 2C', T]
            # (!!!) 如果长度不匹配，ray_embed 将保持为 None

            # --- 融合 scale 和 bias ---
            if self.cond_predict_scale:
                time_scale, time_bias = torch.chunk(time_embed, 2, dim=1) # [B, C', 1]
                final_scale = time_scale
                final_bias = time_bias

                # (!!!) 只有在 ray_embed 被计算出来时才融合它
                if ray_embed is not None:
                    ray_scale, ray_bias = torch.chunk(ray_embed, 2, dim=1) # [B, C', T]
                    # 融合策略
                    final_scale = final_scale + ray_scale # 广播 [B, C', 1] 到 [B, C', T]
                    final_bias = final_bias + ray_bias   # 广播 [B, C', 1] 到 [B, C', T]
                # (!!!) 如果 ray_embed 是 None，则 final_scale/bias 只有时间部分

                out = out * (1 + final_scale) + final_bias

            else: # 只预测 bias (加法融合)
                final_embed = time_embed # [B, C', 1]
                # (!!!) 只有在 ray_embed 被计算出来时才融合它
                if ray_embed is not None:
                    final_embed = final_embed + ray_embed # 广播

                out = out + final_embed

            # --- 继续残差块 ---
            out = self.blocks[1](out)
            out = out + self.residual_conv(x)
            return out

class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None, # 仍然用来获取 per_ray_cond_dim=128
                 diffusion_step_embed_dim=256,
                 down_dims=[256,512,1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False 
                 ):
        super().__init__()
        
        self.down_dims = down_dims
        all_dims = [input_dim] + list(down_dims)
        self.all_dims = all_dims
        start_dim = down_dims[0]
        
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        time_cond_dim = dsed # 用于时间

        # (!!!) 获取逐射线条件的维度
        per_ray_cond_dim = global_cond_dim # 假设传入的是 128

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # (!!!) local_cond_encoder: 确保传递正确的 time_cond_dim 和 per_ray_cond_dim
        local_cond_encoder = None
        if local_cond_dim is not None:
             _, dim_out = in_out[0]
             dim_in = local_cond_dim
             local_cond_encoder = nn.ModuleList([
                 ConditionalResidualBlock1D(
                     dim_in, dim_out, cond_dim=time_cond_dim,
                     kernel_size=kernel_size, n_groups=n_groups,
                     cond_predict_scale=cond_predict_scale,
                     per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
                 ConditionalResidualBlock1D(
                     dim_in, dim_out, cond_dim=time_cond_dim,
                     kernel_size=kernel_size, n_groups=n_groups,
                     cond_predict_scale=cond_predict_scale,
                     per_ray_cond_dim=per_ray_cond_dim)  # <--- 传递
             ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=time_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=time_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                per_ray_cond_dim=per_ray_cond_dim)  # <--- 传递
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=time_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=time_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=time_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=time_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    per_ray_cond_dim=per_ray_cond_dim), # <--- 传递
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self,
                sample: torch.Tensor, # 预期 B C T = [B, 1, 40]
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond=None, **kwargs): # global_cond 是 [B, 40, 128]
        """
        sample: (B, input_dim, T=40)
        timestep: (B,) or int
        local_cond: (B, local_cond_dim, T=40)
        global_cond: (B, T=40, per_ray_cond_dim=128) - 注意输入形状!
        output: (B, T=40, input_dim) or (B, T=40)
        """
        # 确保 sample 形状是 B C T, C=input_dim=1, T=40
        if sample.shape[-1] != self.down_dims[0]: # 最好用更可靠的方法检查
            if sample.dim() == 2: # [B, 40]
                sample = sample.unsqueeze(1) # -> [B, 1, 40]
            elif sample.dim() == 3 and sample.shape[-1] == 1: # [B, 40, 1]
                sample = sample.permute(0, 2, 1) # -> [B, 1, 40]
        assert sample.shape[1] == self.all_dims[0]

        # 1. 时间嵌入 (全局)
        timesteps = timestep
        # ... (处理 timesteps) ...
        timesteps = timesteps.expand(sample.shape[0])
        time_feature = self.diffusion_step_encoder(timesteps) # [B, dsed]

        # (!!!) 2. 移除旧的全局条件拼接
        # global_feature = time_feature
        # if global_cond is not None:
        #     global_feature = torch.cat([...])

        # (!!!) 3. 调整 global_cond 形状为 B C_cond T 以便残差块处理
        if global_cond is not None:
            # 输入: [B, 40, 128] (B T C_cond)
            global_cond = global_cond.permute(0, 2, 1) # -> [B, 128, 40] (B C_cond T)

        # (!!!) local_cond 部分 (如果需要，传递正确的 time_feature 和 global_cond)
        h_local = list()
        if local_cond is not None:
             # 假设 local_cond 已经是 B C T
             resnet, resnet2 = self.local_cond_encoder
             x_local = resnet(local_cond, time_feature=time_feature, global_cond=global_cond)
             h_local.append(x_local)
             x_local = resnet2(local_cond, time_feature=time_feature, global_cond=global_cond)
             h_local.append(x_local)

        x = sample
        h = []

        # --- Down path ---
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            # (!!!) 传递 time_feature 和 global_cond
            x = resnet(x, time_feature=time_feature, global_cond=global_cond)
            # if idx == 0 and len(h_local) > 0: x = x + h_local[0] # local cond 融合
            x = resnet2(x, time_feature=time_feature, global_cond=global_cond)
            h.append(x)
            x = downsample(x)

        # --- Middle path ---
        for mid_module in self.mid_modules:
            x = mid_module(x, time_feature=time_feature, global_cond=global_cond)

        # --- Up path ---
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            skip = h.pop()
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, time_feature=time_feature, global_cond=global_cond)
            # if idx == len(self.up_modules)-1 and len(h_local) > 0: x = x + h_local[1] # local cond 融合
            x = resnet2(x, time_feature=time_feature, global_cond=global_cond)
            x = upsample(x)

        # --- Final Layer ---
        x = self.final_conv(x) # [B, input_dim, 40]

        # --- 调整输出形状 ---
        x = x.permute(0, 2, 1) # -> [B, 40, input_dim]
        if x.shape[-1] == 1:
             x = x.squeeze(-1) # -> [B, 40]

        return x