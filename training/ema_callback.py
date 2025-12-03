import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from diffusers.training_utils import EMAModel  # <--- 核心改变 1

_logger = logging.getLogger(__name__)

class EMACallback(Callback):
    """
    EMA Callback using HuggingFace Diffusers' EMAModel implementation.
    Optimized for Diffusion models.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema_model = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        # <--- 核心改变 2: 初始化方式不同
        # diffusers 的 EMA 需要传入 parameters, 并且建议传入 model_cls 和 config 以便更好地处理保存
        # 假设你的核心模型在 pl_module.model 或直接就是 pl_module
        # 这里为了通用，我们直接对 pl_module 进行 EMA
        self.ema_model = EMAModel(
            pl_module.parameters(), 
            decay=self.decay,
            model_cls=type(pl_module), 
            model_config=None # 如果有 config 最好传进来，没有也行
        )
        print(f"EMA initialized with decay {self.decay}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # <--- 核心改变 3: 更新方式
        # diffusers 使用 .step(parameters)
        if self.ema_model is not None:
            self.ema_model.step(pl_module.parameters())

    def on_validation_start(self, trainer, pl_module):
        if self.ema_model is not None:
            # <--- 核心改变 4: 这里的逻辑被大大简化了
            # diffusers 内置了 store (存原参数) 和 copy_to (应用 EMA 参数)
            self.ema_model.store(pl_module.parameters())
            self.ema_model.copy_to(pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        if self.ema_model is not None:
            # <--- 核心改变 5: 恢复原参数
            self.ema_model.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        if self.ema_model is not None and self.use_ema_weights:
            self.ema_model.copy_to(pl_module.parameters())
            msg = "Model weights replaced with the EMA version."
            # log_main_process(_logger, logging.INFO, msg) # 你的 logger 封装
            print(msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema_model is not None:
            # 直接存 diffusers 的 state_dict，包含 shadow params 和 step 计数
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        if "ema_state_dict" in callback_state:
            # 需要先确保 ema_model 被初始化了 (通常 on_fit_start 会做，但 resume 时可能需要手动处理一下)
            if self.ema_model is None:
                 self.ema_model = EMAModel(pl_module.parameters(), decay=self.decay)
            
            self.ema_model.load_state_dict(callback_state["ema_state_dict"])
            print("EMA state loaded from checkpoint.")

    # 你的 store, restore, copy_to 方法全都可以删掉了
    # 因为 diffusers.training_utils.EMAModel 已经内置了这些