import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the project's existing implementations
from swin_hsi_model_base import PatchExtract, SwinTransformer


class HSISwinTiny4Stage(nn.Module):
    """Four-stage Swin-Tiny baseline adapted to HSI using the SAME HSI operations you already use.

    HSI处理流程（与HSISwinModel一致）:
        x(B,C,H,W)
        -> (training时) 随机裁剪到img_size + 随机水平翻转
        -> (必要时) padding到pad_size（保证4阶段下采样链路合法）
        -> PatchExtract(2x2) 得到 [B,N, 4*C]
        -> Linear投影到96 + LayerNorm
        -> reshape成 [B,96,Hp,Wp]
        -> SwinTransformer(4 stages) -> logits

    说明：
      - 4-stage Swin 需要 token 网格尺寸能被 2^(num_layers-1)=8 整除。
        你原始 25x25 + 2x2 unfold 会产生 12x12 token，无法走到第3/4阶段。
        因此这里仅在forward里对裁剪后的输入做padding到pad_size(默认32)，
        使得 token 网格变为 16x16，四阶段下采样为 16->8->4->2->1，可正常训练。
      - 未引入任何额外“谱域Stem/新模块”，只使用你已有的PatchExtract+proj+norm。
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_chans: int = 30,
        img_size: int = 25,
        pad_size: int = 32,
        pretrained: bool = True,
        drop_rate: float = 0.03,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.img_size = img_size
        self.pad_size = pad_size

        # 你现有的 HSI patch 提取逻辑：2x2 unfold
        self.patch_extract = PatchExtract(patch_size=(2, 2))
        self.patch_size = 2

        # 这里不是简单 img_size//2，而是以pad_size为准确保4-stage可行
        # token_grid = pad_size/patch_size
        if pad_size % self.patch_size != 0:
            raise ValueError(f"pad_size({pad_size}) must be divisible by patch_size({self.patch_size}).")
        self.patches_per_dim = pad_size // self.patch_size

        # 与HSISwinModel一致：2*2*in_chans -> 96
        self.patch_dim = self.patch_size * self.patch_size * in_chans
        self.proj = nn.Linear(self.patch_dim, 96)
        self.norm = nn.LayerNorm(96)

        # 4-stage Swin-Tiny-like backbone（尽量贴近官方配置，但复用你项目内Swin实现）
        self.backbone = SwinTransformer(
            img_size=self.patches_per_dim,  # 16
            patch_size=1,                   # 仍然保留你当前做法：不在backbone里再切patch
            in_chans=96,
            num_classes=num_classes,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            attn_drop_rate=drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_rate,
        )

        # 仍沿用你原本的初始化逻辑
        self.apply(self._init_weights)

    def forward(self, x):
        # 保持你原来的None输入处理（用于SNN/占位）
        if x is None:
            return self.backbone(None)

        # 与HSISwinModel一致的训练时增强
        if self.training:
            B, C, H, W = x.shape
            if H > self.img_size:
                h_start = torch.randint(0, H - self.img_size + 1, (1,), device=x.device)
                w_start = torch.randint(0, W - self.img_size + 1, (1,), device=x.device)
                x = x[:, :, h_start:h_start + self.img_size, w_start:w_start + self.img_size]
            if torch.rand(1, device=x.device) > 0.5:
                x = torch.flip(x, [3])

        # 为了让4-stage可跑，padding到pad_size（不引入新模块，只是必要的尺寸对齐）
        B, C, H, W = x.shape
        if H != self.pad_size or W != self.pad_size:
            if H > self.pad_size or W > self.pad_size:
                # 若用户输入大于pad_size，做中心裁剪到pad_size（避免改变训练代码接口）
                h0 = max((H - self.pad_size) // 2, 0)
                w0 = max((W - self.pad_size) // 2, 0)
                x = x[:, :, h0:h0 + self.pad_size, w0:w0 + self.pad_size]
            else:
                pad_h = self.pad_size - H
                pad_w = self.pad_size - W
                # pad顺序: (left,right,top,bottom)；这里采用右/下padding，保持简单且可重复
                x = F.pad(x, (0, pad_w, 0, pad_h))

        # PatchExtract: [B, N, patch_dim]
        patches = self.patch_extract(x)

        # proj + norm: [B, N, 96]
        patches = self.proj(patches)
        patches = self.norm(patches)

        # reshape to NCHW for backbone patch_embed
        B, N, Cemb = patches.shape
        Hp = Wp = int(N ** 0.5)
        x_reshaped = patches.permute(0, 2, 1).reshape(B, Cemb, Hp, Wp)

        return self.backbone(x_reshaped)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
