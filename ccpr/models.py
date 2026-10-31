# -*- coding: utf-8 -*-
"""ResNet18（20×20 单通道输入）+ CS-CBAM 成本敏感注意力 + 投影头。

CS-CBAM 在原生 CBAM 的通道/空间注意力 logit 后引入 λ·log(N/n_c)
类别不平衡偏置项，使注意力权重在训练时自动放大少数类样本的响应。
该偏置是训练期归纳先验，推理时不注入（未知类别）。

模型 forward 返回三元组 (logits, feat, proj)：
  - logits : [B, K] 分类 logits
  - feat   : [B, 512] 全局平均池化后的特征（供对比损失 / t-SNE）
  - proj   : [B, 128] 投影头特征（供 InfoNCE），无投影头时为 None
"""
import torch
import torch.nn as nn


# ============================================================
#  通道 / 空间注意力（输出 sigmoid 前的 logit，便于注入偏置）
# ============================================================
class ChannelAttentionLogit(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )

    def forward(self, x):
        return self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))  # [B,C,1,1]


class SpatialAttentionLogit(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv(torch.cat([avg, mx], dim=1))  # [B,1,H,W]


class CS_CBAM(nn.Module):
    """成本敏感 CBAM：注意力 logit 加 λ·log(N/n_c) 偏置后再 sigmoid。

    mode ∈ {"both", "channel", "spatial"}：串行通道+空间 / 仅通道 / 仅空间，
    用于消融 CBAM 两个子模块各自的贡献。
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7, lam=0.5, mode="both"):
        super().__init__()
        self.lam = lam
        self.mode = mode
        self.ca = ChannelAttentionLogit(in_planes, ratio)
        self.sa = SpatialAttentionLogit(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, class_bias=None):
        # 通道注意力
        if self.mode in ("both", "channel"):
            ca = self.ca(x)
            if class_bias is not None:
                ca = ca + self.lam * class_bias.view(-1, 1, 1, 1)
            x = x * self.sigmoid(ca)
        # 空间注意力
        if self.mode in ("both", "spatial"):
            sa = self.sa(x)
            if class_bias is not None:
                sa = sa + self.lam * class_bias.view(-1, 1, 1, 1)
            x = x * self.sigmoid(sa)
        return x


class SELayer(nn.Module):
    """Squeeze-and-Excitation：通道注意力（全局池化 + 两层 MLP + sigmoid）。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        return x * y


class ECALayer(nn.Module):
    """Efficient Channel Attention：1D 卷积实现局部跨通道交互（无需降维）。"""

    def __init__(self, channels, k=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y


class CoordAtt(nn.Module):
    """Coordinate Attention：沿高/宽两个方向分解注意力，保留位置信息。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                       # [n,c,h,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # [n,c,w,1]
        y = torch.cat([x_h, x_w], dim=2)           # [n,c,h+w,1]
        y = self.act(self.bn1(self.conv1(y)))      # [n,mip,h+w,1]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)              # [n,mip,1,w]
        a_h = self.conv_h(x_h).sigmoid()           # [n,c,h,1]
        a_w = self.conv_w(x_w).sigmoid()           # [n,c,1,w]
        return identity * a_h * a_w


# ============================================================
#  残差块与 ResNet18
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CSBasicBlock(nn.Module):
    """带 CS-CBAM 的残差块（在第二个卷积后、残差相加前插入注意力）。"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 ratio=16, kernel_size=7, lam=0.5, mode="both"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.attn = CS_CBAM(planes, ratio, kernel_size, lam, mode)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, class_bias=None):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out, class_bias)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class AttnBasicBlock(nn.Module):
    """带普通注意力（无类别偏置）的残差块，attn_type ∈ {cbam, se, eca, ca}。"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 attn_type="se", ratio=16, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if attn_type == "cbam":
            self.attn = CS_CBAM(planes, ratio, kernel_size, lam=0.0)
        elif attn_type == "se":
            self.attn = SELayer(planes, ratio)
        elif attn_type == "eca":
            self.attn = ECALayer(planes)
        elif attn_type == "ca":
            self.attn = CoordAtt(planes, ratio)
        else:
            self.attn = None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18CCPR(nn.Module):
    """ResNet18 适配 20×20 单通道输入：首层 3×3/stride1、无 maxpool。

    use_cs_cbam=True 时在 layer3/layer4 的块中插入 CS-CBAM（多尺度：
    中等尺度 + 较大尺度特征），use_proj=True 时加 512→128 投影头。
    """

    def __init__(self, num_classes=6, use_cs_cbam=True, use_proj=True,
                 ratio=16, kernel_size=7, lam=0.5, attn_type=None,
                 in_channels=2, stem="ccpr", attn_stages=(3, 4), attn_mode="both"):
        super().__init__()
        self.in_planes = 64
        self.use_cs_cbam = use_cs_cbam
        self.use_proj = use_proj
        self.stem = stem
        # 注意力类型：use_cs_cbam=True 时强制为成本敏感 CBAM；
        # 否则取 attn_type ∈ {None, "cbam", "se", "eca", "ca"}（普通注意力，无类别偏置）。
        self.attn_type = "cs_cbam" if use_cs_cbam else attn_type
        # 注意力注入的阶段（1-indexed：1=layer1 ... 4=layer4），默认第 3、4 阶段。
        self.attn_stages = tuple(attn_stages)
        self.attn_mode = attn_mode

        # stem="ccpr"：适配 20×20 GAF 输入（3×3/stride1，无 maxpool）；
        # stem="imagenet"：标准 ResNet18 主干（7×7/stride2 + maxpool），适配 224×224 真实图像。
        if stem == "imagenet":
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
            self.maxpool = None
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        def _a(i):
            return self.attn_type if i in self.attn_stages else None

        self.layer1 = self._make_layer(64, 2, stride=1, attn=_a(1), ratio=ratio, kernel_size=kernel_size, lam=lam)
        self.layer2 = self._make_layer(128, 2, stride=2, attn=_a(2), ratio=ratio, kernel_size=kernel_size, lam=lam)
        self.layer3 = self._make_layer(256, 2, stride=2, attn=_a(3), ratio=ratio, kernel_size=kernel_size, lam=lam)
        self.layer4 = self._make_layer(512, 2, stride=2, attn=_a(4), ratio=ratio, kernel_size=kernel_size, lam=lam)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if use_proj:
            self.proj_head = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 128)
            )
        else:
            self.proj_head = None

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride, attn, ratio, kernel_size, lam):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(self._make_block(self.in_planes, planes, stride, downsample, attn, ratio, kernel_size, lam))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(self._make_block(self.in_planes, planes, 1, None, attn, ratio, kernel_size, lam))
        return nn.ModuleList(layers)

    def _make_block(self, in_planes, planes, stride, downsample, attn, ratio, kernel_size, lam):
        if attn == "cs_cbam":
            return CSBasicBlock(in_planes, planes, stride, downsample, ratio, kernel_size, lam, self.attn_mode)
        if attn in ("cbam", "se", "eca", "ca"):
            return AttnBasicBlock(in_planes, planes, stride, downsample, attn, ratio, kernel_size)
        return BasicBlock(in_planes, planes, stride, downsample)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward_layer(self, layer, x, class_bias):
        for block in layer:
            if isinstance(block, CSBasicBlock):
                x = block(x, class_bias)
            else:
                x = block(x)
        return x

    def forward(self, x, class_bias=None):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self._forward_layer(self.layer1, x, None)
        x = self._forward_layer(self.layer2, x, None)
        x = self._forward_layer(self.layer3, x, class_bias)
        x = self._forward_layer(self.layer4, x, class_bias)
        feat = self.avgpool(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        proj = self.proj_head(feat) if self.proj_head is not None else None
        return logits, feat, proj

    def embed(self, x):
        """返回 512 维特征（供 t-SNE 可视化）。"""
        with torch.no_grad():
            _, feat, _ = self.forward(x)
        return feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    for cs in (False, True):
        m = ResNet18CCPR(num_classes=6, use_cs_cbam=cs, use_proj=True)
        x = torch.randn(4, 2, 20, 20)
        cb = torch.rand(4) * 3.0
        logits, feat, proj = m(x, cb)
        print(f"use_cs_cbam={cs}: params={count_parameters(m):.2f}M "
              f"logits={tuple(logits.shape)} feat={tuple(feat.shape)} proj={tuple(proj.shape)}")
