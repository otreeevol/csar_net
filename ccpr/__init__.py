# -*- coding: utf-8 -*-
"""控制图模式识别（CCPR）小样本实验包。

实现"三级长尾校准"统一框架（TLC-ResNet18）：
  - 损失级：LDAM 间隔损失
  - 特征级：CS-CBAM 成本敏感注意力
  - 表征级：CFC 对比特征凝聚（InfoNCE）
"""
