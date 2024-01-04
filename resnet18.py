#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.models as models

## 加载一个预训练的 PyTorch 模型
# model = torchvision.models.resnet18(pretrained=True)
# 使用 weights 参数加载预训练模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 设置为评估模式
model.eval()

# 创建一个输入张量（例如，一个具有3个通道、高度224、宽度224的图像）
# 这里的尺寸取决于模型的输入要求
# x = torch.randn(1, 3, 224, 224, requires_grad=True)
# x = torch.randn(1, 3, 32, 32, requires_grad=True)
x = torch.randn(1, 3, 512, 512, requires_grad=True)
y = model(x)
print(y.shape)