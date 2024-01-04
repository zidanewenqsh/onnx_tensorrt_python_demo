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
x = torch.randn(1, 3, 224, 224, requires_grad=True)

# 导出模型
torch_out = torch.onnx.export(model,  # 运行模型
                              x,  # 模型输入 (或一个元组，用于多个输入)
                              "resnet18.onnx",  # 输出文件名
                              export_params=True,  # 存储模型的训练参数
                              opset_version=10,  # ONNX 版本
                              do_constant_folding=True,  # 优化模型
                              input_names=['input'],  # 输入名
                              output_names=['output'],  # 输出名
                              dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                                            'output': {0: 'batch_size'}})
