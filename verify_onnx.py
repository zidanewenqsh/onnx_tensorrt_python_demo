#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import onnx
# 加载 ONNX 模型
onnx_model = onnx.load("resnet18.onnx")

# 检查模型是否有损坏并且所有的节点和初始值都有正确的 schema
onnx.checker.check_model(onnx_model)

# 打印模型的信息（可选）
print(onnx.helper.printable_graph(onnx_model.graph))
# 获取模型的输入和输出信息
input_all = [node.name for node in onnx_model.graph.input]
output_all = [node.name for node in onnx_model.graph.output]

print("Inputs: ", input_all)
print("Outputs: ", output_all)
