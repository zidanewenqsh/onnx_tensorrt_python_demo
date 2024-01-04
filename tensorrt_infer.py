#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA，管理CUDA设备的上下文

def load_engine(trt_file_path):
    """加载TensorRT引擎"""
    with open(trt_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers1(engine):
    """为引擎的输入和输出分配缓冲区"""
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream
def allocate_buffers(engine):
    """为引擎的输入和输出分配缓冲区"""
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        # 使用EXPLICIT_BATCH标志
        binding_shape = engine.get_tensor_shape(binding)
        print(f"binding shape: {binding_shape}")
        if len(binding_shape) == 4:  # 如果是显式批次大小
            size = trt.volume(binding_shape)  # 不需要乘以最大批次大小
        else:
            size = trt.volume(binding_shape) * engine.max_batch_size

        # 使用get_tensor_dtype
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        print(f"{size=}")
        # exit()
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
        # if engine.get_tensor_mode(binding) == trt.TensorMode.INPUT:
        #     inputs.append({'host': host_mem, 'device': device_mem})
        # else:
        #     outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def infer1(engine, inputs, outputs, bindings, stream, input_data):
    """执行推理"""
    # 将输入数据复制到设备
    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # 执行推理
    with engine.create_execution_context() as context:
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

    # 返回输出
    return outputs[0]['host']

def infer(engine, inputs, outputs, bindings, stream, input_data):
    """执行推理"""
    # 将输入数据复制到设备
    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # 执行推理
    with engine.create_execution_context() as context:
        # 使用execute_async_v2
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

    # 返回输出
    return outputs[0]['host']

def resnet18_inference(trt_file_path, input_data):
    """使用TensorRT运行ResNet18模型推理"""
    engine = load_engine(trt_file_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 假设input_data是预处理后的输入数据
    result = infer(engine, inputs, outputs, bindings, stream, input_data)
    return result
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    对输入图像进行预处理以适用于ResNet18模型。

    参数:
    image_path (str): 图像文件的路径。

    返回:
    numpy.ndarray: 预处理后的图像数据。
    """
    # 加载图像
    img = Image.open(image_path)

    # 调整图像尺寸
    img = img.resize((224, 224))

    # 将图像转换为numpy数组
    img_array = np.array(img, dtype=np.float32)

    # 归一化处理
    # ImageNet预训练模型的平均值和标准差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 标准化图像
    img_array = (img_array / 255.0 - mean) / std

    # 调整维度顺序为CHW（通道、高度、宽度）
    img_array = img_array.transpose(2, 0, 1)

    return img_array

# 使用示例
# processed_image = preprocess_image('path_to_your_image.jpg')

# 使用示例
# 假设input_data是预处理后的输入数据，形状和类型与模型输入兼容
# result = resnet18_inference('resnet18.trt', input_data)
if __name__ == '__main__':
    img_path = "cat01.jpg"
    input_data = preprocess_image(img_path)
    result = resnet18_inference('resnet18.trt', input_data)
    print(type(result))
    print(result.shape)
    cls = np.argmax(result, axis=0)
    print(cls)
