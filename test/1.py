import tensorflow as tf
# 判断GPU是否可用
flag = tf.test.is_gpu_available()
if flag:
    # 获取GPU信息
    print("CUDA可使用")
    gpu_device_name = tf.test.gpu_device_name()
    print("GPU型号： ", gpu_device_name)
else:
    print("CUDA不可用")
import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
    print("GPU型号： ",torch.cuda.get_device_name())
else:
    print("CUDA不可用")