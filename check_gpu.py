# import torch

# if torch.cuda.is_available():
#     print("PyTorch can detect CUDA (GPU is available).")
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}") # 打印第一个 GPU 的名称
#     # 可以选择使用哪个 GPU，例如使用 GPU 0
#     device = torch.device("cuda:0")
#     print(f"Using device: {device}")
# else:
#     print("PyTorch cannot detect CUDA (GPU is not available or not configured correctly).")
#     print("Training will be performed on CPU.")
#     device = torch.device("cpu")
#     print(f"Using device: {device}")

import sys
import torch

# 打印出当前正在使用的Python解释器的完整路径
print("Python Executable:", sys.executable)

# 打印出当前PyTorch库的安装位置
print("PyTorch Path:", torch.__file__)

# 打印PyTorch版本，看是否带有 +cu118
print("PyTorch Version:", torch.__version__)