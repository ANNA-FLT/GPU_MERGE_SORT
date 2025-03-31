# Merge Sort
归并排序的串行、并行和两种优化并行
## 运行步骤
- 检查GPU：运行 nvidia-smi 看看你的系统是否有可用的 NVIDIA GPU。
- 安装 CUDA Toolkit：确保你的 CUDA 版本与你的 GPU 驱动兼容，可以用 nvcc --version 检查 CUDA 版本。
- 进入Merge目录，执行下列语句：
  - nvcc -o merge_sort kernel.cu
  - ./merge_sort
 
## 参考
https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort.git


　
