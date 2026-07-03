# 让我们来优化矩阵乘法

![](assets/comparison.png)

> **朴素实现 (Naive)**：最容易理解但性能最差的实现方式。
> **合并访存 (Coalesced Memory Access)**：确保我们以对 GPU 最优的物理对齐方式加载数据。
> **共享内存 (Shared Memory)**：通过减少对全局内存的访问次数来提升有效显存带宽。
> **一维/二维分块 (1D/2D Blocktiling)**：在网格（Grid）中的所有 SM / 线程块（Block）之间均等分摊工作量。
> **向量化内存访问 (Vectorized Memory Access)**：通过单条指令加载更多数据（例如用 128 位宽指令代替 32 位宽指令）。
> **自动调优 (Autotuning)**：针对您的具体 GPU 硬件架构，通过网格搜索（Grid Search）找到最适合该内核的超参数。
> **cuBLAS**：NVIDIA 官方闭源的密集线性代数加速库，用于高效执行 Matmul 等操作。

**原作者注：因为我也比较懒，所以我们直接跳转到 Simon Boehm 的 [博客](https://siboehm.com/articles/22/CUDA-MMM) 和 [Git 仓库](https://github.com/siboehm/SGEMM_CUDA)**

---

## 行优先 (Row Major) vs 列优先 (Column Major)

* cuBLAS 要求输入矩阵为**列优先**格式，因此我们在计算前必须进行转置。
* **行优先 (Row Major)**：矩阵元素 `A[i][j]` 在一维物理内存中存储在 `A[i * N + j]`。
* **列优先 (Column Major)**：矩阵元素 `A[i][j]` 在一维物理内存中存储在 `A[j * M + i]`。

```python
# 行优先 (Row Major)
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

# 物理内存中的存储顺序：
A = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 列优先 (Column Major)
A = [[1, 4, 7],
     [2, 5, 8],
     [3, 6, 9]]

# 物理内存中的存储顺序：
A = [1, 4, 7, 2, 5, 8, 3, 6, 9]
```

---

## `#pragma unroll` 的作用

* 理想情况下，我们希望在每次循环迭代中执行更多有意义的计算。如果能在单次迭代中执行 4 个数学运算而不是 1 个，那将会非常高效（减少了循环控制开销和分支跳转）。
* 在某些上下文中，即使您没有显式声明，GPU 编译器也会自动展开循环（这就是 `unrolling.cu` 中发生的情况）。
* 您可以通过 `nvcc -ptx v1.cu -o - | less` 命令行输出查看 PTX 汇编代码，以确认编译器是否已经自动展开了该循环。
* 通过编写一个未展开的内核，并与手动展开的内核进行基准测试对比，您可以判断循环展开是否真正带来了性能增益。接下来可以通过查看 PTX 汇编来确认。当您未能获得预期的加速且需要进一步深入调查底层时，这种对比分析是非常有帮助的。
* 要进行快速基准测试，只需记录内核的平均运行时间并与展开版本进行比较。如果展开版本更快，则说明展开是有益的。反之，则无益。在测试中请务必注意验证计算结果的正确性（逐元素对比输出）。

---

## 什么是占用率 (Occupancy)？

> **占用率 (Occupancy)** 被定义为每个 SM 上**活动的 Warp 数量**与该 SM **最大可能支持的活动 Warp 数量**之比。
>
> 限制 SM 加载更多活动线程块（Active Blocks）的主要物理约束有三个：**寄存器数量 (Register count)**、**Warp 数量限制 (Warp count)** 以及 **共享内存容量 (SMEM capacity)**。让我们针对当前的内核来进行一次示例计算。
>
> 详细官方指南：[NVIDIA CUDA 最佳实践：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)

* 相关矩阵乘法性能优化参考：[Matmul Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

---

## 汇编指令 (Assembly Instructions)

* [PTX 指令集官方文档 (Parallel Thread Execution)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-machine-model)
* [如何阅读着色器汇编代码 (SASS) 指南](https://interplayoflight.wordpress.com/2021/04/18/how-to-read-shader-assembly/)

### 我们为什么想要探究或者手动编写汇编代码？

1. 它能让我们看清究竟是哪些物理操作限制了性能（例如：Warp 分歧导致的串行执行、等待显存数据加载入寄存器所导致的延迟、耗时极长的特殊数学操作等）。
2. 它允许我们进行时钟周期级别（Clock-cycle）的硬件极致调优（这是能接触到 GPU 物理硬件底盘的最直接手段）。

---

## 灵感来源

1. [Simon Boehm @ Anthropic](https://siboehm.com/articles/22/CUDA-MMM)
2. [Lei Mao @ NVIDIA](https://github.com/leimao/CUDA-GEMM-Optimization)

---

## 更进一步：

* 如果您想进一步了解 NVIDIA 等公司为了使 cuBLAS能够达到极高 TFLOPS 算力而在**矩阵乘法 (matmul)** 中采用的内核性能优化技术，可以学习 CUTLASS（用于线性代数子程序的 CUDA 模板库）：
  * [CUTLASS Github 仓库](https://github.com/NVIDIA/cutlass)
  * [CUTLASS 官方技术博客](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
  * [CUTLASS 官方文档](https://nvidia.github.io/cutlass/)
