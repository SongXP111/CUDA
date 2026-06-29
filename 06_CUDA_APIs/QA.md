# 第六章 Q&A：CUDA API 库 (06 CUDA APIs)

本文档记录第六章学习过程中遇到的常见问题与核心概念，涵盖 cuBLAS、cuDNN、cuBLASmp 等 CUDA 官方加速库的使用、配置、错误检查及性能调优。

---

## 目录 (Table of Contents)
- [Q1: 为什么 cuBLAS 库底层默认要求输入矩阵是列优先 (Column-Major) 存储？在 C/C++ 中该如何应对？](#q1-为什么-cublas-库底层默认要求输入矩阵是列优先-column-major-存储在-cc-中该如何应对)

---

### Q1: 为什么 cuBLAS 库底层默认要求输入矩阵是列优先 (Column-Major) 存储？在 C/C++ 中该如何应对？
**A**:

cuBLAS 默认采用**列优先**存储，主要是一个**历史遗留的技术标准问题**。

#### 1. 历史原因：继承 Fortran 时代的 BLAS 标准
* **BLAS 的起源**：BLAS（Basic Linear Algebra Subprograms，基础线性代数子程序）规范最早诞生于 1979 年，最初是用 **Fortran** 语言实现的。
* **语言存储特性差异**：
  * **Fortran** 语言中，多维数组在内存中是按**列优先**（Column-Major）连续存储的。
  * **C/C++** 语言中，多维数组则是按**行优先**（Row-Major）连续存储的。
* **直接替换（Drop-in Replacement）的设计初衷**：在 NVIDIA 开发 CUDA 和 cuBLAS 时，高性能计算（HPC）和科学计算领域早已形成了以 Fortran 编写的 LAPACK/BLAS 为基石的生态系统。为了让这些巨量的科学计算代码能够在**不修改矩阵数据排布逻辑的前提下，直接迁移到 GPU 上跑**，cuBLAS 从 API 设计到内存布局上都完全模仿了经典的 Fortran BLAS 规范，因而默认采用了列优先规范。

#### 2. C/C++ 程序员的应对策略
在行优先的 C/C++ 中调用列优先的 cuBLAS 时，通常有以下几种处理方式：

* **方法 A：数学转置法（无开销，推荐）**
  利用矩阵乘法的转置等式：
  `C^T = B^T * A^T`
  * 在 C++ 中按行优先存储的数组 `A`，直接传给 cuBLAS 时，在 cuBLAS 眼里它物理上就是一个列优先的 `A^T`。
  * 因此，我们直接通过交换乘法顺序计算 `B * A`，计算出的结果列优先矩阵在 C++ 眼里正好就是行优先的 `A * B`。
  * 这种方法**完全没有任何额外计算开销**，代码实现非常优雅：
    ```cpp
    // 计算 C = A * B，其中 A (M x K), B (K x N), C (M x N) 为行优先
    // 在 cuBLAS 中实际上计算 C^T = B^T * A^T，传入参数顺序交换为 d_B, d_A
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    ```

* **方法 B：显式转置参数法**
  cuBLAS 允许在调用时传入 `CUBLAS_OP_T` 参数，告诉库函数在计算前在硬件上将矩阵进行转置。
  * 缺点：在某些旧架构 GPU 上，硬件转置或非连续维度的计算可能会导致显存访问合并率下降，从而降低实际运行性能。

* **方法 C：使用面向现代 AI 的 cuBLASLt 库**
  随着深度学习（PyTorch/TensorFlow）的爆发，现代 AI 计算几乎全盘采用行优先布局。为此，NVIDIA 推出了 **cuBLASLt** (Lightweight) 库。它允许开发人员通过定义矩阵描述符（Matrix Layout Descriptors），直接声明矩阵是行优先（Row-Major）还是列优先（Column-Major），库底层会自动根据声明在硬件上选择最优的内核（Kernel）去运行，不再强求程序员必须用列优先去适配。


