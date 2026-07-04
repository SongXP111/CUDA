# CUDA 编程学习仓库 (CUDA Programming Learning)

这个仓库用于记录 CUDA 编程学习的过程，包含核心基础概念、核函数编写、性能分析、原子操作、流（Stream）以及 CUDA API 库的实践代码。

---

## 📂 目录结构与学习路线 (Repository Structure)

| 章节 | 目录 | 核心内容 |
| :--- | :--- | :--- |
| **第五章** | `05_Writing_your_First_Kernels/` | CUDA 基础、核函数编写、Profiling 分析、原子操作、Streams |
| **第六章** | `06_CUDA_APIs/` | cuBLAS、cuDNN、cuBLASmp 等官方加速库 |
| **第七章** | `07_Faster_Matmul/` | 矩阵乘法（SGEMM）性能分阶优化与极致调优 |

---

## 💡 Q&A 问答索引 (QA Quick Links)

为了方便快速复习与查阅关键概念，下面按章节列出所有 Q&A 问题的快速跳转链接。

---

### 📘 第五章：编写你的第一个 CUDA 核函数

> 完整 Q&A 文档：[05_Writing_your_First_Kernels/QA.md](./05_Writing_your_First_Kernels/QA.md)

* 🌐 **GPU 维度与坐标索引**
  * [Q1: 如何在 3D Grid 和 3D Block 的配置下计算全局线程 ID (Global Thread ID)？](./05_Writing_your_First_Kernels/QA.md#q1-如何在-3d-grid-和-3d-block-的配置下计算全局线程-id-global-thread-id)
  * [Q2: CUDA 内置变量（如 `blockIdx`、`threadIdx` 等）是哪里来的？](./05_Writing_your_First_Kernels/QA.md#q2-cuda-内置变量如-blockidxthreadidx-等是哪里来的)

* 💾 **内存管理与访存优化**
  * [Q3: 什么是统一内存 (Unified Memory / UMA)，它有什么优缺点？](./05_Writing_your_First_Kernels/QA.md#q3-什么是统一内存-unified-memory--uma它有什么优缺点)
  * [Q7: 什么是全局内存合并访问 (Coalesced Memory Access)？](./05_Writing_your_First_Kernels/QA.md#q7-什么是全局内存合并访问-coalesced-memory-access)

* ⚙️ **执行架构与调度**
  * [Q4: 什么是 Warp（线程束）？为什么它的物理大小是 32？](./05_Writing_your_First_Kernels/QA.md#q4-什么是-warp线程束为什么它的物理大小是-32)
  * [Q5: CUDA Kernel（核函数）和普通 CPU Function（函数）有什么区别？](./05_Writing_your_First_Kernels/QA.md#q5-cuda-kernel核函数和普通-cpu-function函数有什么区别)
  * [Q8: 请用通俗的比喻解释 CUDA 软件层（线程）与硬件层（显卡芯片）的完整架构映射？](./05_Writing_your_First_Kernels/QA.md#q8-请用通俗的比喻解释-cuda-软件层线程与硬件层显卡芯片的完整架构映射)

* ⚠️ **同步与错误处理**
  * [Q6: 为什么核函数执行后立即调用 `cudaGetLastError()` 无法捕获异步执行中的运行时错误？](./05_Writing_your_First_Kernels/QA.md#q6-为什么核函数执行后立即调用-cudagetlasterror-无法捕获异步执行中的运行时错误)
  * [Q9: `cudaDeviceSynchronize()`、`__syncthreads()` 和 `__syncwarp()` 三种同步函数有什么区别和联系？](./05_Writing_your_First_Kernels/QA.md#q9-cudadevicesynchronize__syncthreads-和-__syncwarp-三种同步函数有什么区别和联系)

* 🔒 **原子操作 (Atomics)**
  * [Q10: 什么是 CUDA 原子操作 (Atomic Operations)？为什么需要它？有哪些主要函数及代价？](./05_Writing_your_First_Kernels/QA.md#q10-什么是-cuda-原子操作-atomic-operations为什么需要它有哪些主要函数及代价)
  * [Q11: 关于 CUDA 原子操作 (Atomic Operations)，在实际开发和面试中需要掌握到什么程度？](./05_Writing_your_First_Kernels/QA.md#q11-关于-cuda-原子操作-atomic-operations在实际开发和面试中需要掌握到什么程度)

* 🔀 **流与异步并发 (Streams)**
  * [Q12: 结合 Thread、Block、Grid 解释 CUDA Stream (流) 的底层工作原理与软硬件映射？](./05_Writing_your_First_Kernels/QA.md#q12-结合-threadblockgrid-解释-cuda-stream-流的底层工作原理与软硬件映射)

---

### 📗 第六章：CUDA API 库

> 完整 Q&A 文档：[06_CUDA_APIs/QA.md](./06_CUDA_APIs/QA.md)

* 🧮 **cuBLAS 矩阵计算库**
  * [Q1: 为什么 cuBLAS 库底层默认要求输入矩阵是列优先 (Column-Major) 存储？在 C/C++ 中该如何应对？](./06_CUDA_APIs/QA.md#q1-为什么-cublas-库底层默认要求输入矩阵是列优先-column-major-存储在-cc-中该如何应对)
  * [Q2: 在 GPU 上运行单精度 (SGEMM) 与半精度 (HGEMM) 矩阵乘法，在显存消耗、计算速度以及精度误差上有什么实际差距？（附实验数据）](./06_CUDA_APIs/QA.md#q2-在-gpu-上运行单精度-sgemm-与半精度-hgemm-矩阵乘法在显存消耗计算速度以及精度误差上有什么实际差距附实验数据)
  * [Q3: 什么是 cuBLASLt 和 cuBLASXt？它们与普通的 cuBLAS 有什么区别和优势？](./06_CUDA_APIs/QA.md#q3-什么是-cublaslt-和-cublasxt它们与普通的-cublas-有什么区别和优势)
* 🧠 **cuDNN 深度神经网络加速库**
  * [Q4: 在 AI Infra (AI 工程基建) 领域，对于 cuDNN 必须掌握的核心知识点有哪些？](./06_CUDA_APIs/QA.md#q4-在-ai-infra-ai-工程基建-领域对于-cudnn-必须掌握的核心知识点有哪些)
  * [Q5: 什么是 Pointwise (点对点/逐元素) 算子？为什么它在 GPU 性能调优中如此重要？](./06_CUDA_APIs/QA.md#q5-什么是-pointwise-点对点逐元素-算子为什么它在-gpu-性能调优中如此重要)
  * [Q6: cuDNN Graph API 是如何通过运行时即时编译 (JIT) 实现算子融合并让数据在寄存器中流转的？](./06_CUDA_APIs/QA.md#q6-cudnn-graph-api-是如何通过运行时即时编译-jit-实现算子融合并让数据在寄存器中流转的)
  * [Q7: 什么是 NCHW 与 NHWC 数据排布格式？为什么在 GPU 性能调优中更推荐使用 NHWC (Channels-Last)？](./06_CUDA_APIs/QA.md#q7-什么是-nchw-与-nhwc-数据排布格式为什么在-gpu-性能调优中更推荐使用-nhwc-channels-last)
* 🏢 **集群与多卡数据中心计算**
  * [Q8: 什么是 cuBLASmp、NCCL 和 MIG？在大型集群或多卡数据中心中，它们各自扮演着什么角色？](./06_CUDA_APIs/QA.md#q8-什么是-cublasmp-nccl-和-mig在大型集群或多卡数据中心中它们各自扮演着什么角色)

---

### 📙 第七章：优化矩阵乘法

> 完整 Q&A 文档：[07_Faster_Matmul/QA.md](./07_Faster_Matmul/QA.md)

* 🧮 **通用矩阵乘法 (GEMM) 优化**
  * [Q1: 什么是全局内存的合并访存 (Coalesced Memory Access)？它的底层硬件机理和优化法则是什么？](./07_Faster_Matmul/QA.md#q1-什么是全局内存的合并访存-coalesced-memory-access它的底层硬件机理和优化法则是什么)
  * [Q2: 在这个 benchmark 里，我们是怎么从“内存受限”变成“计算受限”的？](./07_Faster_Matmul/QA.md#q2-在这个-benchmark-里我们是怎么从内存受限变成计算受限的)
  * [Q3: 分块 (Blocktiling) 是什么？是 Shared Memory Tiling 吗？](./07_Faster_Matmul/QA.md#q3-分块-blocktiling-是什么是-shared-memory-tiling-吗)
  * [Q4: 向量化访存 (Vectorized Mem Access, Kernel 6) 是什么？](./07_Faster_Matmul/QA.md#q4-向量化访存-vectorized-mem-access-kernel-6-是什么)
  * [Q5: 双缓冲 (Double Buffering / 软件流水线) 是什么？](./07_Faster_Matmul/QA.md#q5-双缓冲-double-buffering--软件流水线-是什么)
  * [Q6: 如何规避 Shared Memory 的 Bank Conflict（银行冲突）？](./07_Faster_Matmul/QA.md#q6-如何规避-shared-memory-的-bank-conflict银行冲突)
  * [Q7: 解释一下 Thread Coarsening 与 Vectorization（线程粗化与向量化）](./07_Faster_Matmul/QA.md#q7-解释一下-thread-coarsening-与-vectorization线程粗化与向量化)
  * [Q8: Shared Memory（共享内存）数组是怎么定义和同步的？](./07_Faster_Matmul/QA.md#q8-shared-memory共享内存数组是怎么定义和同步的)
  * [Q9: 什么是 Roofline Model（屋顶线模型）？如何用它判断 Kernel 的性能瓶颈？](./07_Faster_Matmul/QA.md#q9-什么是-roofline-model屋顶线模型如何用它判断-kernel-的性能瓶颈)
  * [Q10: 什么是 Occupancy（占用率）？它和性能是什么关系？](./07_Faster_Matmul/QA.md#q10-什么是-occupancy占用率它和性能是什么关系)
  * [Q11: 循环展开 (#pragma unroll) 的作用和原理是什么？](./07_Faster_Matmul/QA.md#q11-循环展开-pragma-unroll-的作用和原理是什么)
  * [Q12: 如何使用 Nsight Compute (ncu) 对 CUDA Kernel 进行性能分析？](./07_Faster_Matmul/QA.md#q12-如何使用-nsight-compute-ncu-对-cuda-kernel-进行性能分析)


