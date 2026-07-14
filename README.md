# CUDA 编程学习仓库 (CUDA Programming Learning)

这个仓库用于记录 CUDA 与 Triton GPU 编程的学习过程，包含核心基础概念、核函数编写、性能分析、原子操作、流（Stream）、CUDA API 库以及 Triton 算子开发的实践代码。

---

## 📂 目录结构与学习路线 (Repository Structure)

| 章节 | 目录 | 核心内容 |
| :--- | :--- | :--- |
| **第五章** | `05_Writing_your_First_Kernels/` | CUDA 基础、核函数编写、Profiling 分析、原子操作、Streams |
| **第六章** | `06_CUDA_APIs/` | cuBLAS、cuDNN、cuBLASmp 等官方加速库 |
| **第七章** | `07_Faster_Matmul/` | 矩阵乘法（SGEMM）性能分阶优化与极致调优 |
| **第八章** | `08_Triton/` | Triton 编程模型、Block-level 算子开发与 GPU 性能对比 |

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
  * [Q3: 什么是统一内存 (Unified Memory)，它有什么优缺点？](./05_Writing_your_First_Kernels/QA.md#q3-什么是统一内存-unified-memory它有什么优缺点)
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
  * [Q6: cuDNN Graph API 如何选择融合执行计划？运行时编译与寄存器数据流是什么关系？](./06_CUDA_APIs/QA.md#q6-cudnn-graph-api-如何选择融合执行计划运行时编译与寄存器数据流是什么关系)
  * [Q7: 什么是 NCHW 与 NHWC 数据排布格式？为什么 NHWC 在部分卷积场景值得优先测试？](./06_CUDA_APIs/QA.md#q7-什么是-nchw-与-nhwc-数据排布格式为什么-nhwc-在部分卷积场景值得优先测试)
* 🏢 **集群与多卡数据中心计算**
  * [Q8: 什么是 cuBLASmp、NCCL 和 MIG？在大型集群或多卡数据中心中，它们各自扮演着什么角色？](./06_CUDA_APIs/QA.md#q8-什么是-cublasmp-nccl-和-mig在大型集群或多卡数据中心中它们各自扮演着什么角色)

---

### 📙 第七章：优化矩阵乘法

> 完整 Q&A 文档：[07_Faster_Matmul/QA.md](./07_Faster_Matmul/QA.md)

* 🧮 **通用矩阵乘法 (GEMM) 优化**
  * [Q1: 什么是全局内存的合并访存 (Coalesced Memory Access)？它的底层硬件机理和优化法则是什么？](./07_Faster_Matmul/QA.md#q1-什么是全局内存的合并访存-coalesced-memory-access它的底层硬件机理和优化法则是什么)
  * [Q2: 什么是 Roofline Model（屋顶线模型）？如何用它判断 Kernel 的性能瓶颈？](./07_Faster_Matmul/QA.md#q2-什么是-roofline-model屋顶线模型如何用它判断-kernel-的性能瓶颈)
  * [Q3: 什么是 Occupancy（占用率）？它和性能是什么关系？](./07_Faster_Matmul/QA.md#q3-什么是-occupancy占用率它和性能是什么关系)
  * [Q4: Shared Memory（共享内存）数组是怎么定义和同步的？](./07_Faster_Matmul/QA.md#q4-shared-memory共享内存数组是怎么定义和同步的)
  * [Q5: 在这个 benchmark 里，我们是怎么从“内存受限”变成“计算受限”的？](./07_Faster_Matmul/QA.md#q5-在这个-benchmark-里我们是怎么从内存受限变成计算受限的)
  * [Q6: 分块 (Blocktiling) 是什么？是 Shared Memory Tiling 吗？](./07_Faster_Matmul/QA.md#q6-分块-blocktiling-是什么是-shared-memory-tiling-吗)
  * [Q7: 如何规避 Shared Memory 的 Bank Conflict（银行冲突）？](./07_Faster_Matmul/QA.md#q7-如何规避-shared-memory-的-bank-conflict银行冲突)
  * [Q8: 解释一下 Thread Coarsening 与 Vectorization（线程粗化与向量化）](./07_Faster_Matmul/QA.md#q8-解释一下-thread-coarsening-与-vectorization线程粗化与向量化)
  * [Q9: 向量化访存 (Vectorized Mem Access, Kernel 6) 是什么？](./07_Faster_Matmul/QA.md#q9-向量化访存-vectorized-mem-access-kernel-6-是什么)
  * [Q10: 循环展开 (#pragma unroll) 的作用和原理是什么？](./07_Faster_Matmul/QA.md#q10-循环展开-pragma-unroll-的作用和原理是什么)
  * [Q11: 双缓冲 (Double Buffering / 软件流水线) 是什么？](./07_Faster_Matmul/QA.md#q11-双缓冲-double-buffering--软件流水线-是什么)
  * [Q12: 如何使用 Nsight Compute (ncu) 对 CUDA Kernel 进行性能分析？](./07_Faster_Matmul/QA.md#q12-如何使用-nsight-compute-ncu-对-cuda-kernel-进行性能分析)

---

### 📙 第八章：Triton 编程与优化

> 完整 Q&A 文档：[08_Triton/QA.md](./08_Triton/QA.md)

* 🐍 **Triton 编程模型与底层原理**
  * [Q1: CUDA 与 Triton 的编程模型有什么本质区别？什么是“Block-level”编程？](./08_Triton/QA.md#q1-cuda-与-triton-的编程模型有什么本质区别什么是block-level编程)
  * [Q2: 一个 Triton Kernel 由哪些部分组成？`@triton.jit`、Grid、Program ID 和 `tl.constexpr` 分别做什么？](./08_Triton/QA.md#q2-一个-triton-kernel-由哪些部分组成tritonjitgridprogram-id-和-tlconstexpr-分别做什么)
  * [Q3: Vector Add 中如何计算 offsets？为什么必须使用 mask？](./08_Triton/QA.md#q3-vector-add-中如何计算-offsets为什么必须使用-mask)
  * [Q4: `tl.load()`、`tl.store()`、pointer tensor、mask 和 `other` 应该怎样理解？](./08_Triton/QA.md#q4-tlloadtlstorepointer-tensormask-和-other-应该怎样理解)

* ⚡ **Softmax、融合与性能**
  * [Q5: 为什么 Softmax 要减去最大值？Triton 版本为什么可能比多个 PyTorch Kernel 更快？](./08_Triton/QA.md#q5-为什么-softmax-要减去最大值triton-版本为什么可能比多个-pytorch-kernel-更快)
  * [Q6: `tl.max()`、`tl.sum()` 如何完成 Reduction？“加载到 SRAM”是否等于手写 Shared Memory？](./08_Triton/QA.md#q6-tlmaxtlsum-如何完成-reduction加载到-sram是否等于手写-shared-memory)
  * [Q7: `BLOCK_SIZE` 为什么常取 2 的幂？当前 Softmax 实现有哪些输入限制？](./08_Triton/QA.md#q7-block_size-为什么常取-2-的幂当前-softmax-实现有哪些输入限制)

* 🧪 **验证、Benchmark 与工程实践**
  * [Q8: Triton Kernel 如何与 PyTorch Tensor 集成？为什么返回结果时 Kernel 可能仍未完成？](./08_Triton/QA.md#q8-triton-kernel-如何与-pytorch-tensor-集成为什么返回结果时-kernel-可能仍未完成)
  * [Q9: 如何正确验证和 Benchmark Triton Kernel？GB/s 应该怎样计算？](./08_Triton/QA.md#q9-如何正确验证和-benchmark-triton-kernelgbs-应该怎样计算)
  * [Q10: `BLOCK_SIZE`、`num_warps`、`num_stages` 和 Autotune 有什么关系？](./08_Triton/QA.md#q10-block_sizenum_warpsnum_stages-和-autotune-有什么关系)
  * [Q11: Triton Kernel 应该如何调试？](./08_Triton/QA.md#q11-triton-kernel-应该如何调试)
  * [Q12: 对 AI Infra Engineer，学完本章必须具备哪些能力？什么时候应该使用 Triton？](./08_Triton/QA.md#q12-对-ai-infra-engineer学完本章必须具备哪些能力什么时候应该使用-triton)
