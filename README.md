# CUDA 编程学习仓库 (CUDA Programming Learning)

这个仓库用于记录 CUDA 编程学习的过程，包含核心基础概念、核函数编写、性能分析、原子操作、流（Stream）以及 CUDA API 库的实践代码。

---

## 📂 目录结构与学习路线 (Repository Structure)

| 章节 | 目录 | 核心内容 |
| :--- | :--- | :--- |
| **第五章** | `05_Writing_your_First_Kernels/` | CUDA 基础、核函数编写、Profiling 分析、原子操作、Streams |
| **第六章** | `06_CUDA_APIs/` | cuBLAS、cuDNN、cuBLASmp 等官方加速库 |

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

