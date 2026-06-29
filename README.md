# CUDA 编程学习仓库 (CUDA Programming Learning)

这个仓库用于记录 CUDA 编程学习的过程，包含核心基础概念、核函数编写、性能分析、原子操作及流（Stream）等实践代码。

---

## 💡 CUDA 学习问答索引 (QA Quick Links)

为了方便快速复习与查阅关键概念，可以直接点击下方链接跳转到 [QA.md](./QA.md) 中的对应详细解答：

* 🌐 **GPU 维度与坐标索引**
  * [Q1: 如何在 3D Grid 和 3D Block 的配置下计算全局线程 ID (Global Thread ID)？](./QA.md#q1-如何在-3d-grid-和-3d-block-的配置下计算全局线程-id-global-thread-id)
  * [Q2: CUDA 内置变量（如 `blockIdx`、`threadIdx` 等）是哪里来的？](./QA.md#q2-cuda-内置变量如-blockidxthreadidx-等是哪里来的)

* 💾 **内存管理与访存优化**
  * [Q3: 什么是统一内存 (Unified Memory / UMA)，它有什么优缺点？](./QA.md#q3-什么是统一内存-unified-memory--uma它有什么优缺点)
  * [Q7: 什么是全局内存合并访问 (Coalesced Memory Access)？](./QA.md#q7-什么是全局内存合并访问-coalesced-memory-access)

* ⚙️ **执行架构与调度**
  * [Q4: 什么是 Warp（线程束）？为什么它的物理大小是 32？](./QA.md#q4-什么是-warp线程束为什么它的物理大小是-32)
  * [Q5: CUDA Kernel（核函数）和普通 CPU Function（函数）有什么区别？](./QA.md#q5-cuda-kernel核函数和普通-cpu-function函数有什么区别)
  * [Q8: 请用通俗的比喻解释 CUDA 软件层（线程）与硬件层（显卡芯片）的完整架构映射？](./QA.md#q8-请用通俗的比喻解释-cuda-软件层线程与硬件层显卡芯片的完整架构映射)

* ⚠️ **同步与错误处理**
  * [Q6: 为什么核函数执行后立即调用 `cudaGetLastError()` 无法捕获异步执行中的运行时错误？](./QA.md#q6-为什么核函数执行后立即调用-cudagetlasterror-无法捕获异步执行中的运行时错误)
  * [Q9: `cudaDeviceSynchronize()`、`__syncthreads()` 和 `__syncwarps()` 三种同步函数有什么区别和联系？](./QA.md#q9-cudadevicesynchronize__syncthreads-和-__syncwarps-三种同步函数有什么区别-和-联系)

* 🔒 **原子操作 (Atomics)**
  * [Q10: 什么是 CUDA 原子操作 (Atomic Operations)？为什么需要它？有哪些主要函数及代价？](./QA.md#q10-什么是-cuda-原子操作-atomic-operations为什么需要它有哪些主要函数及代价)
  * [Q11: 关于 CUDA 原子操作 (Atomic Operations)，在实际开发和面试中需要掌握到什么程度？](./QA.md#q11-关于-cuda-原子操作-atomic-operations在实际开发和面试中需要掌握到什么程度)

* 🔀 **流与异步并发 (Streams)**
  * [Q12: 结合 Thread、Block、Grid 解释 CUDA Stream (流) 的底层工作原理与软硬件映射？](./QA.md#q12-结合-threadblockgrid-解释-cuda-stream-流的底层工作原理与软硬件映射)

---

## 📂 目录结构与学习路线 (Repository Structure)
* `01 CUDA Basics`: 介绍 CUDA 的基础定位、线程划分与全局坐标计算。
* `02 Kernels`: 核函数编写与 CPU/GPU 向量加法、矩阵乘法的性能比较。
* `03 Profiling`: GPU 性能分析工具的使用（NVTX，naive vs tiled matmul 等）。
* `04 Atomics`: 原子操作的原理及数据竞争示例。
* `05 Streams`: 利用流实现数据传输与计算的异步重叠。
