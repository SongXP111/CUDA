# 第七章 Q&A：矩阵乘法性能优化 (07 Faster Matmul)

本文档记录第七章学习过程中遇到的常见问题与核心概念。

---

## 目录 (Table of Contents)

以下问题按推荐的学习顺序排列，分为三个阶段：

#### 第一阶段：基础概念（优化之前必须理解的底层原理）
- [Q1: 什么是全局内存的合并访存 (Coalesced Memory Access)？它的底层硬件机理和优化法则是什么？](#q1-什么是全局内存的合并访存-coalesced-memory-access它的底层硬件机理和优化法则是什么)
- [Q2: 什么是 Roofline Model（屋顶线模型）？如何用它判断 Kernel 的性能瓶颈？](#q2-什么是-roofline-model屋顶线模型如何用它判断-kernel-的性能瓶颈)
- [Q3: 什么是 Occupancy（占用率）？它和性能是什么关系？](#q3-什么是-occupancy占用率它和性能是什么关系)
- [Q4: Shared Memory（共享内存）数组是怎么定义和同步的？](#q4-shared-memory共享内存数组是怎么定义和同步的)

#### 第二阶段：优化主线（从 Naive 到接近 cuBLAS 的完整链路）
- [Q5: 在这个 benchmark 里，我们是怎么从“内存受限”变成“计算受限”的？](#q5-在这个-benchmark-里我们是怎么从内存受限变成计算受限的)
- [Q6: 分块 (Blocktiling) 是什么？是 Shared Memory Tiling 吗？](#q6-分块-blocktiling-是什么是-shared-memory-tiling-吗)
- [Q7: 如何规避 Shared Memory 的 Bank Conflict（银行冲突）？](#q7-如何规避-shared-memory-的-bank-conflict银行冲突)
- [Q8: 解释一下 Thread Coarsening 与 Vectorization（线程粗化与向量化）](#q8-解释一下-thread-coarsening-与-vectorization线程粗化与向量化)
- [Q9: 向量化访存 (Vectorized Mem Access, Kernel 6) 是什么？](#q9-向量化访存-vectorized-mem-access-kernel-6-是什么)
- [Q10: 循环展开 (#pragma unroll) 的作用和原理是什么？](#q10-循环展开-#pragma-unroll-的作用和原理是什么)
- [Q11: 双缓冲 (Double Buffering / 软件流水线) 是什么？](#q11-双缓冲-double-buffering-/-软件流水线-是什么)

#### 第三阶段：性能分析（定位瓶颈、验证优化效果）
- [Q12: 如何使用 Nsight Compute (ncu) 对 CUDA Kernel 进行性能分析？](#q12-如何使用-nsight-compute-ncu-对-cuda-kernel-进行性能分析)

---

### Q1: 什么是全局内存的合并访存 (Coalesced Memory Access)？它的底层硬件机理和优化法则是什么？

合并访存 (Coalesced Memory Access) 是 GPU 全局内存 (Global Memory / VRAM) 读写优化中最核心的概念。

当一个 Warp（32 个线程）同时发起全局内存读写请求时，硬件会把请求合并为满足该访问所需的最少内存事务数。连续且对齐的地址通常能减少事务和无效搬运；事务粒度与计算能力和缓存路径有关。

#### 1. 底层硬件机制：为什么需要合并？
在 GPU 的物理硬件层面，显存数据传输不是一次只读写一个 4 字节的浮点数，而是以 **32 字节、64 字节或 128 字节的显存段 (Memory Segment)** 为基本单位进行物理传输的。

* **合并访存（理想情况）**：
  如果一个 Warp 中的 32 个线程分别访问连续的地址（例如线程 0 读 `ptr[0]`，线程 1 读 `ptr[1]` ... 线程 31 读 `ptr[31]`），这 32 个单精度浮点数总共占用 `32 * 4 = 128` 字节。对计算能力 6.0 及以上的常见路径，这通常由 **4 个 32-byte transaction** 服务；关键是四个段都被充分使用，而非把它记成固定的“1 次 128-byte 事务”。
* **非合并访存（糟糕情况）**：
  如果这 32 个线程访问的地址是分散的（例如线程之间有跨度），显存控制器被迫发起 **32 次独立的显存段传输事务**。为了帮每个线程拿它要的 4 字节数据，GPU 实际搬运了 `32 * 32 = 1024` 字节的数据，显存总线的**有效利用率骤降至 12.5%**，其余 87.5% 的带宽全部被浪费了。

#### 2. 一个通俗的比喻：快递大卡车
我们可以把 GPU 的**显存控制器**想象成一辆**快递大卡车**，把 **32 个线程**想象成 **32 个订购了快递的客户**：
* **合并访存**：这 32 个客户都住在**同一栋公寓楼的相邻房间**。卡车只需要开到这栋楼前，把一整车箱的快递卸下来（一次 128 字节传输），32 个人就能同时拿到自己的快递。效率极高。
* **非合并访存**：这 32 个客户散落在**城市的不同角落**。卡车不得不装上快递，挨个敲门送货，在全城跑了 32 趟。虽然送的货物总量一模一样，但卡车的里程和时间开销暴增了 32 倍。

#### 3. 在代码中如何实现合并访存？
要在 CUDA 编程中实现合并访存，通常遵循以下法则：

* **法则 A：让最内层索引与 `threadIdx.x` 线性对齐**：
  ```cuda
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = input[idx]; // 线程 0, 1, 2 访问连续物理内存，完美合并！
  ```
  避免使用跨步索引，例如 `int idx = threadIdx.x * stride;`（其中 `stride > 1`）。
* **法则 B：对于二维矩阵，按行分配线程计算**：
  Warp 内的线程应该共同横向扫过矩阵的行（即列索引与 `threadIdx.x` 相关），而不是纵向扫过列（这会导致跨行的大步长访存，无法合并）。
* **法则 C：使用共享内存 (Shared Memory) 作为中转站**：
  当算法本身物理上要求非连续访问时（如矩阵转置），可以先以合并的方式将数据读入共享内存，在共享内存内重排（共享内存没有合并访问限制，仅有 Bank Conflict），再以合并的方式写回全局内存。

#### 4. 补充：L1/L2 Cache 的角色
在现代 GPU（如 Ampere 及之后的架构）中，全局内存访问实际上会先查询 **L1 Cache**（SM 内部的 Texture/Data Cache）和 **L2 Cache**（全局共享的大容量缓存）。合并访存的优势不仅在于减少物理传输事务的次数，还在于让连续访问更容易命中 Cache Line（通常为 128 字节），从而避免反复从 DRAM 取数据。非合并的分散访问会导致每个线程的请求触及不同的 Cache Line，大幅降低缓存命中率。


---

### Q2: 什么是 Roofline Model（屋顶线模型）？如何用它判断 Kernel 的性能瓶颈？

**Roofline Model（屋顶线模型）** 是分析 GPU/CPU 程序性能瓶颈的标准框架。它通过一张图，直观地告诉你：**你的 Kernel 当前是被"算力"还是被"带宽"卡住了，以及离硬件极限还有多远。**

#### 1. 核心概念

Roofline 模型基于两个关键指标：

* **算术强度 (Arithmetic Intensity, AI)**：每从内存搬运 1 字节数据，能够执行多少次浮点运算。单位为 **FLOPs/Byte**。
  ```
  AI = 总浮点运算次数 (FLOPs) / 总内存传输字节数 (Bytes)
  ```
* **可达性能上限 (Attainable Performance)**：硬件在给定算术强度下能够提供的最大 FLOPS。

#### 2. "屋顶"长什么样？

Roofline 图的 X 轴是算术强度 (FLOPs/Byte)，Y 轴是性能 (GFLOPs/s)：

```
性能 (GFLOPS/s)
     │              ╱‾‾‾‾‾‾‾‾‾‾‾‾‾  ← 算力天花板 (Peak Compute)
     │            ╱
     │          ╱
     │        ╱  ← 带宽斜坡 (Memory Bandwidth)
     │      ╱
     │    ╱
     │  ╱
     │╱
     └──────────────────────────── 算术强度 (FLOPs/Byte)
              ↑
         拐点 (Ridge Point)
```

* **左侧斜坡区**（算术强度低）：性能被**显存带宽**限制，称为 **Memory-Bound（内存受限）**。无论计算单元多强，数据搬不上来就白搭。
* **右侧平顶区**（算术强度高）：性能被**计算单元峰值**限制，称为 **Compute-Bound（计算受限）**。数据已经足够快地喂到计算单元。
* **拐点 (Ridge Point)**：两条线的交汇处。算术强度恰好让带宽和算力同时满载。

#### 3. 如何使用 Roofline 分析 SGEMM

以本仓库的 SGEMM 优化为例：

| 阶段 | 算术强度 (AI) | 瓶颈区域 | 优化方向 |
|:---|:---|:---|:---|
| Naive (Kernel 1) | ~0.25 FLOPs/Byte | 深度 Memory-Bound | 提升数据复用 |
| SMEM Tiling (Kernel 3-5) | ~32 FLOPs/Byte | 越过拐点，进入 Compute-Bound | 优化指令效率 |
| Warptiling (Kernel 10) | ~32 FLOPs/Byte | 贴近算力天花板 | 榨干 ALU 利用率 |

#### 4. Nsight Compute 的 Roofline 视图

在 NVIDIA Nsight Compute 中，可以直接生成 Roofline 图：
```bash
ncu --set full -o profile_output ./sgemm <kernel_number>
```
然后在 Nsight Compute GUI 中打开 `profile_output.ncu-rep`，切换到 **Roofline** 标签页，即可看到你的 Kernel 在图上的位置（一个点），以及它距离屋顶线（硬件极限）的差距。


---

### Q3: 什么是 Occupancy（占用率）？它和性能是什么关系？

**Occupancy（占用率）** 是 GPU 性能调优中的核心资源指标，定义为：

> **Occupancy = SM 上活跃的 Warp 数量 / SM 最大可支持的 Warp 数量**

例如，如果一个 SM 最多支持 64 个 Warp，而你的 Kernel 只激活了 32 个，那么 Occupancy 就是 50%。

#### 1. 三大资源约束

SM 能同时调度多少个 Block，受以下三个物理资源的**最短板**限制：

* **寄存器数量 (Registers per SM)**：
  每个线程使用的寄存器越多，能同时运行的线程就越少。例如 SM 有 65536 个寄存器，每线程用 64 个，则最多只能有 `65536 / 64 = 1024` 个线程 = 32 个 Warp。
* **共享内存容量 (Shared Memory per SM)**：
  每个 Block 使用的 SMEM 越多，SM 能放下的 Block 就越少。例如 SM 有 48KB SMEM，每个 Block 用 32KB，则最多只能容纳 1 个 Block。
* **每个 SM 的最大 Block/Warp 数量**：
  硬件限制了每个 SM 最多有多少个 Block（如 32 个）和 Warp（如 64 个），即使寄存器和 SMEM 还有余量。

#### 2. Occupancy 和性能的关系：并不是越高越好！

这是一个常见误区。更高的 Occupancy **不一定**意味着更好的性能：

* **高 Occupancy 的优势**：更多 Warp 意味着调度器有更多的候选 Warp 来**隐藏访存延迟**。当一个 Warp 在等待数据时，调度器可以切换到其他就绪的 Warp 继续执行。
* **低 Occupancy 的优势**：如果我们降低 Occupancy（比如通过增大每线程的分块大小 TM×TN），每个线程可以使用**更多寄存器**来缓存数据，从而减少对 SMEM 和全局内存的访问。这在计算受限的场景下反而更快。

在本仓库的 SGEMM 中，Kernel 5 (2D Blocktiling) 使用了 `__launch_bounds__` 来限制每个 Block 的线程数，主动降低了 Occupancy，但因为更高的寄存器复用率，性能反而更好。

#### 3. 如何计算和优化 Occupancy

* **CUDA Occupancy Calculator**：NVIDIA 提供了 [Excel 表格工具](https://developer.nvidia.com/cuda-occupancy-calculator) 或使用 API `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` 在运行时计算。
* **`__launch_bounds__` 指令**：在核函数声明时使用，告知编译器每个 Block 的最大线程数和期望的最小 Block 数，帮助编译器优化寄存器分配。
  ```cuda
  __global__ void __launch_bounds__(256, 1) myKernel(...) { ... }
  // 256: 每个Block最多256线程；1: 每个SM至少1个Block
  ```
* **开发建议**：先以合理的 Occupancy（50%~75%）为起点，再根据 Profiling 结果决定是否需要牺牲 Occupancy 换取更多寄存器复用。


---

### Q4: Shared Memory（共享内存）数组是怎么定义和同步的？

共享内存（Shared Memory）是位于 SM（Streaming Multiprocessor）内部的高速片上缓存，访问延迟极低，主要用于同一个线程块（Thread Block）内部的数据复用与线程协作。

#### 1. 共享内存数组的定义方式

主要有两种定义方式：**静态分配**与**动态分配**。

* **静态分配（Static Allocation）**：
  在编译时就必须确定数组的大小。直接在核函数内使用 `__shared__` 修饰符声明即可。
  ```cuda
  template <const int BM, const int BN, const int BK>
  __global__ void sgemmKernel(...) {
      // 编译时通过模板参数确定 As 和 Bs 的大小
      __shared__ float As[BM * BK];
      __shared__ float Bs[BK * BN];
  }
  ```
* **动态分配（Dynamic Allocation）**：
  在运行时根据输入参数动态指定大小。使用 `extern __shared__` 关键字声明，且数组大小留空。在 Host 端启动 Kernel 时，通过三括号 `<<<...>>>` 的**第三个参数**传入分配的共享内存字节数（Bytes）。
  ```cuda
  // Kernel 内部声明
  __global__ void myKernel(float *d_in) {
      extern __shared__ float s_array[]; // 动态共享内存入口地址
      int tid = threadIdx.x;
      s_array[tid] = d_in[tid];
  }

  // Host 端调用（分配 threadsPerBlock 个 float 的空间）
  int sharedMemBytes = threadsPerBlock * sizeof(float);
  myKernel<<<gridSize, threadsPerBlock, sharedMemBytes>>>(d_in);
  ```
  *(注：如果同一个 Kernel 内动态声明多个不同类型的数组，它们会共享同一个起始地址，需通过指针偏移手动切分。)*

#### 2. 共享内存数组的同步机制

由于共享内存在同一个 Block 内的所有线程之间是共享的，并发读写同一块地址会带来**数据竞争**，因此必须引入同步。

* **块级同步：`__syncthreads()`**：
  最核心的同步屏障。执行到这一步的线程会暂停，直到**该 Block 内的所有线程**都到达该点。它同时保证了之前的所有内存读写对块内所有线程皆可见。
  典型的**“加载-同步-计算-同步”**模式（以本项目 `5_kernel_2D_blocktiling.cuh` 为例）：
  ```cuda
  // 1. 各个线程协作把数据从全局内存加载到 As 和 Bs
  As[(innerRowA + loadOffset) * BK + innerColA] = A[...];
  Bs[(innerRowB + loadOffset) * BN + innerColB] = B[...];
  
  // 2. 必须同步，确保 As 和 Bs 已全部写入完成，防止慢线程读取到垃圾数据
  __syncthreads(); 
  
  // 3. 各线程计算/消费 As 和 Bs 中的数据
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      sum += regM[i] * regN[j];
  }
  
  // 4. 再次同步，确保所有线程都计算完毕，防止下一轮循环的加载过早覆盖当前数据
  __syncthreads();
  ```
  > [!WARNING]
  > `__syncthreads()` 不能放在包含分支分化（如 `if-else`）的分支内部，除非所有线程都必然会进入该分支，否则会导致死锁。

* **Warp 级同步：`__syncwarp(mask)`**：
  若仅需要 Warp（32个线程）内部协作，可使用更轻量的 `__syncwarp()` 仅同步 Warp 内指定掩码的线程，开销远小于 `__syncthreads()`。此外，将指针声明为 `volatile` 可强制每次读写都直达共享内存而绕过寄存器缓存。

* **异步拷贝同步（Ampere 及以上）**：
  现代架构提供了异步拷贝指令（如 `cp.async`），配合屏障（`cuda::barrier`）或协作组，在数据从全局内存异步拷入共享内存的过程中，允许计算单元同时并行计算，隐藏访存延迟。

---

### Q5: 在这个 benchmark 里，我们是怎么从“内存受限”变成“计算受限”的？

在这个 Benchmark 的演进中，核心手段是**利用共享内存 (Shared Memory) 和分块 (Tiling) 技术，大幅提高数据的复用率**。我们可以将这个过程拆解为以下几个关键阶段：

#### 1. 起点：为什么 Naive 实现是“内存受限”？（Kernel 1）
在**朴素实现 (Naive)** 中，计算矩阵 C 的某个元素时，线程会直接去读取矩阵 A 的一整行和矩阵 B 的一整列。
* **计算强度极低**：每做 1 次乘法加法运算（对应 2 个 FLOPs），就需要从全局内存 (VRAM) 读取 2 个单精度浮点数（对应 8 字节）。计算强度仅为 0.25 FLOPs/Byte。
* **瓶颈所在**：现代 GPU 的计算能力（如数十 TFLOPS）远远大于显存带宽（如 1 TB/s）。因为计算强度太低，GPU 计算单元总是在**闲置等待数据从显存搬运过来**。此时系统完全被全局内存的带宽卡了脖子。

*(注：Kernel 2 全局内存合并访存 提高了带宽的实际利用率，但并未改变 0.25 的理论计算强度，所以依然是内存受限)*

#### 2. 破局与跨越：分块与共享内存缓存（Kernel 3, 4, 5）
要打破内存受限，唯一的办法是**减少对全局内存的读取次数**。我们通过引入**分块 (Blocktiling)** 实现了这一点：

* **缓存到 SMEM**：我们将矩阵划分为一个个小块（比如 BM 乘 BN 的大小）。一个线程块 (Thread Block) 会协作把 A 的分块和 B 的分块，从慢速的全局内存，统一读入到极速的**共享内存 (Shared Memory)** 中。
* **巨大的数据复用**：假设一个线程块负责计算 BM 乘 BN 的 C 矩阵块（如 128 乘 128）。沿 K 维度，每次读入一对 A 块 (BM 乘 BK) 和 B 块 (BK 乘 BN)：
  * 每个 K 维分块，读入的数据量正比于：BK × (BM + BN)。
  * 进行的计算量正比于：BM × BN × BK × 2（每个乘加 = 2 FLOPs）。
  * 二者的比值中 BK 约掉，计算强度 ≈ `(BM × BN) / (2 × (BM + BN))` FLOPs/Byte。
  * 当 BM = BN = 128 时：`128 × 128 / (2 × 256) = 32 FLOPs/Byte`。
* **计算强度飙升**：通过分块，**每一个从显存读上来的数据，都在共享内存里被复用了约 BM（或 BN）次**！相比 Naive 的 0.25 FLOPs/Byte，计算强度翻了 **128 倍**，达到了 **32 FLOPs/Byte** 的量级。
* **成功转换**：此时，全局内存带宽的压力被大幅释放，取而代之的是，GPU 的计算单元开始满负荷连轴转。这时候，我们就成功**突破了内存墙，转变为了计算受限 (Compute-Bound)**。

#### 3. 计算受限下的极致压榨（Kernel 5 - 11）
一旦转变为了计算受限，我们的优化方向就从“如何快点搬数据”变成了**“如何让计算核心一刻也不停歇，并且减少多余的指令浪费”**：

* **寄存器分块 (Register Tiling / 2D Blocktiling, Kernel 5)**：让每个线程负责计算 TM×TN（如 8×8）的微型块，把所需的 A、B 数据读入最快的**寄存器 (Registers)** 中复用，极大地消除了对共享内存的重复读取指令，让算术逻辑单元 (ALU) 全力做乘加。
* **向量化访存 (Vectorized Mem Access, Kernel 6)**：使用 float4 一次提取 4 个浮点数。这不仅提高了带宽，更重要的是**减少了 75% 的访存指令数量**，把宝贵的指令发射槽位让给计算指令。
* **自动调优 (Autotuned, Kernel 9)**：在 2D Blocktiling 基础上，通过系统性搜索 BM/BN/BK/TM/TN 等参数组合，找到特定 GPU 上的最优配置。
* **Warp 级分块 (Warptiling, Kernel 10)**：在寄存器分块的基础上，将 Thread Block 的大块先显式拆分给不同的 Warp，Warp 内部再进行线程级分块，实现更精细的调度和更高的数据局部性。
* **双缓冲 (Double Buffering, Kernel 11)**：用两份共享内存 tile 交错安排加载和计算，创造访存与计算重叠的机会。本仓库的 Kernel 11 由两组线程发出普通 load/compute 指令，并非 `cp.async`；重叠程度和能否隐藏延迟必须用 profiler 验证。

**总结**：从内存受限到计算受限的转折点发生在 **Kernel 3 到 Kernel 5 的分块阶段**。在此之前，我们在和总线带宽做斗争；在此之后，我们在和指令调度、流水线延迟与寄存器数量做斗争。


---

### Q6: 分块 (Blocktiling) 是什么？是 Shared Memory Tiling 吗？

简单来说，**是的，Blocktiling（块级平铺/分块）在物理层面上就是 Shared Memory Tiling（共享内存分块）**。

在 CUDA SGEMM 的上下文中，这两个术语通常指代同一个优化概念，只是侧重点不同：
* **Blocktiling**：从**线程组织（网格划分）**的角度命名。它指的是我们将输出矩阵 C 划分成一个个大小为 BM 乘 BN 的矩阵块，并把计算每一个矩阵块的任务分配给一个 **Thread Block（线程块）**。
* **Shared Memory Tiling**：从**数据缓存（存储介质）**的角度命名。为了让这个 Thread Block 能够计算这个分块，我们需要把矩阵 A 和 B 的数据块先搬运到 **Shared Memory（共享内存）** 中进行缓存。

---

为了理清概念，CUDA 矩阵乘法中其实有**三个不同层级**的“分块 (Tiling)”，它们对应的缓存介质和粒度也不同：

#### 1. 块级分块 (Blocktiling / SMEM Tiling)
* **粒度**：Thread Block 级别（通常是 128 乘 128 或 64 乘 64 的大块）。
* **介质**：**Shared Memory（共享内存）**。
* **做法**：一个 Thread Block协作，将全局内存中的 A 块和 B 块读入到共享内存，然后在共享内存中重复使用这些数据。

#### 2. 线程级分块 (Threadtiling / Register Tiling)
* **粒度**：单个 Thread 级别（通常是 8 乘 8 或 4 乘 4 的微型块）。
* **介质**：**Registers（寄存器）**。
* **做法**：如果一个线程只计算 C 矩阵的一个元素，它每次都要去共享内存读 A 和 B。如果我们让一个线程负责计算一个 8 乘 8 的微型块，那它就可以把这 8 行和 8 列的数据读入到自己的寄存器里，在寄存器中进行 8 乘 8 = 64 次乘加运算。这极大地减少了对共享内存（SMEM）的访问。

#### 3. Warp 级分块 (Warptiling)
* **粒度**：Warp 级别（32个线程组成的线程束，通常负责 64 乘 64 或者是 64 乘 32 的中型块）。
* **介质**：**寄存器 + 共享内存 + 寄存器（双层结构）**。
* **做法**：介于 Blocktiling 和 Threadtiling 之间。在更高级的优化中（如 Tensor Core 编程），我们不直接给单个线程分块，而是先将 Blocktiling 的大块拆分给不同的 Warp（Warp 级分块），Warp 内部再进行线程级分块。这能够有效利用 Warp 内部的洗牌指令 (Shuffle) 提升效率。


---

### Q7: 如何规避 Shared Memory 的 Bank Conflict（银行冲突）？

在 CUDA 编程中，**Bank Conflict（银行冲突 / 存储体冲突）** 是影响共享内存 (Shared Memory, SMEM) 访问性能的最常见瓶颈。

在当前 `SGEMM_CUDA` 的 Benchmark 中，主要采用了**两种**经典的方案来规避它：**Padding（填充列）** 和 **Swizzling（重排/对角线化线性映射）**。

---

#### 1. 什么是 Bank Conflict？

在硬件层面，共享内存被均分为 **32 个同样大小的、可独立访问的内存模块**，称为 **Banks（存储体）**。
* **物理规则**：在一个 Warp（32个线程）中，如果多个线程**同时访问同一个 Bank 中的不同地址**，这些访问就无法并发执行，必须被串行化（排队），这就发生了 **Bank Conflict**。
* **无冲突情况**：
  * 32 个线程访问 32 个**不同**的 Banks（完美并行）。
  * 32 个线程同时访问同一个 Bank 的**同一个**地址（触发**广播/Broadcast**机制，无冲突且极快）。

---

#### 2. 规避方案 A：Padding（填充列） —— 最常用、最简单 (对应 Kernel 8)

在 `8_kernel_bank_extra_col.cuh` 中采用的就是这种方案。

##### 💡 原理：
共享内存中，连续的 4 字节（一个 float）依次映射到 Bank 0, 1, 2, …, 31, 0, 1, 2, … 如此循环。如果我们的矩阵分块宽度是 `BN = 128`，因为 128 是 32 的整数倍，**每行末尾恰好落在 Bank 31，下一行的首元素又回到了 Bank 0**。这意味着所有行的第 j 列元素都落在同一个 Bank 上（比如第 0 行首元素在 Bank 0，第 1 行首元素也在 Bank 0，第 2 行首元素还是 Bank 0...）。当不同的线程同时访问不同行的同一列时，它们计算出的索引对应的 Bank 就会产生冲突。

**Padding 的做法是：在每一行的末尾塞入多余的“填充列 (Extra Columns)”。**

##### 🛠️ 代码实现：
```cuda
// 1. 申请共享内存时，每行额外多申请 5 列（一般取奇数，如 1 或者是 5 等）
const int extraCols = 5;
__shared__ float Bs[BK * (BN + extraCols)]; // 原本是 BK * BN

// 2. 读写索引都加上这个偏移量
Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
...
regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
```

##### ⚡ 效果：
通过将一行的物理宽度从 `128` 强行变成 `133`（128 + 5），第二行的首元素就会从原先的 Bank 0 错位移动到 Bank 5，第三行的首元素移动到 Bank 10。
这样，**原本垂直对齐、映射到同一个 Bank 的数据在物理上被错开分摊到了不同的 Banks 中**，以极小的空间开销避开了冲突。

---

#### 3. 规避方案 B：Swizzling（地址重排） (对应 Kernel 7)

在 `7_kernel_resolve_bank_conflicts.cuh` 中使用的是这种方案。

##### 💡 原理：
Padding 虽然简单，但它会**浪费一部分共享内存空间**（本例中浪费了 5/128 约为 4% 的共享内存），这在共享内存极度紧张的核函数中可能会降低 Occupancy（占用率）。

**Swizzling（重排）通过修改读写索引的数学映射公式，不浪费任何内存空间，直接在逻辑上打乱访存模式，使线程访问均匀分散到 32 个 Banks 中。**

可以把 Swizzling 想象成将原本“按行顺序”排列的二维数组，重新“洗牌”成一种类似**对角线交错**的排列方式。这样，原本同一列（映射到同一个 Bank）的元素，在物理上被分散到不同的 Bank 上——类似于把整齐排列的棋盘格旋转了一个角度。

##### 🛠️ 代码实现：
通过位操作或巧妙的取模运算重新映射索引：
```cuda
// 写入共享内存时，将原本线性的 innerColB 重新编排映射
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
...

// 读取时也使用重新计算的索引
regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
```
*(通过逻辑上将一维索引进行变换，让相邻线程在访问相邻数据时，物理地址映射到不同的 Bank 线上。)*

##### ⚡ 效果：
* 优点：**零内存浪费**。
* 缺点：索引计算公式变得非常复杂，增加了一些整数运算指令开销，可读性变差。

---

#### 4. 开发建议

1. **优先使用 Padding**：因为它极其直观，开发和调试成本低。通常只需要将二维共享内存的列宽设置为 `Width + 1`，就能以极小的空间代价干掉大部分 Bank 冲突。
2. **在高级库（如 CUTLASS）中使用 Swizzling**：当性能压榨到极致，且共享内存容量成为 Occupancy 的瓶颈时，会通过底层宏和 Swizzle 模板类实现复杂的无损重排。


---

### Q8: 解释一下 Thread Coarsening 与 Vectorization（线程粗化与向量化）

**线程粗化 (Thread Coarsening)** 与 **向量化 (Vectorization)** 是 GPU 高性能计算（特别是 CUDA 矩阵乘法）中非常核心且经常结合使用的两种优化技术。

简单来说：
* **线程粗化** 改变的是**“每个线程分配的工作量大小”**（即合并多个线程的任务，提高寄存器数据复用，减少冗余开销）。
* **向量化** 改变的是**“单条指令处理的数据宽度”**（即一次指令读写/计算多个数据，榨干硬件带宽和指令发射槽）。

---

#### 1. 线程粗化 (Thread Coarsening)

##### 💡 核心概念：
在最原始的 GPU 编程思维中，我们倾向于“一个线程只计算一个输出元素”（比如 Naive 矩阵乘法，每个线程只算 C[i][j] 的 1 个 float）。
**线程粗化**则是反其道而行之：**将本该由多个线程完成的工作，合并交给同一个线程来串行完成。**

在本仓库中，线程粗化的演进分为两步：
* **Kernel 4 (1D Blocktiling)**：每线程只在 M 维度扩展，负责计算 TM 个元素（1D 粗化）。这是粗化的第一步，已经带来了显著的 A 矩阵数据复用。
* **Kernel 5 (2D Blocktiling)**：进一步在 M 和 N 两个维度同时扩展，每线程负责 TM×TN 个元素（2D 粗化），获得更高的 A 和 B 的双向数据复用率。

##### ⚡ 为什么粗化反而更快？
虽然 GPU 拥有极强的并行度，但如果划分得太细，会带来很多副作用。线程粗化的优势在于：
* **减少冗余访存（数据复用）**：在矩阵乘法中，计算 C[i][j] 和 C[i][j+1] 都需要读取 A 矩阵的第 i 行。如果分配给两个线程，它们会各自从 Shared Memory 读取相同的 A 元素。如果**粗化**为一个线程计算 C 的一个 TM 乘 TN 微型块（即 **Threadtiling / Register Tiling**），该线程只需要将 A 的元素读入寄存器一次，就可以用来跟 B 的多列进行乘加。这大幅减少了对共享内存的访问次数。
* **可能改变同步成本**：线程粗化可能减少完成同一输出所需的线程/Block 数，但不必然减少 `__syncthreads()` 次数或等待时间；是否收益取决于 tile 划分、占用率和寄存器压力。
* **减少线程创建与调度开销**。

##### ⚠️ 缺点/代价：
* **消耗更多寄存器**：粗化线程意味着该线程需要在寄存器中暂存更多中间结果（如 TM 乘 TN 个累加值）。如果粗化度过大，会导致**寄存器溢出 (Register Spilling)** 到慢速内存，或者降低 GPU 的 Block 占用率 (Occupancy)。

---

#### 2. 向量化 (Vectorization)

##### 💡 核心概念：
**向量化**是指利用硬件的 SIMD（单指令多数据）或宽内存总线特性，**用一条指令同时处理多个相邻的数据元素**。

##### 🛠️ 在 CUDA 中的分类：
* **访存向量化 (Memory Vectorization)**：使用 CUDA 内置的向量数据类型（如 float4、double2、int4）。比如用一条 float4 加载指令（128位宽）代替四条传统的 float 加载指令（32位宽）。其优势在于**指令数减少 75%**，极大缓解了 GPU 的指令发射瓶颈，并提升了总线利用效率。
* **计算向量化 (Arithmetic Vectorization)**：利用特定的硬件指令同时计算多个数据。比如使用 half2（一次指令同时对两个半精度浮点数做乘加），或者 DP4A（八位整数点积指令）。

---

#### 3. 它们之间的区别与协同关系

| 维度 | 线程粗化 (Thread Coarsening) | 向量化 (Vectorization) |
| :--- | :--- | :--- |
| **优化维度** | **线程/任务分配维度** (线程做的事情变多了) | **硬件/指令宽度维度** (单条指令变宽了) |
| **主要目标** | 提高数据复用（寄存器），减少共享内存冗余访问 | 减少指令数量，填满硬件访存/计算位宽 |
| **制约瓶颈** | 受限于**寄存器数量 (Registers)** | 受限于**内存对齐 (Alignment)** 和**硬件指令集** |

##### 🤝 完美协同（以本仓库的 Kernel 6 为例）：
在高性能 CUDA SGEMM 中，这两者通常是结合在一起使用的：
1. 我们首先进行 **Thread Coarsening**：让一个线程负责计算 C 的 TM 乘 TN（如 8 乘 8）的微型块。这个线程会把 A 的 8 个元素和 B 的 8 个元素暂存在寄存器中进行复用。
2. 随后我们使用 **Vectorization**：
   * **读取时**，由于线程粗化需要一次性读取多个数据，我们不使用单 float 读取，而是用 float4 向量化读取，一次搬运 4 个元素。
   * **写回时**，计算完的 64 个 C 元素，我们也是通过强转为 float4，以 128 位向量化写回全局内存。


---

### Q9: 向量化访存 (Vectorized Mem Access, Kernel 6) 是什么？

**向量化访存 (Vectorized Memory Access)** 是 CUDA 编程中非常重要的底层性能优化手段。

在 Kernel 6 中，它的核心思想是：**让每个线程在单次指令中读取或写入连续的多个数据（通常是 4 个 float，共 16 字节 / 128 位），而不是用 4 次独立的指令每次只读写 1 个 float。**

在代码层面，这主要是通过将 `float*` 指针强制类型转换为 `float4*` 指针来实现的。

---

#### 1. 向量化访存的实现原理（以 Kernel 6 代码为例）

在 `6_kernel_vectorize.cuh` 中，数据从全局内存搬运到共享内存的代码如下：

```cuda
// 使用 reinterpret_cast 将普通的 float 指针转换为 float4 指针
// float4 是 CUDA 内置的向量类型，包含 x, y, z, w 四个 float 成员（共 128 位）
float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];

// 然后分别写入共享内存中
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
```

对于矩阵 B 的加载也是同理：
```cuda
// 一次性读取 128 位，再一次性写入 128 位
reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
```

#### 2. 向量化访存带来的性能优势

引入向量化访存后，性能通常会有显著提升（在 A6000 上的测试中，性能从 74% 提升到了 78% 以上）。它带来了三个核心好处：

* **减少指令数量 (Instruction Reduction)**：若编译器保留向量化访问，一个线程读 4 个 float 可能由更少的加载指令完成，从而减轻发射压力；但 `float4` 不保证一定生成单条 `LDG.E.128`，编译器可按对齐、别名和目标架构拆分访问。应检查 SASS/PTX 或 Nsight Compute，而不是把 75% 当作保证。
* **提高内存总线吞吐量 (Bus Efficiency)**：GPU 的显存控制器和 L2 缓存是专门针对宽数据传输进行优化的。一次搬运 16 字节（128位）数据，能够更好地填满内存总线的物理通道，使显存的实际带宽利用率逼近理论上限。
* **降低延迟**：发起一次 128 位的请求，其物理延迟与发起一次 32 位的请求几乎是一样的。因此，一次性读取 4 个数相当于用一次延迟的代价完成了四倍的工作，从而隐式地隐藏了访存延迟。

---

#### 3. 使用向量化访存的物理限制

向量化访存并不是无脑就能用的，它有两个强烈的物理约束：

* **内存地址对齐 (Alignment) - 最关键**：
  * 对此处将 `float*` 重解释为 `float4*` 的实现，起始地址必须满足 **16 字节对齐**；否则代码不满足该向量类型的对齐要求。
  * 如果首地址不对齐，程序在运行时会直接触发段错误（Illegal Memory Access）或者降级为极慢的非对齐加载。
  * 因此在代码中，我们计算偏移时会有 `innerColA * 4` 和 `innerColB * 4`，确保每个线程读入的起点都是 4 的倍数个 float。
* **硬件 and 维度对齐要求**：
  * 本仓库这个无 tail-handling 的 kernel 要求相关宽度与 tile 能安全按 4 个 float 处理；通用向量化 kernel 可通过边界分支或标量尾部路径支持非 4 倍数维度。
  * 线程块分配的数据量（如 `BM * BK` 和 `BN * BK`）必须能被 `4 * 线程数` 整除，否则多出来的零散数据无法凑成 `float4` 就会导致编译或运行报错。这也是为什么我们在 `runner.cu` 中会看到各种针对向量化对齐的静态断言检查。


---

### Q10: 循环展开 (#pragma unroll) 的作用和原理是什么？

**循环展开 (Loop Unrolling)** 是将循环体复制多份以减少循环控制开销的编译器优化技术。在 CUDA 编程中，它通过 `#pragma unroll` 指令显式控制。

#### 1. 为什么循环展开能提升性能？

一个普通的 `for` 循环，每次迭代都伴随着：
* **比较指令**：判断 `j < LOOP_COUNT` 是否成立。
* **自增指令**：`j++`。
* **跳转指令**：跳回循环头部。

这些**循环控制指令不产生任何有用的计算**，却占用了宝贵的指令发射槽。展开后，这些开销被大幅消除：

```cuda
// 展开前：每次迭代都有 比较 + 自增 + 跳转 开销
for (int j = 0; j < 4; j++) {
    sum += a[tid] + b[tid];
}

// 展开后：没有循环控制开销，GPU 可以连续发射计算指令
sum += a[tid] + b[tid];
sum += a[tid] + b[tid];
sum += a[tid] + b[tid];
sum += a[tid] + b[tid];
```

展开带来的三个核心好处：
* **减少控制指令**：可消除循环的比较和跳转。若循环计数在 Warp 内一致，它本来通常不会造成分支分歧；收益主要来自控制开销和潜在 ILP，而非“消除分歧”。
* **增加指令级并行 (ILP)**：展开后多条独立指令暴露出来，GPU 的流水线可以同时处理更多操作。
* **寄存器复用机会**：编译器在展开后能看到更大的代码窗口，更好地进行寄存器分配和常量折叠优化；但也可能增加寄存器压力、spill 和指令缓存压力，因此需实测。

#### 2. 在 CUDA 中如何使用

* **完全展开**：当循环次数是编译时常量时，使用 `#pragma unroll` 让编译器将循环完全展开。
  ```cuda
  #pragma unroll
  for (int i = 0; i < TM; ++i) {  // TM 是模板常量
      regM[i] = As[dotIdx * BM + threadRow * TM + i];
  }
  ```
* **部分展开**：指定展开因子，适用于循环次数较大的情况：
  ```cuda
  #pragma unroll 4  // 每次展开 4 次迭代
  for (int j = 0; j < LOOP_COUNT; j++) { ... }
  ```
* **禁止展开**：在某些场景下展开反而有害（如增大代码体积导致指令缓存压力），可以使用 `#pragma unroll 1` 禁止展开。

#### 3. 编译器自动展开 vs 手动声明

在本仓库的 `unrolling_example.cu` 实验中发现，**即使不写 `#pragma unroll`，`nvcc` 编译器在很多情况下也会自动展开循环**（尤其是循环次数为编译时常量且循环体简单时）。

可以通过查看 PTX 汇编来验证编译器是否进行了展开：
```bash
nvcc -ptx unrolling_example.cu -o - | less
```
如果在汇编中看不到循环的分支跳转指令（如 `@p bra`），说明编译器已经自动将循环展开了。

#### 4. 在 SGEMM 中的应用

在本仓库的矩阵乘法 Kernel 中，`#pragma unroll` 主要应用在以下关键内循环中：
* **从 SMEM 加载到寄存器**的循环（`for i in TM / TN`）。
* **内积计算**的循环（`for dotIdx in BK`）。
* **写回结果到全局内存**的循环。

这些循环的迭代次数都是模板常量，编译器可以完全展开，从而将内层计算变成一长串无分支的乘加指令流。


---

### Q11: 双缓冲 (Double Buffering / 软件流水线) 是什么？

在 CUDA 矩阵乘法优化中，**双缓冲 (Double Buffering)**，也被称为**软件流水线 (Software Pipelining)**，是一种**用于隐藏全局内存（显存）访问延迟的经典并行优化技术**。

它的核心思想是：**让“数据搬运（访存）”与“数据计算（算术）”在时间上重叠。** 

当计算单元在计算当前批次（第 k 阶段）的数据时，访存单元已经在后台把下一批次（第 k+1 阶段）的数据从显存异步加载到共享内存中了。

---

#### 1. 传统单缓冲 vs 双缓冲的区别

##### 🔴 传统单缓冲方式（Kernel 3 - 10）：
在之前的 Kernel 中，循环内部的步骤是串行同步的：
1. **等待**：从全局内存（GMEM）加载数据块 `k` 到共享内存（SMEM）。
2. **同步**：调用 `__syncthreads()` 确保数据全部写完。
3. **计算**：从 SMEM 读数据，进行矩阵乘加运算。
4. **同步**：调用 `__syncthreads()` 确保计算完成（防止下一轮的加载覆盖了当前还没算完的数据）。
5. **循环**：进入 `k+1` 阶段。

在这种模式下，GPU 处于**“加载数据（计算闲着） $\rightarrow$ 开始计算（访存闲着） $\rightarrow$ 循环往复”**的交替状态。

##### 🟢 双缓冲方式（Kernel 11）：
双缓冲在共享内存中开辟了**两倍**的空间（Buffer 0 和 Buffer 1）：
```cuda
__shared__ float As[2 * BM * BK]; // 2倍大小的共享内存
__shared__ float Bs[2 * BK * BN];
```
执行流程变为：
1. **预加载**：在循环开始前，先把第 0 阶段的数据加载进 Buffer 0。
2. **循环开始**：
   * **后台访存**：在后台，将第 `k+1` 阶段的数据从显存加载到 **Buffer 1** 中。
   * **前台计算**：与此同时，计算单元读取 **Buffer 0** 中第 `k` 阶段的数据进行乘法累加计算。
3. **切换指针**：当这一轮计算和下一轮的加载都完成后，Buffer 0 和 Buffer 1 **角色互换**。
   * 计算单元改去计算 **Buffer 1** 的数据（第 `k+1` 阶段）。
   * 访存单元改去把第 `k+2` 阶段的数据加载到 **Buffer 0** 中。

##### 📊 流水线时序示意图
```
时间 →     | 阶段 k     | 阶段 k+1   | 阶段 k+2   |
           |            |            |            |
Load(访存)  | ████ B0    |   ████ B1  |   ████ B0  |
Compute    |   ░░░░░░░  | ████ B0    | ████ B1    |
                          ↑ Load 与 Compute 重叠
```
在单缓冲中，Load 和 Compute 是串行交替的；在双缓冲中，它们在时间上重叠，GPU 几乎不存在空闲等待。

---

#### 2. 双缓冲在 Kernel 11 代码中的体现

在 `11_kernel_double_buffering.cuh` 代码中，双缓冲通过**将线程块分为两组、交错安排加载和计算的顺序**来实现流水线重叠。关键点是：**两组线程都参与计算和加载，只是执行顺序不同**。

```cuda
// 1. 将线程块分为两组
bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

// 2. 预加载：组 0 先把第一批数据加载进 Buffer 0
if (doubleBufferIdx == 0) {
    db::loadFromGmem(... As, Bs ...);
}
__syncthreads();

// 3. 主循环，每次步进 2 * BK（一次处理两个缓冲区的数据）
for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // 组 0 的执行顺序：
      db::processFromSmem(... Buffer0 ...);  // ① 计算 Buffer 0
      __syncthreads();                       //    ← 此时组 1 在加载 Buffer 1
      db::processFromSmem(... Buffer1 ...);  // ② 计算 Buffer 1（组 1 已加载完）
      __syncthreads();
      db::loadFromGmem(... Buffer0 ...);     // ③ 为下一轮预加载 Buffer 0
    } else {
      // 组 1 的执行顺序：
      db::loadFromGmem(... Buffer1 ...);     // ① 加载 Buffer 1
      __syncthreads();                       //    ← 此时组 0 在计算 Buffer 0
      db::processFromSmem(... Buffer0 ...);  // ② 计算 Buffer 0
      __syncthreads();
      db::processFromSmem(... Buffer1 ...);  // ③ 计算 Buffer 1
    }
}
```

**核心机制**：在 `__syncthreads()` 同步点上，组 0 在计算时组 1 在加载，组 1 在计算时组 0 在加载，从而实现了“加载”与“计算”的流水线重叠。两组线程最终都会完成所有数据的计算，只是通过时间上的错位安排消除了等待。

#### 3. 双缓冲的优化效果

* **隐藏延迟**：将一部分全局内存访问与计算重叠。只有在资源、同步和调度条件都合适时才可能充分隐藏延迟；计算时间大于加载时间是必要但非充分条件。
* **流水线并行**：让 GPU 内部的访存引擎（LDST 单元）和计算引擎（ALU 单元）同时满载运转，榨干硬件的并发能力。

*(注：在较新的 GPU 架构如 Ampere (SM 8.0) 及之后，NVIDIA 引入了硬件级的异步拷贝指令 `cp.async`，可以直接将数据从全局内存搬运至共享内存而不需要寄存器中转，这使得双缓冲可以在硬件层面更高效、更容易地实现。)*


---

### Q12: 如何使用 Nsight Compute (ncu) 对 CUDA Kernel 进行性能分析？

**NVIDIA Nsight Compute (ncu)** 是 NVIDIA 官方提供的 GPU Kernel 级性能分析工具。它能对**单个 Kernel 的执行**进行极其详细的硬件计数器采集和瓶颈诊断。

#### 1. 基本使用方法

```bash
# 分析指定 Kernel（默认采集基本指标）
ncu ./sgemm <kernel_number>

# 采集完整指标集（包含 Roofline、Memory、Compute 等所有分析）
ncu --set full -o profile_output ./sgemm <kernel_number>

# 只分析第 N 次 Kernel 启动（跳过 warmup）
ncu --launch-skip 5 --launch-count 1 ./sgemm <kernel_number>
```

分析结果可以在终端直接查看，也可以用 Nsight Compute GUI 打开 `.ncu-rep` 文件进行交互式分析。

#### 2. 关键性能指标

在 ncu 的分析报告中，以下指标对 SGEMM 调优最为关键：

| 指标类别 | 关键指标 | 含义 |
|:---|:---|:---|
| **Compute** | SM 利用率 (SM Throughput) | 计算单元的繁忙程度 |
| **Memory** | 显存吞吐 (DRAM Throughput) | 实际显存带宽利用率 |
| **Memory** | L2 命中率 (L2 Hit Rate) | L2 缓存的有效性 |
| **Instruction** | 指令发射效率 (Issue Slot Utilization) | Warp 调度器的指令发射率 |
| **Warp** | Warp Stall 原因分布 | 线程束暂停的原因（等数据？等同步？等指令？） |
| **Occupancy** | 理论 vs 实际 Occupancy | 资源受限情况 |

#### 3. 如何根据指标定位瓶颈

* **如果 DRAM Throughput 接近硬件上限**（如 A100 的 ~2 TB/s 的 80%+）→ **Memory-Bound**。优化方向：增加数据复用（Tiling）、向量化访存（float4）。
* **如果 SM Throughput 很高但 Issue Slot Utilization 不满** → **指令发射瓶颈**。优化方向：减少指令数（向量化、循环展开）、增加 ILP。
* **如果 Warp Stall 主要是 "Wait"（等待 Barrier）** → **同步开销过大**。优化方向：减少 `__syncthreads()` 次数（双缓冲）、调整分块大小。
* **如果 Occupancy 很低且 Warp Stall 主要是 "Long Scoreboard"（等待长延迟操作）** → **延迟隐藏不足**。优化方向：提高 Occupancy 或使用双缓冲来预取数据。

#### 4. 与 Nsight Systems (nsys) 的区别

| 工具 | 分析粒度 | 适用场景 |
|:---|:---|:---|
| **Nsight Systems (nsys)** | 系统级时间线 | 分析 CPU-GPU 交互、Kernel 启动延迟、Stream 并发、整体 Pipeline |
| **Nsight Compute (ncu)** | 单 Kernel 级硬件计数器 | 深入分析单个 Kernel 的计算效率、访存模式、瓶颈原因 |

**开发建议**：先用 `nsys` 确认宏观瓶颈在哪个 Kernel（或 CPU-GPU 数据传输），再用 `ncu` 对具体 Kernel 做深度剖析。
