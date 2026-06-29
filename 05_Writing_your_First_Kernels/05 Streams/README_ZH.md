# CUDA 流示例 (CUDA Streams Examples)

## 💡 直观理解
你可以将“流（Stream）”想象成一条“河流的支流”，其中的操作只能随着时间向前流动（就像一条时间轴）。例如，第一步拷贝数据，第二步进行计算，第三步将结果拷回。这就是流的基本概念。

在 CUDA 中，我们可以同时使用多个流，每个流都有自己独立的时间轴。这使得我们可以将不同的操作（例如数据传输和内核计算）重叠（Overlap）起来，从而更充分地利用 GPU 硬件资源。

* **大模型训练中的应用**：当训练一个超大型语言模型时，如果把大量时间花在“将 Token 数据拷入/拷出 GPU”上，将会导致算力闲置。流允许我们**在 GPU 进行计算的同时，在后台并发进行数据搬运**。流引入了软件层面称为**“预取（Prefetching）”**的抽象，即在数据被计算需要之前提前搬运它，从而完美地隐藏了数据传输的延迟。

本项目旨在展示如何使用 CUDA 流来实现并发执行和提高 GPU 利用率。它包含两个核心示例：

---

## 💻 代码片段与语法说明
* **默认流 (Default Stream)** = 流 0 = 空流 (Null Stream)
  ```cpp
  // 这种核函数启动方式默认使用空流 (0)
  myKernel<<<gridSize, blockSize>>>(args);
  
  // 它完全等价于：
  myKernel<<<gridSize, blockSize, 0, 0>>>(args);
  ```

* **核函数执行配置**：正如在“核函数”章节中介绍的，`<<<gridDim, blockDim, Ns, S>>>` 的各个参数含义为：
  * `gridDim` (dim3)：指定网格的维度和大小。
  * `blockDim` (dim3)：指定每个线程块的维度和大小。
  * `Ns` (size_t)：动态分配的共享内存字节数（通常省略）。
  * `S` (cudaStream_t)：关联的流，是一个可选参数，默认值为 0（代表空流）。

* **流优先级配置**：`stream1` 和 `stream2` 在创建时可以配置不同的优先级，这决定了在运行时计算资源紧张时它们在 GPU 上的执行调度顺序，给予我们对内核并发执行更精细的控制。
  ```cpp
  // 创建具有不同优先级的流
  int leastPriority, greatestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
  ```

---

## 📂 示例文件
1. [01_stream_basics.cu](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/CUDA/05_Writing_your_First_Kernels/05%20Streams/01_stream_basics.cu)：演示基本的流用法，包括异步内存传输和核函数启动。
2. [02_stream_advanced.cu](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/CUDA/05_Writing_your_First_Kernels/05%20Streams/02_stream_advanced.cu) ：演示流优先级、回调函数以及流之间的 Event 依赖关系等高级概念。

---

## 🚀 编译指令
使用以下命令编译示例文件：
```bash
nvcc -gencode arch=compute_120,code=sm_120 -o basics "05 Streams/01_stream_basics.cu"
nvcc -gencode arch=compute_120,code=sm_120 -o advanced "05 Streams/02_stream_advanced.cu"
```

---

## 📂 官方文档参考
* [NVIDIA 官方流与并发研讨课 PDF](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

---

## 💾 锁页内存 (Pinned Memory)
* **直观理解**：相当于向操作系统申请“这块物理内存非常重要，在运行期间锁死在物理插槽上，千万不要进行页面交换或搬动”。
* **原理解析**：锁页内存是锁定在物理内存中不可被操作系统虚拟内存分页机制换入换出的内存。当我们要向 GPU 传输数据或从中读取时，使用锁页内存可以使 DMA（直接内存访问）硬件以极高的带宽在不经过 CPU 介入的情况下完成传输。如果操作系统在传输中途移动了内存页，会导致 GPU 读写到错误的位置，引发段错误（Segfault）。
  ```cpp
  // 分配锁页内存 (Host 端)
  float* h_data;
  CUDA_CHECK(cudaMallocHost((void**)&h_data, size));
  ```

---

## 🚩 事件 (Events)
* **精准测量时间**：在核函数启动前和启动后各记录一个 Event，用来精准测量 GPU 端的实际计算耗时。
* **流间同步**：Event 可以作为同步栅栏，在不同的流之间建立任务依赖，确保某个流的任务在另一个流的特定操作完成后才开始。
* **数据与计算重叠**：Event 可以标记数据传输的完成状态，从而通知计算引擎可以开始对这部分数据进行计算。

```cpp
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start, stream));
kernel<<<grid, block, 0, stream>>>(args);
CUDA_CHECK(cudaEventRecord(stop, stream));

CUDA_CHECK(cudaEventSynchronize(stop)); // 阻塞 CPU，直到 stop 被记录
float milliseconds = 0;
CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); // 计算耗时
```

---

## 📞 回调函数 (Callbacks)
* 使用回调函数，你可以搭建起一个“CPU-GPU 协同流水线”：GPU 上某个流的任务完成，会自动触发 CPU 上的回调函数，CPU 回调函数再根据情况向 GPU 的队列中追加下一批计算任务。
```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // 在这里触发 CPU 端的下一批处理，或者追加新的 GPU 任务
}

kernel<<<grid, block, 0, stream>>>(args);
// 添加回调函数到流中，排在 kernel 之后执行
CUDA_CHECK(cudaStreamAddCallback(stream, MyCallback, nullptr, 0));
```
