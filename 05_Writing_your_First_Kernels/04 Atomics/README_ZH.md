# 什么是原子操作 (What are Atomic Operations)

“原子（Atomic）”在物理学中指代“不可分割的最小物质概念”，在计算机领域中指代“不可被中断的一组操作”。

**原子操作（Atomic Operation）**能够确保当一个线程对某个内存位置进行“读取-修改-写回”操作时，在整个动作完成之前，其它任何线程都不能访问或修改同一个内存位置。这可以防止多线程并发时产生的数据竞争（Race Conditions）。

因为原子操作在执行期间会限制其它线程对同一块内存的并发访问，所以它在一定程度上会**牺牲程序执行的速度**。这是一种由 GPU 硬件保证的数据安全机制，代价是部分并行的丢失。

---

### 🔢 整型原子操作 (Integer Atomic Operations)

* **`atomicAdd(int* address, int val)`**：原子加法。将 `val` 加到 `address` 指向的值上，并返回旧值（即操作前的值）。
* **`atomicSub(int* address, int val)`**：原子减法。
* **`atomicExch(int* address, int val)`**：原子交换。直接将 `address` 指向的值替换为 `val`，并返回旧值。
* **`atomicMax(int* address, int val)`**：原子最大值。将 `address` 的值设为当前值与 `val` 的较大者。
* **`atomicMin(int* address, int val)`**：原子最小值。
* **`atomicAnd(int* address, int val)`**：原子按位与（Bitwise AND）。
* **`atomicOr(int* address, int val)`**  ：原子按位或（Bitwise OR）。
* **`atomicXor(int* address, int val)`**  ：原子按位异或（Bitwise XOR）。
* **`atomicCAS(int* address, int compare, int val)`**：原子比较并交换（Compare-And-Swap）。如果 `address` 的值等于 `compare`，则将其替换为 `val`。无论是否发生替换，该函数均返回 `address` 的原始值。它是实现各种高级锁机制的底层基石。

---

### 浮点数原子操作 (Floating-Point Atomic Operations)

* **`atomicAdd(float* address, float val)`**：单精度浮点数原子加法，自 CUDA 2.0 起支持。
* **双精度支持**：双精度浮点数原子加法 `atomicAdd(double* address, double val)` 从 **Compute Capability 6.0** (Pascal 架构) 开始在硬件层原生支持。

---

### 🛠️ 底层工作机制与软件模拟

现代 GPU 拥有专用的片上硬件逻辑来高效地执行这些原子操作。它们在硬件层面上采用诸如**比较并交换（CAS）**等机制。

你可以把原子操作想象成一种在硬件层面上执行的、速度极快的互斥锁（Mutex）操作。逻辑如下：
1. `lock(memory_location)`（锁定内存位置）
2. `old_value = *memory_location`（读取旧值）
3. `*memory_location = old_value + increment`（写入新值）
4. `unlock(memory_location)`（解锁）
5. `return old_value`（返回旧值）

下面的 C++ 代码展示了如何使用 `atomicCAS` 锁机制在软件层面模拟出一个 `atomicAdd` 操作：

```cpp
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    // 尝试获取锁：如果 lock 的值为 0，则设为 1。
    // 如果返回的值不为 0（说明已被占用），则一直自旋等待。
    while (atomicCAS(&lock, 0, 1) != 0);  
    
    // 临界区：安全读写
    old = *address;
    *address = old + increment;
    
    __threadfence();  // 内存栅栏，确保写入的值对其它线程立刻可见
    
    atomicExch(&lock, 0);  // 释放锁
    
    return old;
}
```

---

### 🔒 互斥锁 (Mutual Exclusion)

* [互斥锁教学视频参考](https://www.youtube.com/watch?v=MqnpIwN7dz0&t)
* **“Mutual” (相互/互)**：表示实体（在本项目中指线程）之间的共享关系，说明该排他性对所有参与的线程都一视同仁。
* **“Exclusion” (排他/斥)**：指防止并发访问的动作。在这里代表在同一时刻阻止多个线程同时访问同一个临界区资源。

下面的完整代码展示了如何利用 `atomicCAS` 实现一个简易自旋锁保护核函数中的临界区计算（对应 [00_atomicAdd.cu](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/CUDA/05_Writing_your_First_Kernels/04%20Atomics/00_atomicAdd.cu) 示例）：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 互斥锁结构体
struct Mutex {
    int *lock;
};

// 初始化锁（在主机端调用）
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// 加锁（在设备端调用）
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // 自旋等待 (Spin-wait)
    }
}

// 解锁（在设备端调用）
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// 核函数：演示锁的使用
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // 临界区 (Critical section)
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动包含 1000 个线程的单块核函数
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}
```
