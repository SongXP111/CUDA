import re

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 替换目录表格
old_table = """| **第五章** | `05_Writing_your_First_Kernels/` | CUDA 基础、核函数编写、Profiling 分析、原子操作、Streams |
| **第六章** | `06_CUDA_APIs/` | cuBLAS、cuDNN、cuBLASmp 等官方加速库 |
| **第七章** | `07_Faster_Matmul/` | 矩阵乘法（SGEMM）性能分阶优化与极致调优 |"""

new_table = """| **第五章** | `05_Writing_your_First_Kernels/` | CUDA 基础、核函数编写、Profiling 分析、原子操作、Streams |
| **第六章** | `06_CUDA_APIs/` | cuBLAS、cuDNN、cuBLASmp 等官方加速库 |
| **第七章** | `07_Faster_Matmul/` | 矩阵乘法（SGEMM）性能分阶优化与极致调优 |
| **第八章** | `08_Triton/` | Triton 编程模型、Block-level 算子开发与 GPU 性能对比 |
| **第九章** | `09_PyTorch_Extensions/` | PyTorch C++ / CUDA 扩展开发与算子融合性能优化 |"""

content = content.replace(old_table, new_table)

# 2. 替换第八章以后的所有内容
old_qa_pattern = r'### 📙 第八章：Triton 编程与优化.*'

new_qa = """### 📙 第八章：Triton 编程与优化

> 完整 Q&A 文档：[08_Triton/QA.md](./08_Triton/QA.md)

* 🐍 **Triton 编程模型与底层原理**
  * [Q1: CUDA 与 Triton 的编程模型有什么本质区别？什么是“Block-level”编程？](./08_Triton/QA.md#q1-cuda-与-triton-的编程模型有什么本质区别什么是block-level编程)

---

### 📙 第九章：PyTorch C++ / CUDA 扩展开发

> 完整 Q&A 文档：[09_PyTorch_Extensions/QA.md](./09_PyTorch_Extensions/QA.md)

* 🔌 **自定义扩展开发与绑定机制**
  * [Q1: 为什么自定义 CUDA 扩展（CUDA Extension）比 PyTorch 原生组合（如 `x^2 + x + 1`）快这么多？](./09_PyTorch_Extensions/QA.md#q1-为什么自定义-cuda-扩展cuda-extension比-pytorch-原生组合如-x2--x--1快这么多)
  * [Q2: PyTorch 提供了哪几种编写自定义 CUDA 扩展的方式？它们各有什么优缺点？](./09_PyTorch_Extensions/QA.md#q2-pytorch-提供了哪几种编写自定义-cuda-扩展的方式它们各有什么优缺点)
  * [Q3: 什么是 `AT_DISPATCH_FLOATING_TYPES` 宏？它的作用和底层机制是什么？](./09_PyTorch_Extensions/QA.md#q3-什么是-at_dispatch_floating_types-宏它的作用和底层机制是什么)
  * [Q4: 在 Windows 平台编译 PyTorch CUDA 扩展，最常见的编译报错有哪些？如何完美解决？](./09_PyTorch_Extensions/QA.md#q4-在-windows-平台编译-pytorch-cuda-扩展最常见的编译报错有哪些如何完美解决)
"""

content = re.sub(old_qa_pattern, new_qa, content, flags=re.DOTALL)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Root README.md updated successfully!")
