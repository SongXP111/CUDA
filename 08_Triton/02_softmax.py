import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(axis=0)

    # Compute the memory offsets for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Load the row into SRAM
    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max from row and exponentiate
    numerator = tl.exp(row - row_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store the output
    tl.store(out_row_start_ptr + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  
    
    # Launch the Triton kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# Set up the input tensor
torch.manual_seed(0)
x = torch.randn(256, 1024, device='cuda')

# Compute softmax using PyTorch
torch_result = torch.softmax(x, dim=1)

# Compute softmax using Triton
triton_result = triton_softmax(x)

# Compare results using torch.testing.assert_close
try:
    torch.testing.assert_close(torch_result, triton_result, rtol=1e-5, atol=1e-5)
    print("Correctness check passed: PyTorch and Triton results are close!")
except AssertionError as e:
    print(f"Correctness check failed:\n{e}")

# Set up benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_cols'],  # Argument names to use as an x-axis for the plot.
        x_vals=[128 * i for i in range(1, 9)],  # Columns: 128, 256, 384, 512, 640, 768, 896, 1024
        x_log=False,  # x axis is linear.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='softmax-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={'num_rows': 4096},  # Fixed number of rows.
    ))
def benchmark(num_rows, num_cols, provider):
    x = torch.randn(num_rows, num_cols, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
    
    # Calculate throughput (GB/s): 1 Read + 1 Write = 2 memory accesses
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Run benchmark
benchmark.run(print_data=True, show_plots=False, save_path='.')