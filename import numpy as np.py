import numpy as np
import cupy as cp
import time


# -------------------------
# CPU PIPELINE
# -------------------------
def cpu_pipeline(data):
    start = time.time()

    data = np.sort(data)
    data = data[data > 0.5]

    idx = np.searchsorted(data, 0.7)

    total = np.sum(data)
    avg = np.mean(data)

    end = time.time()

    return end - start, idx, total, avg


# -------------------------
# GPU PIPELINE
# -------------------------
def gpu_pipeline(data):
    data_gpu = cp.asarray(data)

    cp.cuda.Device().synchronize()
    start = time.time()

    data_gpu = cp.sort(data_gpu)
    data_gpu = data_gpu[data_gpu > 0.5]

   
    target = cp.array(0.7)
    idx = cp.searchsorted(data_gpu, target)

    total = cp.sum(data_gpu)
    avg = cp.mean(data_gpu)

    cp.cuda.Device().synchronize()
    end = time.time()

    return end - start, idx, total, avg


# -------------------------
# MAIN DEMO
# -------------------------
if __name__ == "__main__":

    sizes = [100_000, 500_000, 1_000_000]

    for size in sizes:
        print("\n====================")
        print("Dataset size:", size)

        data = np.random.rand(size)

        # CPU run
        cpu_time, cpu_idx, cpu_sum, cpu_avg = cpu_pipeline(data)
        print("CPU time:", cpu_time)

        # GPU run
        gpu_time, gpu_idx, gpu_sum, gpu_avg = gpu_pipeline(data)
        print("GPU time:", gpu_time)

        # Speedup
        if gpu_time > 0:
            print("Speedup:", cpu_time / gpu_time)
        else:
            print("Speedup: N/A")