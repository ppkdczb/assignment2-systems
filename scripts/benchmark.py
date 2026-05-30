import argparse
from typing import Callable

from numpy import isin, sort
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import time

def get_model(args:dict) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=args["vocab_size"],
        context_length=args["ctx"],
        d_model=args["d_model"],
        num_layers=args["num_layers"],
        num_heads=args["num_heads"],
        d_ff=args["d_ff"],
        rope_theta=args["rope_theta"]
    )

def get_data(batch_size:int, seq_len:int, vocab_size:int = 10000) -> torch.Tensor:
    return torch.randint(vocab_size, (batch_size, seq_len)) # [b, s]

def get_optimizer(model: BasicsTransformerLM):
    #return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    return AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

def get_run(model: BasicsTransformerLM, data: torch.Tensor, optimizer:AdamW|None = None, pattern:str = "forward") -> Callable:
    if pattern == "forward":
        def run():  
            with torch.cuda.nvtx.range("forward"):
                output = model(data)
                loss = output.mean()
    elif pattern == "forward_backward":
        def run():
            with torch.cuda.nvtx.range("forward"):
                output = model(data)
                loss = output.mean()
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
    elif pattern == "forward_backward_step" and optimizer is not None:
        def run():
            with torch.cuda.nvtx.range("forward"):
                output = model(data)
                loss = output.mean()
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
            with torch.cuda.nvtx.range("step"):
                optimizer.step()
    return run

def benchmark(run:Callable, model, data, num_trials:int = 10, warmup_steps=5):
    # Warmup
    for _ in range(warmup_steps):
        run()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Time it for real now!
    times: list[float] = []

    for trial in range(num_trials):  # Do it multiple times to capture variance
    # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = time.time()
        run()  # Actually perform computation
        end_event = time.time()
        torch.cuda.synchronize()  # Wait for CUDA threads to finish
        peak_time = torch.cuda.max_memory_allocated() / 1e6  # Peak memory in MB
        times.append((end_event - start_event) * 1000)  # Convert to milliseconds

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    print(f"Mean time: {mean_time:.2f} ms, Std time: {std_time:.2f} ms")
    print(f"Peak GPU memory usage: {peak_time:.2f} MB")
    return mean_time, std_time, peak_time
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load args")
    parser.add_argument("--mode", type=str, default="fbs", help="f, fb, or fbs for forward, forward+backward, or forward+backward+step")
    parser.add_argument("--model_size", type=str, default="small", help="small, medium, large, xl, or 10B for predefined model sizes")
    parser.add_argument("--ctx", type=int, default=512, help="context length")
    args = parser.parse_args()
    batch_sz = 1
    ctx = args.ctx
    small = {"batch_size": batch_sz, "ctx": ctx, "d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 10000, "rope_theta": 10000.0}
    medium = {"batch_size": batch_sz, "ctx": ctx, "d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096, "vocab_size": 10000, "rope_theta": 10000.0}
    large = {"batch_size": batch_sz, "ctx": ctx, "d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120, "vocab_size": 10000, "rope_theta": 10000.0}
    xl = {"batch_size": batch_sz, "ctx": ctx, "d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240, "vocab_size": 10000, "rope_theta": 10000.0}
    _10B = {"batch_size": batch_sz, "ctx": ctx, "d_model": 4608, "num_layers": 50, "num_heads": 36, "d_ff": 12288, "vocab_size": 10000, "rope_theta": 10000.0}
    if torch.cuda.is_available():
        print("Using GPU for benchmarking.")
    else:
        raise ValueError("CUDA is not available. Please run this benchmark on a machine with a compatible NVIDIA GPU and CUDA installed.")
    # result = {} 

    # for mode in ["f", "fb", "fbs"]:
    #     print(f"Benchmarking mode: {mode}")
    #     if mode == "f":
    #         pattern = "forward"
    #     elif mode == "fb":
    #         pattern = "forward_backward"
    #     elif mode == "fbs": 
    #         pattern = "forward_backward_step"
    #     for model_size in ["small", "medium", "large"]:
    #         print(f"Benchmarking model size: {model_size}")
    #         if model_size == "small":
    #             config = small
    #         elif model_size == "medium":
    #             config = medium
    #         elif model_size == "large":
    #             config = large
    #         elif model_size == "xl":
    #             config = xl
    #         elif model_size == "10B":
    #             config = _10B
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         data = get_data(config["batch_size"], config["ctx"], config["vocab_size"])
    #         model = get_model(config)
    #         model.to(device)
    #         data = data.to(device)
    #         optimizer = get_optimizer(model)
    #         run_fn = get_run(model, data, optimizer=optimizer, pattern=pattern)
    #         mean_time, std_time, peak_time = benchmark(run_fn, model, data, num_trials=10, warmup_steps=5)
    #         result[(model_size, mode)] = (mean_time, std_time, peak_time)
    # def print_results(result):
    #       # 最后打印表格, mode size 依次打印
    #     sorted_keys = sorted(result.keys(), key=lambda x: (x[0], x[1]))  # Sort by model size, then mode                                                                                                                                       
    #     sorted_result = {key: result[key] for key in sorted_keys}
    #     print("| Model | Mode | Mean (ms) | Std (ms) | Memory (MB) |")                                                                                          
    #     print("|-------|------|-----------|----------|-------------|") 
    #     for (model_size, mode), (mean_time, std_time, peak_time) in sorted_result.items():
    #         print(f"| {model_size} | {mode} | {mean_time:.2f} | {std_time:.2f} | {peak_time:.2f} |")
    # print_results(result)
    if args.model_size == "small":
        config = small
    elif args.model_size == "medium":
        config = medium
    elif args.model_size == "large":
        config = large
    elif args.model_size == "xl":
        config = xl
    elif args.model_size == "10B":
        config = _10B
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data(config["batch_size"], config["ctx"], config["vocab_size"])
    model = get_model(config)
    model.to(device)
    data = data.to(device)
    optimizer = get_optimizer(model)
    run1 = None
    run2 = None
    run3 = None
    print("size of model: ", sum(p.numel() for p in model.parameters()) / 1e6 , "M parameters")
    print("size of data: ", data.numel() * data.element_size() / 1e6, "MB")
    if args.mode == "f":
        run1 = get_run(model, data, pattern="forward")
    elif args.mode == "fb":
        run2 = get_run(model, data, optimizer=optimizer, pattern="forward_backward")
    elif args.mode == "fbs":
        run3 = get_run(model, data, optimizer=optimizer, pattern="forward_backward_step")
    if isinstance(run1, Callable):
        print("Benchmarking forward pass...")
        benchmark(run1, model, data, num_trials=10, warmup_steps=5)
    if isinstance(run2, Callable):
       print("Benchmarking forward + backward pass...")
       benchmark(run2, model, data, num_trials=10, warmup_steps=5)
    if isinstance(run3, Callable):
       print("Benchmarking forward + backward + step...")
       benchmark(run3, model, data, num_trials=10, warmup_steps=5)
