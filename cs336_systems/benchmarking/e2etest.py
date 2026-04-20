import argparse
from typing import Callable

from numpy import isin
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW



def get_model(vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10000.0):

    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

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
    return AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

def get_run(model: BasicsTransformerLM, data: torch.Tensor, optimizer:AdamW|None = None, pattern:str = "forward") -> Callable:
    if pattern == "forward":
        def run():
            with torch.no_grad():   
                model(data)
    elif pattern == "forward_backward":
        def run():
            optimizer.zero_grad()
            output = model(data)
            loss = output.mean()
            loss.backward()
    elif pattern == "forward_backward_step" and optimizer is not None:
        def run():
            optimizer.zero_grad()
            output = model(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
    else:
        raise ValueError(f"Invalid pattern: {pattern}")
    return run

def benchmark(run:Callable, model, data, num_trials:int = 10, warmup_steps=5):
    # Warmup
    for _ in range(warmup_steps):
        run()

    torch.cuda.synchronize()

    # Time it for real now!
    times: list[float] = []

    for trial in range(num_trials):  # Do it multiple times to capture variance
    # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing
        torch.cuda.synchronize()  # Wait for CUDA threads to finish
        times.append((start_event.elapsed_time(end_event)))  

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    print(f"Mean time: {mean_time:.2f} ms, Std time: {std_time:.2f} ms")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load args")
    parser.add_argument("--ctx", type=int, default=512, help="context length")
    parser.add_argument("--d_model", type=int, default=512, help="model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of feedforward network")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="theta for RoPE positional encoding")
    args = parser.parse_args()
    #model = BasicsTransformerLM(args.vocab_size, args.ctx, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    data = get_data(args.batch_size, args.ctx)
    small = {"ctx": 512, "d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 10000, "rope_theta": 10000.0}
    medium = {"ctx":512, "d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096, "vocab_size": 10000, "rope_theta": 10000.0}
    large = {"ctx": 512, "d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120, "vocab_size": 10000, "rope_theta": 10000.0}
    xl = {"ctx": 512, "d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240, "vocab_size": 10000, "rope_theta": 10000.0}
    _10B = {"ctx": 512, "d_model": 4608, "num_layers": 50, "num_heads": 36, "d_ff": 12288, "vocab_size": 10000, "rope_theta": 10000.0}
    if torch.cuda.is_available():
        print("Using GPU for benchmarking.")
    else:
        raise ValueError("CUDA is not available. Please run this benchmark on a machine with a compatible NVIDIA GPU and CUDA installed.")
    model = get_model(small)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    optimizer = get_optimizer(model)
    run1 = 0
    run2 = 0
    run3 = 0
    print("size of model: ", sum(p.numel() for p in model.parameters()) / 1e6 , "M parameters")
    print("size of data: ", data.numel() * data.element_size() / 1e6, "MB")
    #run1 = get_run(model, data, optimizer=optimizer, pattern="forward")
    #run2 = get_run(model, data, optimizer=optimizer, pattern="forward_backward")
    #run3 = get_run(model, data, optimizer=optimizer, pattern="forward_backward_step")
    if isinstance(run1, Callable):
        print("Benchmarking forward pass...")
        benchmark(run1, model, data, num_trials=10, warmup_steps=5)
    if isinstance(run2, Callable):
       print("Benchmarking forward + backward pass...")
       benchmark(run2, model, data, num_trials=10, warmup_steps=5)
    if isinstance(run3, Callable):
       print("Benchmarking forward + backward + step...")
       benchmark(run3, model, data, num_trials=10, warmup_steps=5)
