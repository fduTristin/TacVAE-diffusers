"""CPU BF16/DDP smoke test; run with CUDA_VISIBLE_DEVICES='' ."""
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train_autoencoderkl import TrainingAutoencoderKL


def worker(rank, port):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)
    torch.manual_seed(42)
    model = TrainingAutoencoderKL(block_out_channels=(8,), norm_num_groups=4,
        down_block_types=('DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',), sample_size=16)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-6)
    before = model.quant_conv.weight.detach().clone()
    torch.manual_seed(rank + 10)
    for update in range(2):
        for micro in range(2):
            from contextlib import nullcontext
            with ddp.no_sync() if micro == 0 else nullcontext():
                x = torch.rand(2, 3, 16, 16) * 2 - 1
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    y, mean, logvar, _ = ddp(x, training_outputs=True)
                loss = (y.float()-x).square().mean() + 1.0416666667e-5 * (0.5*(mean.square()+logvar.exp()-1-logvar)).mean()
                (loss / 2).backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1)
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(loss)
    assert model.quant_conv.weight.dtype == torch.float32
    assert not torch.equal(before, model.quant_conv.weight)
    assert all(v.dtype == torch.float32 for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    other = flat.clone()
    dist.broadcast(other, 0)
    assert torch.equal(flat, other)
    print('PASS rank', rank, 'FP32 Adam, nonzero updates, accumulated DDP agreement', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    mp.start_processes(worker, args=(port,), nprocs=2, start_method='fork')
