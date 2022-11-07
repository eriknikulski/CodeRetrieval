import os
import socket
from contextlib import closing

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp

import const


def find_free_port(addr):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((addr, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = const.MASTER_ADDR
    os.environ['MASTER_PORT'] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(f"cuda:{rank}")
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def run(fn, world_size, train_dataloader, test_dataloader, experiment_name, port):
    mp.spawn(fn, args=(world_size, train_dataloader, test_dataloader, experiment_name, port,),
             nprocs=world_size, join=True)
