import cupy
from mpi4py import MPI
import torch

MPI_size = MPI.COMM_WORLD.Get_size()
MPI_rank = MPI.COMM_WORLD.Get_rank()


torch.distributed.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:23456", world_size=MPI_size, rank=MPI_rank)
torch.cuda.set_device(MPI_rank)

with cupy.cuda.Device(MPI_rank):
    a = cupy.empty(10)
    print(a.device)
