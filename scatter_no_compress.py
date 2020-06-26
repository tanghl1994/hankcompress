
import torch
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import math
import time
import torch.distributed as dist

from mpi4py import MPI


_avg_chunks = cupy.ElementwiseKernel('raw float32 x, int32 chunk_size, int32 num_chunks', 'float32 z',
  '''
  for (int j = 0; j < num_chunks; ++j){
    z = z + x[i + j * chunk_size] / num_chunks;
    }
  ''',
  'add_chunks')


def com_reduce(buffer_m: torch.tensor, rank, world_size, comm):

  tensor_size = torch.numel(buffer_m)
  chunk_size = (tensor_size + world_size - 1) // world_size
  last_chunk_size = tensor_size - chunk_size * (world_size - 1)
  my_chunk_size = last_chunk_size if rank == world_size - 1 else chunk_size

  flatten_buffer_m = buffer_m.flatten()
  flatten_buffer_m_cupy = cupy.fromDlpack(to_dlpack(flatten_buffer_m))

  # First round of communication
  recvbuf = cupy.zeros([world_size, my_chunk_size], dtype = flatten_buffer_m_cupy.dtype)

  requests = []
  for idx in range(world_size):
    start = idx * chunk_size
    length = last_chunk_size if idx == world_size - 1 else chunk_size

    req_sign = comm.Igather(flatten_buffer_m_cupy[start:start+length], recvbuf, root=idx)

    requests.append(req_sign)

  MPI.Request.Waitall(requests)

  # Second round of communication
  recvbuf_flatten = recvbuf.flatten()
  local_reduced_chunk = cupy.zeros(my_chunk_size, dtype=flatten_buffer_m_cupy.dtype)
  _avg_chunks(recvbuf_flatten, my_chunk_size, world_size, local_reduced_chunk)

  recvbuf_server = [cupy.zeros(chunk_size, dtype=flatten_buffer_m_cupy.dtype)] * (world_size - 1)
  recvbuf_server.append(cupy.zeros(last_chunk_size, dtype=flatten_buffer_m_cupy.dtype))
  recvbuf_server[rank] = local_reduced_chunk

  server_requests = []
  for idx in range(world_size):
    if idx != rank:
      req_server_send = comm.Isend(local_reduced_chunk, idx)
      req_server_recv = comm.Irecv(recvbuf_server[idx], idx)

      server_requests.append(req_server_send)
      server_requests.append(req_server_recv)

  MPI.Request.Waitall(server_requests)

  recvbuf_server_flatten = cupy.concatenate(recvbuf_server)
  aggregated_m_tensor = from_dlpack(recvbuf_server_flatten.toDlpack())

  buffer_m.set_(aggregated_m_tensor.type(buffer_m.dtype).view_as(buffer_m))




