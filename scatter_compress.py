import torch
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import math
import time
import torch.distributed as dist

from mpi4py import MPI
import pdb

# CUDA kernels used for compression-related computation
cupy_add = cupy.ElementwiseKernel('float32 arg1, float32 arg2', 'float32 res', 'res = arg1 + arg2', 'cupy_add')
cupy_divide = cupy.ElementwiseKernel('float32 arg1, float32 arg2', 'float32 res', 'res = arg1 / arg2', 'cupy_divide')

_cupy_subtract = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_subtract(const float* x1, const float* x2, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = x1[tid] - x2[tid];
 }
 ''', '_cupy_subtract')

_cupy_square = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_square(const float* x1, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = x1[tid] * x1[tid];
 }
 ''', '_cupy_square')

_cupy_square_root = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_square_root(const float* x1, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = sqrt(x1[tid]);
 }
 ''', '_cupy_square_root')

_decompress_kernel_binary = cupy.RawKernel(r'''
 extern "C" __global__
 void _decompress_kernel_binary(const float* binary_bits, const float* scale, const int chunk_size, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = (binary_bits[tid] * 2 - 1) * scale[tid/chunk_size];
 }
 ''', '_decompress_kernel_binary')


_decompress_kernel_nonbinary = cupy.RawKernel(r'''
 extern "C" __global__
 void _decompress_kernel_nonbinary(const float* binary_bits, const float* scale, const int chunk_size, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = binary_bits[tid] * scale[tid/chunk_size];
 }
 ''', '_decompress_kernel_nonbinary')

_avg_chunks = cupy.ElementwiseKernel('raw float32 x, int32 chunk_size, int32 num_chunks', 'float32 z',
                                     '''
                                     for (int j = 0; j < num_chunks; ++j){
                                       z = z + x[i + j * chunk_size] / num_chunks;
                                       }
                                     ''',
                                     'add_chunks')

_reduction_chunk = cupy.ReductionKernel(
    'float32 x',  # input params
    'float32 y',  # output params
    'x',
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    '_reduction_chunk'  # kernel name
)

block_size = 128


def compress_by_chunk(tensor, num_chunks, chunk_size, last_chunk_size, error):
    
    signs = cupy.sign(tensor)

    if last_chunk_size != chunk_size:
        empty_tensor = cupy.zeros(chunk_size - last_chunk_size, dtype = tensor.dtype)
        tensor_aligned = cupy.concatenate([tensor, empty_tensor])
        signs_aligned = cupy.concatenate([signs, empty_tensor])
    else:
        tensor_aligned = tensor
        signs_aligned = signs

    total = cupy.concatenate([tensor_aligned, signs_aligned])
    # calculate norm in parallel
    # 1.square
    sq_total = cupy.zeros_like(total)
    numBlocks = (total.size + block_size - 1) // block_size
    _cupy_square((numBlocks,), (block_size,), (total, sq_total))
    # 2.add reduction
    sq_total = sq_total.reshape(2 * num_chunks, chunk_size)
    total_chunk_sq_sum = _reduction_chunk(sq_total, axis=1)
    # 3.square root
    total_chunk_norm = cupy.zeros_like(total_chunk_sq_sum)
    numBlocks = (total_chunk_sq_sum.size + block_size - 1) // block_size
    _cupy_square_root((numBlocks,), (block_size,), (total_chunk_sq_sum, total_chunk_norm))

    cupy.cuda.get_current_stream().synchronize()
    # calculate scale
    scale_list = cupy_divide(total_chunk_norm[:num_chunks], total_chunk_norm[num_chunks:])
    # calculate compression error
    uncompressed = cupy.zeros_like(tensor)
    numBlocks = (uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_nonbinary((numBlocks,), (block_size,), (signs, scale_list, chunk_size, uncompressed))

    cupy.cuda.get_current_stream().synchronize()

    error_cupy = cupy.zeros_like(tensor)
    numBlocks = (tensor.size + block_size - 1) // block_size
    _cupy_subtract((numBlocks,), (block_size,), (tensor, uncompressed, error_cupy))
    # update worker error
    error_tensor = from_dlpack(error_cupy.toDlpack())
    error.set_(error_tensor)
    # pack signs into uint8
    signs_bool = cupy_add(signs, 1).astype(bool)
    packed_sign = cupy.packbits(signs_bool)

    return packed_sign, scale_list


def compress_one_chunk(tensor, error):
    
    signs = cupy.sign(tensor)

    # calculate norm in parallel
    total = cupy.concatenate([tensor, signs])

    sq_total = cupy.zeros_like(total)
    numBlocks = (total.size + block_size - 1) // block_size
    _cupy_square((numBlocks,), (block_size,), (total, sq_total))
    sq_total = sq_total.reshape(2, tensor.size)

    total_chunk_sq_sum = _reduction_chunk(sq_total, axis=1)

    total_chunk_norm = cupy.zeros_like(total_chunk_sq_sum)
    numBlocks = (total_chunk_sq_sum.size + block_size - 1) // block_size
    _cupy_square_root((numBlocks,), (block_size,), (total_chunk_sq_sum, total_chunk_norm))

    cupy.cuda.get_current_stream().synchronize()

    scale_list = cupy_divide(total_chunk_norm[0], total_chunk_norm[1])

    uncompressed = cupy.zeros_like(tensor)
    numBlocks = (uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_nonbinary((numBlocks,), (block_size,), (signs, scale_list, tensor.size, uncompressed))
    cupy.cuda.get_current_stream().synchronize()

    error_cupy = cupy.zeros_like(tensor)
    numBlocks = (tensor.size + block_size - 1) // block_size
    _cupy_subtract((numBlocks,), (block_size,), (tensor, uncompressed, error_cupy))

    error_tensor = from_dlpack(error_cupy.toDlpack())
    error.set_(error_tensor)

    signs_bool = cupy_add(signs, 1).astype(bool)
    packed_sign = cupy.packbits(signs_bool)

    return packed_sign, scale_list


def com_reduce(buffer_m: torch.tensor, worker_error, server_error, chunk_buckets, last_chunk_buckets, last_chunk_size, rank, world_size, comm):
    
    tensor_size = torch.numel(buffer_m)
    my_chunk_size = torch.numel(server_error)
    my_chunk_buckets = last_chunk_buckets if rank == world_size - 1 else chunk_buckets

    # add previous worker error before compression
    flatten_buffer_m = buffer_m.flatten()
    compensated_buffer_m = flatten_buffer_m + worker_error
    # compress local tensor by chunk
    compensated_buffer_m_cupy = cupy.fromDlpack(to_dlpack(compensated_buffer_m))
    sign_list_packed, scale_list = compress_by_chunk(compensated_buffer_m_cupy, world_size, chunk_buckets*8, last_chunk_size, worker_error)

    # First round of communication
    recvbuf_sign = cupy.zeros([world_size, my_chunk_buckets], dtype=sign_list_packed.dtype)
    recvbuf_scale = cupy.zeros([world_size, 1], dtype=scale_list.dtype)

    requests = []
    for idx in range(world_size):

        start = idx * chunk_buckets
        length = last_chunk_buckets if idx == world_size - 1 else chunk_buckets

        req_sign = comm.Igather(sign_list_packed[start:start+length], recvbuf_sign, root=idx)
        req_scale = comm.Igather(scale_list[idx], recvbuf_scale, root=idx)

        requests.append(req_sign)
        requests.append(req_scale)

    MPI.Request.Waitall(requests)

    # unpack received signs
    flattened_sign = recvbuf_sign.flatten()
    unpacked_sign = cupy.unpackbits(flattened_sign).astype(cupy.float32)
    # uncompress
    local_uncompressed = cupy.zeros_like(unpacked_sign)
    numBlocks_ = (local_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign, recvbuf_scale, my_chunk_buckets*8, local_uncompressed))
    # average uncompressed chunks
    local_reduced_chunk = cupy.zeros(my_chunk_buckets*8, dtype=cupy.float32)
    _avg_chunks(local_uncompressed, my_chunk_buckets*8, world_size, local_reduced_chunk)

    # the last chunk may have smaller size
    if rank == world_size - 1:
        local_reduced_chunk = local_reduced_chunk[0:last_chunk_size]

    # add server error before second compression
    server_error_cupy = cupy.fromDlpack(to_dlpack(server_error))
    local_reduced_chunk_compansated = cupy.zeros_like(local_reduced_chunk)
    cupy_add(local_reduced_chunk, server_error_cupy, local_reduced_chunk_compansated)
    # compress again
    sign_list_packed_server, scale_list_server = compress_one_chunk(local_reduced_chunk_compansated, server_error)

    # Second round of communication
    recvbuf_sign_server = [cupy.zeros(chunk_buckets, dtype=sign_list_packed_server.dtype)] * (world_size - 1)
    recvbuf_sign_server.append(cupy.zeros(last_chunk_buckets, dtype=sign_list_packed_server.dtype))
    recvbuf_sign_server[rank] = sign_list_packed_server

    recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=scale_list_server.dtype)

    server_requests = []
    for idx in range(world_size):
        if idx != rank:
            req_server_send = comm.Isend(sign_list_packed_server, idx)
            req_server_recv = comm.Irecv(recvbuf_sign_server[idx], idx)

            server_requests.append(req_server_send)
            server_requests.append(req_server_recv)

    req_server_scale = comm.Iallgather(scale_list_server, recvbuf_scale_server)
    server_requests.append(req_server_scale)

    MPI.Request.Waitall(server_requests)

    # unpack signs
    flattened_sign_server = cupy.concatenate(recvbuf_sign_server)
    unpacked_sign_server = cupy.unpackbits(flattened_sign_server).astype(cupy.float32)
    # uncompress
    server_uncompressed = cupy.zeros_like(unpacked_sign_server)
    numBlocks_ = (server_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign_server, recvbuf_scale_server, chunk_buckets*8, server_uncompressed))
    # update the synced tensor
    server_uncompressed = server_uncompressed[0:tensor_size]
    aggregated_m_tensor = from_dlpack(server_uncompressed.toDlpack())
    buffer_m.set_(aggregated_m_tensor.type(buffer_m.dtype).view_as(buffer_m))


