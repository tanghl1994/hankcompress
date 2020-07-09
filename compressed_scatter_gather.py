import torch
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import math
import time
import torch.distributed as dist

from mpi4py import MPI
import pdb

# decompress = cupy.ElementwiseKernel('float32 unified_sign, float32 scale', 'float32 res', 'res = (unified_sign * 2.0 - 1.0) * scale', 'decompress')
cupy_add = cupy.ElementwiseKernel('float32 arg1, float32 arg2', 'float32 res', 'res = arg1 + arg2', 'cupy_add')

_cupy_add = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_add(const float* x1, const float* x2, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = x1[tid] + x2[tid];
 }
 ''', '_cupy_add')

_cupy_subtract = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_subtract(const float* x1, const float* x2, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = x1[tid] - x2[tid];
 }
 ''', '_cupy_subtract')

_cupy_divide = cupy.RawKernel(r'''
 extern "C" __global__
 void _cupy_divide(const float* x1, const float* x2, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = x1[tid] / x2[tid];
 }
 ''', '_cupy_divide')

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

# cupy_subtract = cupy.ElementwiseKernel('float32 arg1, float32 arg2', 'float32 res', 'res = arg1 - arg2', 'cupy_subtract')
cupy_divide = cupy.ElementwiseKernel('float32 arg1, float32 arg2', 'float32 res', 'res = arg1 / arg2', 'cupy_divide')
# cupy_sqrt = cupy.ElementwiseKernel('float32 arg1', 'float32 res', 'res = sqrt(arg1)', 'cupy_sqrt')


_decompress_kernel = cupy.ElementwiseKernel('float32 unpacked_binary, raw float32 scale_array, int32 chunk_size',
                                            'float32 uncompressed',
                                            'uncompressed = (unpacked_binary * 2.0 - 1.0) * scale_array[i/chunk_size];',
                                            'decompress_kernel')

_decompress_kernel_binary = cupy.RawKernel(r'''
 extern "C" __global__
 void _decompress_kernel_binary(const float* binary_bits, const float* scale, const int chunk_size, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = (binary_bits[tid] * 2 - 1) * scale[tid/chunk_size];
 }
 ''', '_decompress_kernel_binary')

##########

# _decompress_kernel_nonbinary = cupy.ElementwiseKernel('float32 signs_array, raw float32 scale_array, int32 chunk_size', 'float32 uncompressed',
#   'uncompressed = signs_array * scale_array[i/chunk_size];',
#   'decompress_kernel_nonbinary')

_decompress_kernel_nonbinary = cupy.RawKernel(r'''
 extern "C" __global__
 void _decompress_kernel_nonbinary(const float* binary_bits, const float* scale, const int chunk_size, float* y) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     y[tid] = binary_bits[tid] * scale[tid/chunk_size];
 }
 ''', '_decompress_kernel_nonbinary')

##############

_avg_chunks = cupy.ElementwiseKernel('raw float32 x, int32 chunk_size, int32 num_chunks', 'float32 z',
                                     '''
                                     for (int j = 0; j < num_chunks; ++j){
                                       z = z + x[i + j * chunk_size] / num_chunks;
                                       }
                                     ''',
                                     'add_chunks')

_l2norm_kernel = cupy.ReductionKernel(
    'float32 x',  # input params
    'float32 y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

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

def myIgather(rank, size, comm, sendbuf, recbuf, root):
    if rank == root:
        for idx in range(size):
            if idx != rank:
                req = comm.Irecv(recbuf[idx], source=idx)
            else:
                recbuf[rank] = sendbuf
    else:
        req = comm.Isend(sendbuf, dest=root)
    # print('rank {} req is: {}'.format(rank,req))

    return req


def compress_by_chunk(tensor, num_chunks, chunk_size, error):
    signs = cupy.sign(tensor)
    # print(signs.device)
    # print(tensor.device)
    total = cupy.concatenate([tensor, signs])


    # sq_total = cupy.square(total).reshape(2*num_chunks, chunk_size)
    sq_total = cupy.zeros_like(total)
    numBlocks = (total.size + block_size - 1) // block_size
    _cupy_square((numBlocks,), (block_size,), (total, sq_total))
    sq_total = sq_total.reshape(2 * num_chunks, chunk_size)

    # cupy.cuda.get_current_stream().synchronize()
    total_chunk_sq_sum = _reduction_chunk(sq_total, axis=1)
    # cupy.cuda.get_current_stream().synchronize()

    # total_chunk_norm = cupy_sqrt(total_chunk_sq_sum)
    total_chunk_norm = cupy.zeros_like(total_chunk_sq_sum)
    numBlocks = (total_chunk_sq_sum.size + block_size - 1) // block_size
    _cupy_square_root((numBlocks,), (block_size,), (total_chunk_sq_sum, total_chunk_norm))

    cupy.cuda.get_current_stream().synchronize()

    scale_list = cupy_divide(total_chunk_norm[:num_chunks], total_chunk_norm[num_chunks:])
    # scale_list = cupy.zeros(num_chunks, dtype=cupy.float32)
    # numBlocks = (num_chunks + block_size - 1) // block_size
    # _cupy_divide((numBlocks,),(block_size,),(total_chunk_norm[:num_chunks], total_chunk_norm[num_chunks:], scale_list))

    uncompressed = cupy.zeros_like(tensor)
    # _decompress_kernel_nonbinary(signs, scale_list, chunk_size, uncompressed)
    numBlocks = (uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_nonbinary((numBlocks,), (block_size,), (signs, scale_list, chunk_size, uncompressed))
    cupy.cuda.get_current_stream().synchronize()

    # error_cupy = cupy_subtract(tensor, uncompressed)
    error_cupy = cupy.zeros_like(tensor)
    numBlocks = (tensor.size + block_size - 1) // block_size
    _cupy_subtract((numBlocks,), (block_size,), (tensor, uncompressed, error_cupy))

    error_tensor = from_dlpack(error_cupy.toDlpack())
    error.set_(error_tensor)

    #################

    signs_cupy = cupy.sign(tensor)
    # signs_bool = (signs_cupy+1).astype(bool)
    signs_bool = cupy_add(signs_cupy, 1).astype(bool)
    packed_sign = cupy.packbits(signs_bool)
    sign_list_packed = cupy.split(packed_sign, num_chunks)

    return sign_list_packed, scale_list


def com_reduce(buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm):
    # print('rank is :', rank)

    all_start_time = time.time()
    chunk_size = torch.numel(server_error)
    flatten_buffer_m = buffer_m.flatten()

    if torch.numel(flatten_buffer_m) != torch.numel(worker_error):
        empty_tensor = torch.cuda.FloatTensor(torch.numel(worker_error) - torch.numel(flatten_buffer_m)).fill_(0)
        flatten_buffer_m = torch.cat([flatten_buffer_m, empty_tensor])

    compensated_buffer_m = flatten_buffer_m + worker_error

    compensated_buffer_m_cupy = cupy.fromDlpack(to_dlpack(compensated_buffer_m))
    # print('cupy device is', compensated_buffer_m_cupy.device)

    sign_list_packed, scale_list = compress_by_chunk(compensated_buffer_m_cupy, world_size, chunk_size, worker_error)

    # First round of communication
    recvbuf_sign = cupy.zeros([world_size, sign_list_packed[rank].size], dtype=sign_list_packed[rank].dtype)
    if rank == 0:
        print('Internal: the size of allgather is {}'.format(len(sign_list_packed[0] ) ))
    recvbuf_scale = cupy.zeros([world_size, 1], dtype=scale_list[rank].dtype)

    requests = []

    gather_start = time.time()
    for idx in range(world_size):
        req_sign = myIgather(rank, world_size, comm, sign_list_packed[idx], recvbuf_sign, root=idx)
        req_scale = myIgather(rank, world_size, comm, scale_list[idx], recvbuf_scale, root=idx)

        requests.append(req_sign)
        requests.append(req_scale)

    MPI.Request.Waitall(requests)

    gather_end = time.time()

    flattened_sign = recvbuf_sign.flatten()
    unpacked_sign = cupy.unpackbits(flattened_sign).astype(cupy.float32)
    local_uncompressed = cupy.zeros_like(unpacked_sign)

    # _decompress_kernel(unpacked_sign, recvbuf_scale, chunk_size, local_uncompressed)
    numBlocks_ = (local_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign, recvbuf_scale, chunk_size, local_uncompressed))
    # cupy.cuda.get_current_stream().synchronize()

    local_reduced_chunk = cupy.zeros(chunk_size, dtype=cupy.float32)
    _avg_chunks(local_uncompressed, chunk_size, world_size, local_reduced_chunk)

    # add server error
    server_error_cupy = cupy.fromDlpack(to_dlpack(server_error))
    local_reduced_chunk_compansated = cupy.zeros_like(local_reduced_chunk)

    cupy_add(local_reduced_chunk, server_error_cupy, local_reduced_chunk_compansated)

    sign_list_packed_server, scale_list_server = compress_by_chunk(local_reduced_chunk_compansated, 1, chunk_size,
                                                                   server_error)

    # prepare buffer
    recvbuf_sign_server = cupy.zeros([world_size, sign_list_packed[0].size], dtype=sign_list_packed_server[0].dtype)
    recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=scale_list_server[0].dtype)

    allgather_start = time.time()

    req_server_sign = comm.Iallgather(sign_list_packed_server[0], recvbuf_sign_server)
    # print('Internal: the size of allgather is {}'.format(len(sign_list_packed_server[0]) * world_size))
    req_server_scale = comm.Iallgather(scale_list_server[0], recvbuf_scale_server)

    MPI.Request.Waitall([req_server_sign, req_server_scale])

    allgather_end = time.time()

    flattened_sign_server = recvbuf_sign_server.flatten()
    unpacked_sign_server = cupy.unpackbits(flattened_sign_server).astype(cupy.float32)
    server_uncompressed = cupy.zeros_like(unpacked_sign_server)

    # _decompress_kernel(unpacked_sign_server, recvbuf_scale_server, chunk_size, server_uncompressed)
    numBlocks_ = (server_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign_server, recvbuf_scale_server, chunk_size, server_uncompressed))

    aggregated_m_tensor = from_dlpack(server_uncompressed.toDlpack())

    aggregated_m_tensor = aggregated_m_tensor[0:torch.numel(buffer_m)]
    buffer_m.set_(aggregated_m_tensor.type(buffer_m.dtype).view_as(buffer_m))

    all_end_time = time.time()

    return gather_end - gather_start, allgather_end - allgather_start, all_end_time - all_start_time


