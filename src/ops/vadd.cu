#include "vadd.hh"

#include <iostream>

namespace
{

    __global__
    void kernel_vadd(const float* a, const float* b, float* out, std::size_t len)
    {
        std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t stride = blockDim.x * gridDim.x;
    
        for (std::size_t i = index; i < len; i += stride)
            out[i] = a[i] + b[i];
    }
    
}

void op_vadd(const float* a, const float* b, float* out, std::size_t len)
{
    std::size_t block_size = 256;
    std::size_t nb_blocks = (len + block_size - 1) / block_size;
    
    kernel_vadd<<<nb_blocks, block_size>>>(a, b, out, len);
    cudaDeviceSynchronize();
}
