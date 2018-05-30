#include "matmul.hh"

#include <iostream>
#include "../app/mode.hh"
#include "../app/timer.hh"

namespace gpu
{

    namespace
    {

        __global__
        void matmul(const float* a, const float* b, float* out,
                    std::size_t arows, std::size_t acols, std::size_t bcols)
        {
            std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t col = blockIdx.y * blockDim.y + threadIdx.y;

            if (row >= arows || col >= bcols)
                return;

            float x = 0;
            for (std::size_t i = 0; i < acols; ++i)
                x += a[row * acols + i] * b[i * bcols + col];
            out[row * bcols + col] = x;
        }
        
    }

    void op_matmul(const float* a, const float* b, float* out,
                   std::size_t arows, std::size_t acols, std::size_t bcols)
    {

        std::size_t block_size = 32;
        dim3 threads_per_block (block_size, block_size);

        std::size_t nb_blocks_x = (arows + block_size - 1) / block_size;
        std::size_t nb_blocks_y = (bcols + block_size - 1) / block_size;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);


        auto start = timer::now();
        matmul<<<blocks_per_grid, threads_per_block>>>(a, b, out, arows, acols, bcols);
        cudaDeviceSynchronize();
        auto time = timer::now() - start;

        logs << "[GPU_MATMUL]: " << time << "ms.\n";
    }

}
