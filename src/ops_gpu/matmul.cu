#include "matmul.hh"

#include <iostream>
#include "../app/mode.hh"
#include "../app/timer.hh"

namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 32;

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

        __global__
        void matmul2(const float* a, const float* b, float* out,
                     std::size_t arows, std::size_t acols, std::size_t bcols)
        {
            std::size_t nb_tiles = (max(arows, acols) + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::size_t kern_i = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t kern_j = blockIdx.y * blockDim.y + threadIdx.y;
            float kern_val = 0;

            for (std::size_t tile_i = 0; tile_i < nb_tiles; ++tile_i)
            {

                __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

                std::size_t a_blocki = blockIdx.x;
                std::size_t a_blockj = tile_i;
                std::size_t b_blocki = tile_i;
                std::size_t b_blockj = blockIdx.y;
                

                std::size_t ai = a_blocki * BLOCK_SIZE + threadIdx.x;
                std::size_t aj = a_blockj * BLOCK_SIZE + threadIdx.y;
                std::size_t bi = b_blocki * BLOCK_SIZE + threadIdx.x;
                std::size_t bj = b_blockj * BLOCK_SIZE + threadIdx.y;

                if (ai < arows && aj < acols)
                    tile_a[threadIdx.x][threadIdx.y] = a[ai * acols + aj];
                else
                    tile_a[threadIdx.x][threadIdx.y] = 0;

                if (bi < acols && bj < bcols)
                    tile_b[threadIdx.x][threadIdx.y] = b[bi * bcols + bj];
                else
                    tile_b[threadIdx.x][threadIdx.y] = 0;

                __syncthreads();

                for (std::size_t k = 0; k < BLOCK_SIZE; ++k)
                    kern_val += tile_a[threadIdx.x][k] * tile_b[k][threadIdx.y];

                __syncthreads();
            }

            if (kern_i < arows && kern_j < bcols)
                out[kern_i * bcols + kern_j] = kern_val;
        }
        
    }

    void op_matmul(const float* a, const float* b, float* out,
                   std::size_t arows, std::size_t acols, std::size_t bcols)
    {

        /*
        std::size_t block_size = 32;
        dim3 threads_per_block (block_size, block_size);

        std::size_t nb_blocks_x = (arows + block_size - 1) / block_size;
        std::size_t nb_blocks_y = (bcols + block_size - 1) / block_size;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);
        */

        dim3 threads_per_block (BLOCK_SIZE, BLOCK_SIZE);
        std::size_t nb_blocks_x = (arows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::size_t nb_blocks_y = (bcols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);
        
        


        auto start = timer::now();
        matmul2<<<blocks_per_grid, threads_per_block>>>(a, b, out, arows, acols, bcols);
        cudaDeviceSynchronize();
        auto time = timer::now() - start;

        logs << "[GPU_MATMUL]: " << time << "ms.\n";
    }

}
