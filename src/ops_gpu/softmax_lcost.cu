#include "matmul.hh"
#include "../app/mode.hh"
#include "../app/timer.hh"
#include <cmath>
#include <math_functions.h>

namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 512;

        __global__
        void softmax1(const float* y, const float* x, float* out,
                      std::size_t rows, std::size_t cols)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto row = blockIdx.x;
            auto col = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            float init = 1e-30;
            for (std::size_t i = col; i < cols; i += step)
                init = max(x[row * cols + i], init);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] = max(partial[col], partial[col + s]);

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile float* vpartial = partial;
            if (col < 32)
            {
                vpartial[col] = max(vpartial[col], vpartial[col + 32]);
                vpartial[col] = max(vpartial[col], vpartial[col + 16]);
                vpartial[col] = max(vpartial[col], vpartial[col + 8]);
                vpartial[col] = max(vpartial[col], vpartial[col + 4]);
                vpartial[col] = max(vpartial[col],vpartial[col + 2]);
                vpartial[col] = max(vpartial[col], vpartial[col + 1]);
            }
            
            __syncthreads();

            float max_x = partial[0];

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += exp(x[row * cols + i] - max_x);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }


            __syncthreads();

            float logsum = max_x + std::log(partial[0]);

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += y[row * cols + i] * (x[row * cols + i] - logsum);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }
            
            if (col == 0)
                out[row] = partial[0];
        }

        __global__
        void reduce_sum(const float* x, float* y, std::size_t len) //8ms
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            float init = 0;
            for (std::size_t j = i; j < len; j += step)
                init += x[j];
        
            partial[i] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (i < s)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile float* vpartial = partial;
            if (i < 32)
            {
                vpartial[i] += vpartial[i + 32];
                vpartial[i] += vpartial[i + 16];
                vpartial[i] += vpartial[i + 8];
                vpartial[i] += vpartial[i + 4];
                vpartial[i] += vpartial[i + 2];
                vpartial[i] += vpartial[i + 1];
            }


            if (i == 0)
                y[0] = - partial[0] / len;
        }
    }
    

    void op_softmax_lcost(const float* y, const float* x, float* out,
                          std::size_t m, std::size_t n)
    {
        

        auto start = timer::now();



        float* tmp;
        cudaMalloc(&tmp, m * sizeof(float));

        softmax1<<<m, BLOCK_SIZE>>>(y, x, tmp, m, n);
        cudaDeviceSynchronize();
        reduce_sum<<<1, BLOCK_SIZE>>>(tmp, out, m);
        cudaDeviceSynchronize();
        
        cudaFree(tmp);

        

        auto time = timer::now() - start;
        logs << "[GPU_SOFTMAX_LCOST]: " << time << "ms.\n";
    }

}
