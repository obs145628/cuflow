#include "vadd.hh"

#include <iostream>
#include "../app/mode.hh"
#include "../app/timer.hh"

namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 512;

        __global__
        void sum0(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            if (i >= len)
                return;
                
            partial[i] = x[i];
            __syncthreads();

            for (std::size_t s = 1; s < BLOCK_SIZE && i + s < len; s *= 2)
            {
                if (i % (2 * s) == 0)
                    partial[i] += partial[i + s];
                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void sum1(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            if (i >= len)
                return;
                
            partial[i] = x[i];
            __syncthreads();

            for (std::size_t s = 1; s < BLOCK_SIZE; s *= 2)
            {
                std::size_t index = 2 * s * i;
                
                if (index + s < len)
                    partial[index] += partial[index + s];
                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void sum2(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            if (i >= len)
                return;
                
            partial[i] = x[i];
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1)
            {
                if (i + s < len)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        
        __global__
        void sum3(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            if (i >= len)
                return;
                
            partial[i] = x[i] + x[i + BLOCK_SIZE];
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1)
            {
                if (i + s < len)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void sum4(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            //partial[i] = x[i] + x[i + BLOCK_SIZE];
            partial[i] = i < len ? x[i] : 0;
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
                y[i] = partial[0];
        }

    

        template <std::size_t BlockSize>
        __global__
        void sum5(const float* x, float* y, std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;

            //partial[i] = x[i] + x[i + BLOCK_SIZE];
            partial[i] = i < len ? x[i] : 0;
            __syncthreads();

            if (BlockSize >= 512)
            {
                if (i < 256)
                    partial[i] += partial[i + 256];
                __syncthreads();
            }

            if (BlockSize >= 256)
            {
                if (i < 128)
                    partial[i] += partial[i + 128];
                __syncthreads();
            }

            if (BlockSize >= 128)
            {
                if (i < 64)
                    partial[i] += partial[i + 64];
                __syncthreads();
            }


            //if not volatile, must use __synctthreads again, why ?
            volatile float* vpartial = partial;
            if (i < 32)
            {
                if (BlockSize >= 64)
                    vpartial[i] += vpartial[i + 32];
                if (BlockSize >= 32)
                    vpartial[i] += vpartial[i + 16];
                if (BlockSize >= 16)
                    vpartial[i] += vpartial[i + 8];
                if (BlockSize >= 8)
                    vpartial[i] += vpartial[i + 4];
                if (BlockSize >= 4)
                    vpartial[i] += vpartial[i + 2];
                if (BlockSize >= 2)
                    vpartial[i] += vpartial[i + 1];
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void full_sum0(const float* x, float* y, std::size_t len) //8ms
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
                

            for (std::size_t s = 1; s < BLOCK_SIZE && i + s < len; s *= 2)
            {
                if (i % (2 * s) == 0)
                    partial[i] += partial[i + s];
                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void full_sum1(const float* x, float* y, std::size_t len) //11ms
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

            for (std::size_t s = 1; s < BLOCK_SIZE; s *= 2)
            {
                std::size_t index = 2 * s * i;
                
                if (index + s < len)
                    partial[index] += partial[index + s];
                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void full_sum2(const float* x, float* y, std::size_t len) //11ms
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

            for (std::size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1)
            {
                if (i + s < len)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            if (i == 0)
                y[i] = partial[0];
        }

        __global__
        void full_sum4(const float* x, float* y, std::size_t len) //8ms
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
                y[i] = partial[0];
        }



        template <std::size_t BlockSize>
        __global__
        void full_sum5(const float* x, float* y, std::size_t len) //
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

            if (BlockSize >= 512)
            {
                if (i < 256)
                    partial[i] += partial[i + 256];
                __syncthreads();
            }

            if (BlockSize >= 256)
            {
                if (i < 128)
                    partial[i] += partial[i + 128];
                __syncthreads();
            }

            if (BlockSize >= 128)
            {
                if (i < 64)
                    partial[i] += partial[i + 64];
                __syncthreads();
            }


            //if not volatile, must use __synctthreads again, why ?
            volatile float* vpartial = partial;
            if (i < 32)
            {
                if (BlockSize >= 64)
                    vpartial[i] += vpartial[i + 32];
                if (BlockSize >= 32)
                    vpartial[i] += vpartial[i + 16];
                if (BlockSize >= 16)
                    vpartial[i] += vpartial[i + 8];
                if (BlockSize >= 8)
                    vpartial[i] += vpartial[i + 4];
                if (BlockSize >= 4)
                    vpartial[i] += vpartial[i + 2];
                if (BlockSize >= 2)
                    vpartial[i] += vpartial[i + 1];
            }

            if (i == 0)
                y[i] = partial[0];
        }

    }
     

    void op_sum(const float* a, float* out, std::size_t len)
    {
        auto start = timer::now();
        full_sum4<<<1, BLOCK_SIZE>>>(a, out, len);
//full_sum5<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(a, out, len);
        cudaDeviceSynchronize();
        auto time = timer::now() - start;

        logs << "[GPU_SUM]: " << time << "ms.\n";
    }

}
