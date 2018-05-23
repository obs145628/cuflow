#pragma once

#include <algorithm>
#include "memory.hh"
#include "mode.hh"

template <class T>
T* vect_alloc(std::size_t size)
{
    if (arch_mode == MODE_CPU)
        return new T[size];
    else
    {
        T* res;
        cudaMalloc(&res, size * sizeof(T));
        return res;
    }
}

template <class T>
void vect_free(T* data)
{
    if (arch_mode == MODE_CPU)
        delete[] data;
    else
        cudaFree(data);
}

template <class T>
void vect_write(const T* ibegin, const T* iend, T* obegin)
{
    if (arch_mode == MODE_CPU)
        std::copy(ibegin, iend, obegin);
    else
        cudaMemcpy(obegin, ibegin, (iend - ibegin) *sizeof(T), cudaMemcpyHostToDevice);
}


template <class T>
void vect_read(const T* ibegin, const T* iend, T* obegin)
{
    if (arch_mode == MODE_CPU)
        std::copy(ibegin, iend, obegin);
    else
        cudaMemcpy(obegin, ibegin, (iend - ibegin) *sizeof(T), cudaMemcpyDeviceToHost);
}
