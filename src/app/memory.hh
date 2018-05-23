#pragma once

#include <cstddef>


template <class T>
T* vect_alloc(std::size_t size);

template <class T>
void vect_free(T* data);

template <class T>
void vect_write(const T* ibegin, const T* iend, T* obegin);


template <class T>
void vect_read(const T* ibegin, const T* iend, T* obegin);


#include "memory.hxx"
