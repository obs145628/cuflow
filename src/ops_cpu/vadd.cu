#include "vadd.hh"

#include <iostream>

namespace cpu
{

    void op_vadd(const float* a, const float* b, float* out, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
                out[i] = a[i] + b[i];
    }

}
