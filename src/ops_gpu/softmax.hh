#pragma once

#include <cstddef>

namespace gpu
{

    void op_softmax(const float* x, float* y,
                    std::size_t m, std::size_t n);

}
