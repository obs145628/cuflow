#pragma once

#include <cstddef>

namespace gpu
{

    void op_vadd(const float* a, const float* b, float* out, std::size_t len);

}
