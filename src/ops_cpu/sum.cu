#include "vadd.hh"
#include "../app/mode.hh"
#include "../app/timer.hh"

#include <iostream>

namespace cpu
{

    void op_sum(const float* a, float* out, std::size_t len)
    {
        auto start = timer::now();
        

        float res = 0;
        for (std::size_t i = 0; i < len; ++i)
                res += a[i];
        *out = res;


        auto time = timer::now() - start;
        logs << "[CPU_SUM]: " << time << "ms.\n";
    }

}
