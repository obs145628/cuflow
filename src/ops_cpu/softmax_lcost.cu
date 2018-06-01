#include "log_softmax.hh"
#include "../app/mode.hh"
#include "../app/timer.hh"
#include <cmath>

namespace cpu
{

    namespace
    {

        float max(const float* begin, const float* end)
        {
            float res = *begin;
            for (const float* it = begin; it != end; ++it)
                res = std::max(res, *it);
            return res;
        }

    }
    

    void op_softmax_lcost(const float* y, const float* x, float* out,
                          std::size_t m, std::size_t n)
    {
        auto start = timer::now();


        float res = 0;
        for (std::size_t i = 0; i < m; ++i)
        {

            //max_x = max(x[i])
            float max_x = max(x + i * n, x + (i + 1) * n);


            //e_x = sum(exp(x[i] - max_x))
            float e_x = 0;
            for (std::size_t j = 0; j < n; ++j)
                e_x += std::exp(x[i * n + j] - max_x);

            float logsum = max_x + std::log(e_x);
            
            //y[i] = x[i] - logsum
            for (std::size_t j = 0; j < n; ++j)
                res += y[i * n + j] * (x[i * n + j] - logsum);
        }
        *out = - res / m;


        auto time = timer::now() - start;
        logs << "[CPU_SOFTMAX_LCOST]: " << time << "ms.\n";
    }

}
