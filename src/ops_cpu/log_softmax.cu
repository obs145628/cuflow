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
    

    void op_log_softmax(const float* x, float* y,
                        std::size_t m, std::size_t n)
    {
        auto start = timer::now();
        

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
                y[i * n + j] = x[i * n + j] - logsum;
        }


        auto time = timer::now() - start;
        logs << "[CPU_LOG_SOFTMAX]: " << time << "ms.\n";
    }

}
