#include "init.hh"
#include <cassert>
#include <string>
#include <vector>
#include "instruction.hh"
#include "runtime.hh"
#include "mode.hh"

#include "../ops_cpu/matmul.hh"
#include "../ops_gpu/matmul.hh"
#include "../ops_cpu/softmax.hh"
#include "../ops_gpu/softmax.hh"
#include "../ops_cpu/sum.hh"
#include "../ops_gpu/sum.hh"
#include "../ops_cpu/vadd.hh"
#include "../ops_gpu/vadd.hh"


namespace
{

    std::vector<std::size_t> to_shape(const std::string* begin,
                                      const std::string* end)
    {
        std::vector<std::size_t> res;
        for (auto it = begin; it != end; ++it)
            res.push_back(std::atoi(it->c_str()));
        return res;
    }

    void ins_i(Runtime& rt, const Instruction& ins)
    {
        std::string name = ins.args[1];
        auto shape = to_shape(&ins.args[2], &*ins.args.end());
        rt.add_input(name, shape);
    }

    void ins_o(Runtime& rt, const Instruction& ins)
    {
        std::string name = ins.args[1];
        auto shape = to_shape(&ins.args[2], &*ins.args.end());
        rt.add_output(name, shape);
    }

    void ins_vadd(Runtime& rt, const Instruction& ins)
    {
        auto a = rt.get_var(ins.args[1]);
        auto b = rt.get_var(ins.args[2]);
        auto out = rt.get_var(ins.args[3]);
        assert(a);
        assert(b);
        assert(out);

        if (arch_mode == MODE_CPU)
            cpu::op_vadd(a->gdata, b->gdata, out->gdata, a->size);
        else
            gpu::op_vadd(a->gdata, b->gdata, out->gdata, a->size);
    }

    void ins_matmul(Runtime& rt, const Instruction& ins)
    {
        auto a = rt.get_var(ins.args[1]);
        auto b = rt.get_var(ins.args[2]);
        auto out = rt.get_var(ins.args[3]);
        assert(a);
        assert(b);
        assert(out);

        if (arch_mode == MODE_CPU)
            cpu::op_matmul(a->gdata, b->gdata, out->gdata,
                           a->shape[0], a->shape[1], b->shape[1]);
        else
            gpu::op_matmul(a->gdata, b->gdata, out->gdata,
                           a->shape[0], a->shape[1], b->shape[1]);
    }

    void ins_softmax(Runtime& rt, const Instruction& ins)
    {
        auto a = rt.get_var(ins.args[1]);
        auto out = rt.get_var(ins.args[2]);
        assert(a);
        assert(out);

        if (arch_mode == MODE_CPU)
            cpu::op_softmax(a->gdata, out->gdata, a->shape[0], a->shape[1]);
        else
            gpu::op_softmax(a->gdata, out->gdata, a->shape[0], a->shape[1]);
    }

    void ins_sum(Runtime& rt, const Instruction& ins)
    {
        auto a = rt.get_var(ins.args[1]);
        auto out = rt.get_var(ins.args[2]);
        assert(a);
        assert(out);

        if (arch_mode == MODE_CPU)
            cpu::op_sum(a->gdata, out->gdata, a->size);
        else
            gpu::op_sum(a->gdata, out->gdata, a->size);
    }

}

void init()
{
    Runtime::add_fun("i", ins_i);
    Runtime::add_fun("o", ins_o);
    Runtime::add_fun("matmul", ins_matmul);
    Runtime::add_fun("softmax", ins_softmax);
    Runtime::add_fun("sum", ins_sum);
    Runtime::add_fun("vadd", ins_vadd);
}
