#include "init.hh"
#include <cassert>
#include <string>
#include <vector>
#include "instruction.hh"
#include "runtime.hh"
#include "../ops/vadd.hh"


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

        op_vadd(a->gdata, b->gdata, out->gdata, a->size);
    }

}

void init()
{
    Runtime::add_fun("i", ins_i);
    Runtime::add_fun("o", ins_o);
    Runtime::add_fun("vadd", ins_vadd);
}
