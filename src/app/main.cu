#include <cstdlib>
#include <cstring>
#include <iostream>
#include "init.hh"
#include "mode.hh"
#include "runtime.hh"



int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: ./cuflow <cmd-file> <in-file> <out-file>.\n";
        return 1;
    }

    char* smode = std::getenv("ARCH_MODE");
    if (!smode || !strcmp(smode, "CPU"))
        arch_mode = MODE_CPU;
    else if (!strcmp(smode, "GPU"))
        arch_mode = MODE_GPU;
    else
    {
        std::cerr << "Unknowd mode: " << smode << std::endl;
        return 1;
    }
    
    init();
    Runtime runtime(argv[1], argv[2], argv[3]);
    runtime.run();
    return 0;
}
