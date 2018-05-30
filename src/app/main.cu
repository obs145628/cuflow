#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
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

    try
    {
        init();
        Runtime runtime(argv[1], argv[2], argv[3]);
        runtime.run();
    }

    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
