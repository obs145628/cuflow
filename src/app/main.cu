#include <iostream>
#include "init.hh"
#include "runtime.hh"


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: ./cuflow <cmd-file> <in-file> <out-file>.\n";
        return 1;
    }
    
    init();
    Runtime runtime(argv[1], argv[2], argv[3]);
    runtime.run();
    return 0;
}
