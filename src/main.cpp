#include "util.h"
#include "application.h"

#include <iostream>

int main(int argc, char* args[])
{
    try
    {
        if(argc < 3)
            throw std::invalid_argument("At least 2 arguments needed: {method} {dimension}");

        bool printProfileInfo = false;
        if(std::string(args[argc-1]) == "--profile")
            printProfileInfo = true;
        
        std::vector<std::string> argsVector(argc);
        for(int i = 0; i < argc; i++)
            argsVector[i] = std::string(args[i]);

        int choice = atoi(args[1]);
        Application app(printProfileInfo);
        switch(choice)
        {
            case 0:
                app.runCPUVersion(argsVector);
                break;
            case 1:
                app.run_v1(argsVector);
                break;
            case 2:
                app.run_v2(argsVector);
                break;
            case 3:
                app.run_v3(argsVector);
                break;
            case 4:
                app.run_v4(argsVector);
                break;
            default: throw std::invalid_argument("Unknown method!");
        }
        app.matchOutAndReferenceVectors();
    }
    catch (const cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit( error.err() );
    }
    catch (const cl::Error& error) // If any OpenCL error happes
    {
        cl_int errorCode = error.err();
        std::string errorString = errorCodeToString(errorCode);
        std::cerr << "Error: " << error.what() << "(" << errorCode << ": " << errorString << ")" << std::endl;

        std::exit( error.err() );
    }
    catch (const std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << "Error: " << error.what() << std::endl;

        std::exit( EXIT_FAILURE );
    }
    return 0;
}