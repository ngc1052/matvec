#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include "util.h"
#include "matrix.h"
#include <vector>
#include <cmath>

int main(int argc, char** args)
{
    try
    {
        cl::Context      context;
        cl::CommandQueue queue;
        cl::Program      program;
        cl::Device       device;
        initializeOpenCL(context, device, queue, program);
        auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v2");

        if(argc < 2)
            throw std::invalid_argument("1 argument needed.");
        size_t dim = atoi(args[1]);
        size_t divideRowsInto = 10;

        Matrix mat(dim);
        std::vector<cl_float> inVector(dim), outVector(dim);
        std::fill(inVector.begin(), inVector.end(), 1.0);

        auto matElements = mat.getElements();
        cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
        cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
        cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);
        auto rowPart = cl::Local((dim/divideRowsInto+1)*sizeof(float));

        auto numWorkItems     = cl::NDRange{dim*divideRowsInto};
        auto groupSize        = cl::NDRange{divideRowsInto};
        auto kernelParameters = cl::EnqueueArgs(queue, numWorkItems, groupSize);
        cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, rowPart));
        kernel_event.wait();
        std::cout << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << std::endl;

        cl::copy(queue, bufferOut, outVector.begin(), outVector.end());
        //std::cout << std::endl;
        //for(auto n : outVector)
        //    std::cout << n << std::endl;

        std::vector<cl_float> ref(dim);
        mat.actsOnVector(inVector, ref);

        //std::cout << std::endl;
        //for(auto n : ref)
        //    std::cout << n << std::endl;

        const float tolerance = 1e-3;
        for(size_t i = 0; i < dim; i++)
        {
            if(abs(outVector[i]-ref[i]) > tolerance)
                throw std::runtime_error{ "Validation failed." };
        }
        //auto result = std::mismatch(outVector.begin(), outVector.end(), ref.begin());
        //if (result.first != outVector.end() || result.second != ref.end()) 
        //    throw std::runtime_error{ "Validation failed." };
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
        std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;

        std::exit( error.err() );
    }
    catch (const std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << "Error: " << error.what() << std::endl;

        std::exit( EXIT_FAILURE );
    }
    return 0;
}