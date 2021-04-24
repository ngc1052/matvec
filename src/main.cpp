#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "util.h"
#include "matrix.h"
#include <vector>
#include <cmath>

int main()
{
    try
    {
        cl::Context      context;
        cl::CommandQueue queue;
        cl::Program      program;
        initializeOpenCL(context, queue, program);
        auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer>(program, "matvec");

        constexpr size_t dim = 10;
        Matrix mat(dim);
        std::vector<cl_float> inVector(dim), outVector(dim);
        std::fill(inVector.begin(), inVector.end(), 1.0);

        cl::Buffer bufferIn (context, inVector.begin(),  inVector.end(),  true  /* readOnly */);
        cl::Buffer bufferOut(context, outVector.begin(), outVector.end(), false /* readOnly */);
        cl::Buffer bufferMat(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pow(mat.getSize(),2)*sizeof(float), mat.getElements());

        auto numWorkItems     = cl::NDRange{dim};
        auto kernelParameters = cl::EnqueueArgs(queue, numWorkItems);
        cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut));
        kernel_event.wait();
        std::cout << "Device (kernel) execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us.\n\n";

        cl::copy(queue, bufferOut, outVector.begin(), outVector.end());
        std::cout << "Output vector: \n";
        for (const auto num : outVector)
            std::cout << num << " ";
        std::cout << std::endl;
    }
    catch (cl::BuildError error) // If kernel failed to build
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
    catch (cl::Error error) // If any OpenCL error happes
    {
        std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;

        std::exit( error.err() );
    }
    catch (std::exception error) // If STL/CRT error occurs
    {
        std::cerr << "Error: " << error.what() << std::endl;

        std::exit( EXIT_FAILURE );
    }
    return 0;
}