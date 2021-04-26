#include "application.h"

#include "matrix.h"
#include "util.h"

#include <vector>
#include <cmath>
#include <iostream>

void Application::run_v1(std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    size_t dim = atoi(args[2].c_str());

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);

    auto globalSize       = cl::NDRange{dim};
    auto localSize        = cl::NDRange{1};
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, localSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer>(program, "matvec_v1");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());

    mat.actsOnVector(inVector, referenceVector);
}

void Application::run_v2(std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    if(args.size() < 4)
        throw std::invalid_argument("Method 2 requires 3 arguments: {method} {dimension} {rowBlockSize}");

    size_t dim = atoi(args[2].c_str());
    size_t rowBlockSize = atoi(args[3].c_str());

    if(rowBlockSize > dim || dim <= 0 || rowBlockSize <= 0 || dim % rowBlockSize != 0)
        throw std::invalid_argument("Invalid dimension or rowBlockSize given to method 2. dim: " + std::to_string(dim) + ", rowBlockSize: " + std::to_string(rowBlockSize));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);
    auto workArray = cl::Local((rowBlockSize+1)*sizeof(float));

    auto globalSize       = cl::NDRange{dim*rowBlockSize};
    auto localSize        = cl::NDRange{rowBlockSize};
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, localSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v2");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, workArray));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());

    mat.actsOnVector(inVector, referenceVector);
}

void Application::run_v3(std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    if(args.size() < 5)
        throw std::invalid_argument("Method 3 requires 3 arguments: {method} {dimension} {rowBlockSize} {columnBlockSize}");

    size_t dim = atoi(args[2].c_str());
    size_t rowBlockSize = atoi(args[3].c_str());    // number of rows to calculate in a work-group
    size_t columnBlockSize = atoi(args[4].c_str()); // number of work-items that calculate a row

    if(dim <= 0 || rowBlockSize > dim || rowBlockSize <= 0 || dim % rowBlockSize != 0 
    || columnBlockSize > dim || columnBlockSize <= 0 || dim % columnBlockSize != 0)
        throw std::invalid_argument("Invalid dimension, rowBlockSize or columnBlockSIze given to method 3. "
        "dim: " + std::to_string(dim) + ", rowBlockSize: " + std::to_string(rowBlockSize) + ", columnBlockSIze: " + std::to_string(columnBlockSize));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true );
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true );
    auto workArray = cl::Local(rowBlockSize * columnBlockSize * sizeof(float));

    auto globalSize       = cl::NDRange(dim, columnBlockSize);
    auto localSize        = cl::NDRange(rowBlockSize, columnBlockSize);
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, localSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v3");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, workArray));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());
    mat.actsOnVector(inVector, referenceVector);
}

bool Application::matchOutAndReferenceVectors() const
{
    static const float tolerance = 1e-2;
    for (size_t i = 0; i < outVector.size(); i++)
    {
        const auto error = abs(outVector[i] - referenceVector[i]);
        if (error > tolerance)
            throw std::runtime_error{"Validation failed, error: " + std::to_string(error)};
    }
    return 1;
}

void Application::initializeVectors(const size_t dim)
{
    inVector.resize(dim);
    outVector.resize(dim);
    referenceVector.resize(dim);
    std::fill(inVector.begin(), inVector.end(), 1.0);
}