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
    auto groupSize        = cl::NDRange{1};
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

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
        throw std::invalid_argument("Method 2 requires 3 arguments: {method} {dimension} {numItemsInGroupRow}");

    size_t dim = atoi(args[2].c_str());
    size_t numItemsInGroupRow = atoi(args[3].c_str());

    if(numItemsInGroupRow > dim || dim <= 0 || numItemsInGroupRow <= 0 || dim % numItemsInGroupRow != 0)
        throw std::invalid_argument("Invalid dimension or numItemsInGroupRow given to method 2. dim: " + std::to_string(dim) + ", numItemsInGroupRow: " + std::to_string(numItemsInGroupRow));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);
    auto work = cl::Local((numItemsInGroupRow+1)*sizeof(float));

    auto globalSize       = cl::NDRange{dim*numItemsInGroupRow};
    auto groupSize        = cl::NDRange{numItemsInGroupRow};
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v2");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, work));
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
        throw std::invalid_argument("Method 3 requires 3 arguments: {method} {dimension} {numItemsInGroupRow} {numItemsInGroupColumn}");

    size_t dim = atoi(args[2].c_str());
    size_t numItemsInGroupRow = atoi(args[3].c_str());    // number of rows to calculate in a work-group
    size_t numItemsInGroupColumn = atoi(args[4].c_str()); // number of work-items that calculate a row

    if(dim <= 0 || numItemsInGroupRow > dim || numItemsInGroupRow <= 0 || dim % numItemsInGroupRow != 0 
    || numItemsInGroupColumn > dim || numItemsInGroupColumn <= 0 || dim % numItemsInGroupColumn != 0)
        throw std::invalid_argument("Invalid dimension, numItemsInGroupRow or columnBlockSIze given to method 3. "
        "dim: " + std::to_string(dim) + ", numItemsInGroupRow: " + std::to_string(numItemsInGroupRow) + ", columnBlockSIze: " + std::to_string(numItemsInGroupColumn));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true );
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true );
    auto work = cl::Local(numItemsInGroupRow * numItemsInGroupColumn * sizeof(float));

    auto globalSize       = cl::NDRange(dim, numItemsInGroupColumn);
    auto groupSize        = cl::NDRange(numItemsInGroupRow, numItemsInGroupColumn);
    auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v3");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, work));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());
    mat.actsOnVector(inVector, referenceVector);
}

bool Application::matchOutAndReferenceVectors() const
{
    static const float tolerance = 1e-4;
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