#include "application.h"

#include "matrix.h"
#include "util.h"

#include <vector>
#include <cmath>
#include <iostream>

void Application::run_v1(const std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    const size_t dim = atoi(args[2].c_str());

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);

    const auto globalSize       = cl::NDRange{dim};
    const auto groupSize        = cl::NDRange{1};
    const auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer>(program, "matvec_v1");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());

    mat.actsOnVector(inVector, referenceVector);
}

void Application::run_v2(const std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    if(args.size() < 4)
        throw std::invalid_argument("Method 2 requires 3 arguments: {method} {dimension} {numItemsInGroupRow}");

    const size_t dim = atoi(args[2].c_str());
    const size_t numItemsInGroupRow = atoi(args[3].c_str());

    if(numItemsInGroupRow > dim || dim <= 0 || numItemsInGroupRow <= 0 || dim % numItemsInGroupRow != 0)
        throw std::invalid_argument("Invalid dimension or numItemsInGroupRow given to method 2. dim: " + std::to_string(dim) + ", numItemsInGroupRow: " + std::to_string(numItemsInGroupRow));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true  /* readOnly */);
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false /* readOnly */);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true);
    auto work = cl::Local((numItemsInGroupRow+1)*sizeof(float));

    const auto globalSize       = cl::NDRange{dim*numItemsInGroupRow};
    const auto groupSize        = cl::NDRange{numItemsInGroupRow};
    const auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v2");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, work));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());

    mat.actsOnVector(inVector, referenceVector);
}

void Application::run_v3(const std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    if(args.size() < 5)
        throw std::invalid_argument("Method 3 requires 3 arguments: {method} {dimension} {numItemsInGroupRow} {numItemsInGroupColumn}");

    const size_t dim = atoi(args[2].c_str());
    const size_t numItemsInGroupRow = atoi(args[3].c_str());    // number of rows to calculate in a work-group
    const size_t numItemsInGroupColumn = atoi(args[4].c_str()); // number of work-items that calculate a row

    if(dim <= 0 || numItemsInGroupRow > dim || numItemsInGroupRow <= 0 || dim % numItemsInGroupRow != 0 
    || numItemsInGroupColumn > dim || numItemsInGroupColumn <= 0 || dim % numItemsInGroupColumn != 0)
        throw std::invalid_argument("Invalid dimension, numItemsInGroupRow or numItemsInGroupColumn given to method 3. "
        "dim: " + std::to_string(dim) + ", numItemsInGroupRow: " + std::to_string(numItemsInGroupRow) + ", numItemsInGroupColumn: " + std::to_string(numItemsInGroupColumn));

    Matrix mat(dim);
    initializeVectors(dim);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),    inVector.end(),    true );
    cl::Buffer bufferOut(context, outVector.begin(),   outVector.end(),   false);
    cl::Buffer bufferMat(context, matElements.begin(), matElements.end(), true );
    auto work = cl::Local(numItemsInGroupRow * numItemsInGroupColumn * sizeof(float));

    const auto globalSize       = cl::NDRange(dim, numItemsInGroupColumn);
    const auto groupSize        = cl::NDRange(numItemsInGroupRow, numItemsInGroupColumn);
    const auto kernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);

    auto kernelEntry = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v3");
    cl::Event kernel_event(kernelEntry(kernelParameters, bufferMat, dim, bufferIn, bufferOut, work));
    kernel_event.wait();

    if(printProfileInfo)
        std::cout << "Execution took: " << get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(kernel_event).count() << " us" << std::endl;

    cl::copy(queue, bufferOut, outVector.begin(), outVector.end());
    mat.actsOnVector(inVector, referenceVector);
}

void Application::run_v4(const std::vector<std::string>& args)
{
    initializeOpenCL(context, device, queue, program);

    if(args.size() < 5)
        throw std::invalid_argument("Method 4 requires 3 arguments: {method} {dimension} {numItemsInGroupColumn} {numGroupsInMatrixRow}");

    const size_t dim = atoi(args[2].c_str());
    const size_t numItemsInGroupColumn = atoi(args[3].c_str()); // number of rows to calculate in a work-group
    const size_t numGroupsInMatrixRow = atoi(args[4].c_str());  // number of work-groups that calculate a row

    if(dim <= 0 || numItemsInGroupColumn > dim || numItemsInGroupColumn <= 0 || dim % numItemsInGroupColumn != 0 
    || numGroupsInMatrixRow > dim || numGroupsInMatrixRow <= 0 || dim % numGroupsInMatrixRow != 0)
        throw std::invalid_argument("Invalid dimension, numItemsInGroupRow or numGroupsInMatrixRow given to method 4. "
        "dim: " + std::to_string(dim) + ", numItemsInGroupColumn: " + std::to_string(numItemsInGroupColumn) + ", numGroupsInMatrixRow: " + std::to_string(numGroupsInMatrixRow));

    if(dim < numGroupsInMatrixRow * numItemsInGroupColumn)
        throw std::invalid_argument("dimension < numGroupsInMatrixRow * numItemsInGroupColumn: work-items have to load at least one element into local memory.");

    Matrix mat(dim);
    initializeVectors(dim);
    std::vector<cl_float> extendedOutVector(dim*numGroupsInMatrixRow);

    auto matElements = mat.getElements();
    cl::Buffer bufferIn (context, inVector.begin(),            inVector.end(),          true);
    cl::Buffer bufferOut(context, extendedOutVector.begin(),   extendedOutVector.end(), false);
    cl::Buffer bufferMat(context, matElements.begin(),         matElements.end(),       true);

    const auto elementsPerItem = dim / numGroupsInMatrixRow;
    auto work = cl::Local(elementsPerItem * sizeof(float));

    auto       globalSize       = cl::NDRange(dim, numGroupsInMatrixRow);
    const auto groupSize        = cl::NDRange(numItemsInGroupColumn, 1);
    const auto computeKernelParameters = cl::EnqueueArgs(queue, globalSize, groupSize);
    auto computeKernel = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "matvec_v4");
    cl::Event computeEvent(computeKernel(computeKernelParameters, bufferMat, dim, bufferIn, bufferOut, work));
    computeEvent.wait();

    globalSize = cl::NDRange(dim);
    const auto reduceKernelParameters = cl::EnqueueArgs(queue, globalSize);
    auto reduceKernel = cl::KernelFunctor<cl::Buffer, cl_int>(program, "reduceRows");
    cl::Event reduceEvent(reduceKernel(reduceKernelParameters, bufferOut, numGroupsInMatrixRow));
    reduceEvent.wait();

    if(printProfileInfo)
    {            
        auto duration1 = get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(computeEvent).count();
        auto duration2 = get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::microseconds>(reduceEvent).count();
        std::cout << "Execution took: " << duration1 + duration2 << " us" << std::endl;
    }

    cl::copy(queue, bufferOut, extendedOutVector.begin(), extendedOutVector.end());
    for(size_t i = 0; i < extendedOutVector.size(); i += numGroupsInMatrixRow)
        outVector[i/numGroupsInMatrixRow] = extendedOutVector[i];

    mat.actsOnVector(inVector, referenceVector);
}

bool Application::matchOutAndReferenceVectors() const
{
    static const float tolerance = 1e-3;
    for (size_t i = 0; i < outVector.size(); i++)
    {
        const auto error = abs(outVector[i] - referenceVector[i]);
        if (error > tolerance)
            throw std::runtime_error{"Validation failed, error: " + std::to_string(error) + ", at index: " + std::to_string(i)};
    }
    return 1;
}

void Application::initializeVectors(const size_t dim)
{
    inVector.resize(dim);
    outVector.resize(dim);
    referenceVector.resize(dim);

    auto prng = [engine = std::default_random_engine{},
                distribution = std::uniform_real_distribution<cl_float>{ -1.0, 1.0 }]() mutable { return distribution(engine); };

    std::generate_n(inVector.begin(), inVector.size(), prng);
    //std::fill(inVector.begin(), inVector.end(), 1.0);
}

void Application::runCPUVersion(const std::vector<std::string>& args)
{
    const size_t dim = atoi(args[2].c_str());

    Matrix mat(dim);
    initializeVectors(dim);

	auto tStart = std::chrono::high_resolution_clock::now();
    mat.actsOnVector(inVector, outVector);
	auto tEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    if(printProfileInfo)
        std::cout << "Execution took: " << duration << " us" << std::endl;

    std::copy(outVector.cbegin(), outVector.cend(), referenceVector.begin());
}