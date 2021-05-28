#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl2.hpp>

class Application
{
public:
    Application(bool printProfileInfo = false) : printProfileInfo(printProfileInfo) {}

    void run_v1(const std::vector<std::string>& args);
    void run_v2(const std::vector<std::string>& args);
    void run_v3(const std::vector<std::string>& args);
    void run_v4(const std::vector<std::string>& args);

    void runCPUVersion(const std::vector<std::string>& args);

    bool matchOutAndReferenceVectors() const;

    private:
        void initializeVectors(const size_t dim);

        cl::Context      context;
        cl::CommandQueue queue;
        cl::Program      program;
        cl::Device       device;

        std::vector<cl_float> inVector, outVector, referenceVector;

        bool printProfileInfo;
};