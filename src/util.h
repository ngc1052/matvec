#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl2.hpp>

void printPlatformInfo(const cl::Platform& platform);

void printDeviceInfo(const cl::Device& device);

template <cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
auto get_duration(cl::Event& ev)
{
    return std::chrono::duration_cast<Dur>(std::chrono::nanoseconds{ ev.getProfilingInfo<To>() - ev.getProfilingInfo<From>() });
}

void initializeOpenCL(cl::Context& context, cl::Device& device, cl::CommandQueue& queue, cl::Program& program);

std::string errorCodeToString(cl_int errorCode);