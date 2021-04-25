#pragma once

#include <iostream>
#include <CL/cl2.hpp>

#include <vector>
#include <fstream>


void printPlatformInfo(const cl::Platform& platform)
{
    std::cout 
        << "Vendor: "  << platform.getInfo<CL_PLATFORM_VENDOR>() 
        << ", name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
}

void printDeviceInfo(const cl::Device& device)
{
    std::cout 
        << "Vendor: "  << device.getInfo<CL_DEVICE_VENDOR>() 
        << ", name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
}

template <cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
auto get_duration(cl::Event& ev)
{
    return std::chrono::duration_cast<Dur>(std::chrono::nanoseconds{ ev.getProfilingInfo<To>() - ev.getProfilingInfo<From>() });
}

void initializeOpenCL(cl::Context& context, cl::Device& device, cl::CommandQueue& queue, cl::Program& program)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

#ifdef PRINT_HARDWARE_INFO
    std::cout << "Platforms:" << std::endl;
    for (const auto &platform : platforms)
        printPlatformInfo(platform);
    std::cout << std::endl;
#endif

    auto platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        throw std::runtime_error("No GPU found! Terminating...");

#ifdef PRINT_HARDWARE_INFO
    std::cout << "Devices:" << std::endl;
    for (const auto &device : devices)
        printDeviceInfo(device);
    std::cout << std::endl;
#endif

    device = devices[0];

    std::vector<cl_context_properties> props{
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>((platform)()),
        0};
    context = {devices, props.data()};
    queue = {context, device, cl::QueueProperties::Profiling};

    std::ifstream clSource("src/matvec.cl");
    if (!clSource.is_open())
        throw std::runtime_error("Kernel source not found!");

    program = {context, std::string{std::istreambuf_iterator<char>{clSource}, std::istreambuf_iterator<char>{}}};
    program.build({device});
}