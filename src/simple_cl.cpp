#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Get available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found. Check your drivers!" << std::endl;
        return 1;
    }

    std::cout << "Found " << platforms.size() << " OpenCL Platform(s):\n" << std::endl;

    // 2. Iterate through platforms and list their devices
    for (const auto& platform : platforms) {
        std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (const auto& device : devices) {
            std::cout << "  -> Device Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "     Device Type: ";
            
            cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
            if (type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
            if (type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
            std::cout << "\n     Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        }
        std::cout << "-----------------------------------------------" << std::endl;
    }

    return 0;
}
