#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

int main() {
    // 1. Инициализация устройства (берем первое доступное)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    // 2. Создание контекста и очереди команд
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // 3. Подготовка данных на стороне CPU (Host)
    const int DATA_SIZE = 1024;
    std::vector<float> host_A(DATA_SIZE, 1.0f); // Массив единиц
    std::vector<float> host_B(DATA_SIZE, 2.0f); // Массив двоек
    std::vector<float> host_C(DATA_SIZE, 0.0f); // Массив для результата

    // 4. Выделение буферов в памяти видеокарты (Device)
    cl::Buffer dev_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, host_A.data());
    cl::Buffer dev_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DATA_SIZE, host_B.data());
    cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE);

    // 5. Загрузка и компиляция кода ядра (.cl)
    std::ifstream kernel_file("kernel_2.cl");
    std::stringstream src_stream;
    src_stream << kernel_file.rdbuf();
    std::string src_code = src_stream.str();

    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});
    cl::Program program(context, sources);
    
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // 6. Создание объекта ядра и передача аргументов
    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, dev_A);
    kernel.setArg(1, dev_B);
    kernel.setArg(2, dev_C);

    // 7. Запуск вычислений на GPU
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NullRange);
    queue.finish(); // Ждем окончания работы GPU

    // 8. Чтение результата обратно в оперативную память
    queue.enqueueReadBuffer(dev_C, CL_TRUE, 0, sizeof(float) * DATA_SIZE, host_C.data());

    // 9. Проверка корректности (1.0 + 2.0 должно быть равно 3.0)
    std::cout << "Result validation (first 5 elements):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << host_A[i] << " + " << host_B[i] << " = " << host_C[i] << std::endl;
    }

    std::cout << "\nSUCCESS! GPU completed computations seamlessly." << std::endl;
    return 0;
}
