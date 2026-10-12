#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

int main() {
    // 1. Инициализация (как и прежде)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    const int DATA_SIZE = 1024;
    size_t bytes = sizeof(float) * DATA_SIZE;

    // 2. ВЫДЕЛЕНИЕ ПАМЯТИ ZERO-COPY
    // Флаг CL_MEM_ALLOC_HOST_PTR заставляет драйвер выделить память, 
    // к которой встроенный графический чип AMD имеет прямой доступ напрямую.
    cl::Buffer dev_A(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes);
    cl::Buffer dev_B(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes);
    cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes);

    // 3. Получаем обычные C++ указатели на эту общую память (Mapping)
    float* host_A = (float*)queue.enqueueMapBuffer(dev_A, CL_TRUE, CL_MAP_WRITE, 0, bytes);
    float* host_B = (float*)queue.enqueueMapBuffer(dev_B, CL_TRUE, CL_MAP_WRITE, 0, bytes);

    // Заполняем данные прямо на CPU
    for (int i = 0; i < DATA_SIZE; ++i) {
        host_A[i] = 1.0f;
        host_B[i] = 2.0f;
    }

    // Возвращаем права на память обратно видеокарте (Unmap) перед вычислениями
    queue.enqueueUnmapMemObject(dev_A, host_A);
    queue.enqueueUnmapMemObject(dev_B, host_B);

    // 4. Загрузка и компиляция ядра (код kernel.cl менять не нужно!)
    std::ifstream kernel_file("kernel_2.cl");
    std::stringstream src_stream;
    src_stream << kernel_file.rdbuf();
    std::string src_code = src_stream.str();
    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});
    cl::Program program(context, sources);
    program.build({device});

    // 5. Передача аргументов и запуск
    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, dev_A);
    kernel.setArg(1, dev_B);
    kernel.setArg(2, dev_C);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NullRange);
    
    // 6. ЧТЕНИЕ РЕЗУЛЬТАТА БЕЗ КОПИРОВАНИЯ
    // Вместо enqueueReadBuffer мы просто маппим буфер C в пространство CPU.
    // Драйвер AMD не будет ничего копировать — он просто даст доступ к физическим ячейкам ОЗУ.
    float* host_C = (float*)queue.enqueueMapBuffer(dev_C, CL_TRUE, CL_MAP_READ, 0, bytes);

    // Проверяем результат
    std::cout << "Zero-Copy Result validation:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << host_A[i] << " + " << host_B[i] << " = " << host_C[i] << std::endl;
    }

    // Освобождаем маппинг
    queue.enqueueUnmapMemObject(dev_C, host_C);
    
    std::cout << "\nSUCCESS! Zero-copy memory architecture utilized." << std::endl;
    return 0;
}
