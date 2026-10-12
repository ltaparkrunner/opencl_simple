cmake_minimum_required(VERSION 3.22)
project(OpenCL_ZeroCopy_Demo LANGUAGES CXX)

# 1. Требуем современный стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Автоматический поиск OpenCL в системе (сработает для ROCm, CUDA и Intel)
find_package(OpenCL REQUIRED)

# 3. Объявляем наш исполняемый файл
add_executable(opencl_test src/simple_cl_3.cpp)

# 4. Подключаем пути к заголовкам и линкуем библиотеку OpenCL
target_include_directories(opencl_test PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(opencl_test PRIVATE OpenCL::OpenCL)

# 5. Копируем файл ядра kernel.cl в папку сборки при компиляции,
# чтобы исполняемый файл всегда видел его рядом с собой.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_2.cl
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_2.cl
    COPYONLY
)
