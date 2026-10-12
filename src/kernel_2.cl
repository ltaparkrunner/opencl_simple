__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    // Получаем уникальный индекс потока
    int id = get_global_id(0);
    
    // Каждый поток складывает только один элемент
    C[id] = A[id] + B[id];
}
