#include <stdexcept>
#include <vector>
#include <cstdio>
#include <chrono>
#include <cstdlib>

// A matrix class is used to simplify the code for GPU matrix operations.
// An NxN matrix is represented as a one-dimensional vector with the size N^2.
// Storing it in a multidimensional vector is possible, but it is not recommended for GPUs.
// I have personally tested using the same implementation for CPUs but the speedup is not worth noting.
template <class T>
class Matrix {
public:
    explicit Matrix() : start(0), end(0) {}

    explicit Matrix(int size) {
        allocate(size * size);
    }

    ~Matrix() {
        free();
    }

    T* getMat() {
        return start;
    }

    void set(const T* src, int size) {
        cudaError_t result = cudaMemcpy(start, src, size * size * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to copy to the GPU");
        }
    }

    void get(T* dest, int size) {
        cudaError_t result = cudaMemcpy(dest, start, size * size * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to copy to the CPU memory");
        }
    }
private:
    void allocate(int size) {
        cudaError_t result = cudaMalloc((void**)&start, size * sizeof(T));
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to allocate memory");
        }
        end = start + size;
    }

    void free() {
        if (start != nullptr) {
            cudaFree(start);
            start = end = nullptr;
        }
    }

    T* start;
    T* end;
};

__global__ void matMulKernel(const float* A, const float* B, float* C, int size) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        for (int i = 0; i < size; i++) {
            C[row * size + col] += A[row * size + i] * B[i * size + col];
        }
    }
}

cudaError_t matMul(float* A, float* B, float* C, int size) {
    dim3 threadsPerBlock(size, size);
    dim3 blocksPerGrid(1, 1);
    if (size * size > 512) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        blocksPerGrid.x = ceil(double(size) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(size) / double(threadsPerBlock.y));
    }

    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

int main() {
    // C++ vectors are used to represent a matrix for easier computing on both CPU and GPU
    int size = 2048;
    std::vector<float> matA(size * size, 2);
    std::vector<float> matB(size * size, 7);
    std::vector<float> matResult(size * size);

    Matrix<float> gpuMatA(size);
    Matrix<float> gpuMatB(size);
    Matrix<float> gpuMatResult(size);

    gpuMatA.set(&matA[0], size);
    gpuMatB.set(&matB[0], size);

    auto start = std::chrono::high_resolution_clock::now();
    matMul(gpuMatA.getMat(), gpuMatA.getMat(), gpuMatResult.getMat(), size);
    gpuMatResult.get(&matResult[0], size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("GPU: %fs\n", elapsed.count());
    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}