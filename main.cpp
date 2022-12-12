#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <vector>

int** init(int size) {
    int** newMat = new int*[size];
    for (int i = 0; i < size; i++) {
        newMat[i] = new int[size];
        for (int j = 0; j < size; j++) {
            newMat[i][j] = 2;
        }
    }
    return newMat;
}

int** add(int** mat1, int** mat2, int size) {
    int** newMat = init(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            newMat[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    return newMat;
}

int** sub(int** mat1, int** mat2, int size) {
    int** newMat = init(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            newMat[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return newMat;
}

int** avx_multiply(int** A, int** B, int size) {
    int** C = init(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
        }
    }
    __m256i mat1 = _mm256_setzero_si256();
    __m256i mat2 = _mm256_setzero_si256();
    __m256i mult = _mm256_setzero_si256();
#pragma omp parallel for simd default(none) shared(A, B, C, size, mat1, mat2, mult) // Pragma directive for parallelization and SIMD vectorization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat1 = _mm256_set1_epi32(A[i][j]);
            for (int k = 0; k < size; k += 8) { // Iterates every 8 integers due to 256-bit register size
                mat2 = _mm256_loadu_si256((__m256i *)(&B[j][k]));
                mult = _mm256_loadu_si256((__m256i *)(&C[i][k]));
                mult = _mm256_add_epi32(_mm256_mullo_epi32(mat1, mat2), mult);
                _mm256_storeu_si256((__m256i *)(&C[i][k]), mult);
            }
        }
    }
    return C;
}

// Using void because it's better to have the result matrix as a parameter, too late to change the others now
void avx_multiply_f(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, int size) {
    __m256 a, b, mult;
#pragma omp parallel for simd default(none) shared(A, B, C, size) private(a, b, mult)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a = _mm256_set1_ps(A[i][j]);
            for (int k = 0; k < size; k += 8) {
                b = _mm256_loadu_ps(&B[j][k]);
                mult = _mm256_loadu_ps(&C[i][k]);
                mult = _mm256_fmadd_ps(a, b, mult);
                _mm256_storeu_ps(&C[i][k], mult);
            }
        }
    }
}

// For matrices with a N^2 vector representation.
// It provides a slight speedup but it is less readable.
void avx_multiply_fv(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int size) {
    __m256 a, b, mult;
#pragma omp parallel for simd default(none) shared(A, B, C, size) private(a,b, mult)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a = _mm256_set1_ps(A[i*size+j]);
            for (int k = 0; k < size; k += 8) {
                b = _mm256_loadu_ps(&B[j*size+k]);
                mult = _mm256_loadu_ps(&C[i*size+k]);
                mult = _mm256_fmadd_ps(a, b, mult);
                _mm256_storeu_ps(&C[i*size+k], mult);
            }
        }
    }
}

int** normal_multiply(int** A, int** B, int size) {
    int** C = init(size);
#pragma omp parallel for simd default(none) shared(A, B, C, size) // Pragma directive for parallelization and SIMD vectorization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] = A[i][k] * B[k][j] + C[i][j];
            }
        }
    }
    return C;
}

int** norm_multiply(int A[2][2], int B[2][2], int size) {
    int** C = init(size);
#pragma omp parallel for simd default(none) shared(A, B, C, size) // Pragma directive for parallelization and SIMD vectorization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] = A[i][k] * B[k][j] + C[i][j];
            }
        }
    }
    return C;
}

static inline uint32_t log2i(const uint32_t x) {
    uint32_t y;
    asm ( "\tbsr %1, %0\n"
        : "=r"(y)
        : "r" (x)
    );
    return y;
}

#pragma omp declare simd
int** strassen(int** A, int** B, int size) {

    if (size <= 64) {
        return normal_multiply(A, B, size);
    }

    int** C = init(size);
    int half = size / 2;

    int** A11 = init(half);
    int** A12 = init(half);
    int** A21 = init(half);
    int** A22 = init(half);

    int** B11 = init(half);
    int** B12 = init(half);
    int** B21 = init(half);
    int** B22 = init(half);

    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }

    int** M1 = strassen(add(A11, A22, half), add(B11, B22, half), half);
    int** M2 = strassen(add(A21, A22, half), B11, half);
    int** M3 = strassen(A11, sub(B12, B22, half), half);
    int** M4 = strassen(A22, sub(B21, B11, half), half);
    int** M5 = strassen(add(A11, A12, half), B22, half);
    int** M6 = strassen(sub(A21, A11, half), add(B11, B12, half), half);
    int** M7 = strassen(sub(A12, A22, half), add(B21, B22, half), half);

    int** C11 = add(sub(add(M1, M4, half), M5, half), M7, half);
    int** C12 = add(M3, M5, half);
    int** C21 = add(M2, M4, half);
    int** C22 = add(sub(add(M1, M3, half), M2, half), M6, half);

    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C[i][j] = C11[i][j];
            C[i][j + half] = C12[i][j];
            C[i + half][j] = C21[i][j];
            C[i + half][j + half] = C22[i][j];
        }
    }

    // Deallocate memory
    for (int i = 0; i < half; i++) {
        delete[] A11[i];
        delete[] A12[i];
        delete[] A21[i];
        delete[] A22[i];

        delete[] B11[i];
        delete[] B12[i];
        delete[] B21[i];
        delete[] B22[i];

        delete[] C11[i];
        delete[] C12[i];
        delete[] C21[i];
        delete[] C22[i];

        delete[] M1[i];
        delete[] M2[i];
        delete[] M3[i];
        delete[] M4[i];
        delete[] M5[i];
        delete[] M6[i];
        delete[] M7[i];
    }

    delete[] A11;
    delete[] A12;
    delete[] A21;
    delete[] A22;

    delete[] B11;
    delete[] B12;
    delete[] B21;
    delete[] B22;

    delete[] C11;
    delete[] C12;
    delete[] C21;
    delete[] C22;

    delete[] M1;
    delete[] M2;
    delete[] M3;
    delete[] M4;
    delete[] M5;
    delete[] M6;
    delete[] M7;
    return C;
}

int** strassen_helper(int** A, int** B, int size) {
    if (size <= 64) {
        return normal_multiply(A, B, size);
    }
    if ((size & (size - 1)) != 0) {
        int copySize = static_cast<int>(pow(2, log2i(size) + 1));
        int** copyA = init(copySize);
        int** copyB = init(copySize);
        for (int i = 0; i < copySize; i++) {
            for (int j = 0; j < copySize; j++) {
                if (i < size && j < size) {
                    copyA[i][j] = A[i][j];
                    copyB[i][j] = B[i][j];
                }
                else {
                    copyA[i][j] = 0;
                    copyB[i][j] = 0;
                }
            }
        }
        int** result = strassen(copyA, copyB, copySize);
        int** finalResult = init(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                finalResult[i][j] = result[i][j];
            }
            delete[] result[i];
        }
        delete[] result;
        return finalResult;
    }
    return strassen(A, B, size);
}

int** tensor_decomposition(int A[2][2], int B[2][2]) {
    int size = 2;
    int rank = 7;
    int* m = new int[rank];
    int** C = init(size);
    int u[4][7] = {{1, 0, 1, 0, 1, -1, 0},
               {0, 0, 0, 0, 1, 0, 1},
               {0, 1, 0, 0, 0, 1, 0},
               {1, 1, 0, 1, 0, 0, -1}};
    int v[4][7] = {{1, 1, 0, -1, 0, 1, 0},
                   {0, 0, 1, 0, 0, 1, 0},
                   {0, 0, 0, 1, 0, 0, 1},
                   {1, 0, -1, 0, 1, 0, 1}};
    int w[4][7] = {{1, 0, 0, 1, -1, 0, 1},
                   {0, 0, 1, 0, 1, 0, 0},
                   {0, 1, 0, 1, 0, 0, 0},
                   {1, -1, 1, 0, 0, 1, 0}};

    for (int i = 0; i < rank; i++) {
        m[i] = 0;
        int temp = 0;
        for (int j = 0; j < (size << 1); j++) {
            m[i] += u[j][i] * A[j / size][j % size];
            temp += v[j][i] * B[j / size][j % size];
        }
        m[i] *= temp;
    }
    int temp[4] = {0, 0, 0, 0};
    for (int i = 0; i < (size << 1); i++) {
        for (int j = 0; j < rank; j++) {
            temp[i] += w[i][j] * m[j];
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = temp[i * size + j];
        }
    }
    return C;
}

int main() {
    // For simplicity, only square matrices are used, and built-in C++ vectors are not used in place of primitive integer 2D arrays.
    int size = 4096; // A 2048x2048 matrix is used for testing.

    // Generate 2 matrices with junk values.
    int** A = init(size);
    int** B = init(size);

    // First algorithm to use is Strassen's algorithm.
    auto start_time = std::chrono::high_resolution_clock::now();
    int** C = strassen_helper(A, B, size);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Strassen's Algorithm: " << elapsed.count() << "s" << std::endl;

    for (int i = 0; i < size; i++) {
        delete[] C[i];
    }
    delete[] C;

    // Naive algorithm.
    start_time = std::chrono::high_resolution_clock::now();
    int** D = normal_multiply(A, B, size);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    std::cout << "Naive Algorithm: " << elapsed.count() << "s" << std::endl;

    for (int i = 0; i < size; i++) {
        delete[] D[i];
    }
    delete[] D;

    // AVX algorithm.
    start_time = std::chrono::high_resolution_clock::now();
    int** E = avx_multiply(A, B, size);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    std::cout << "AVX Algorithm: " << elapsed.count() << "s" << std::endl;

    for (int i = 0; i < size; i++) {
        delete[] E[i];
    }
    delete[] E;

    for (int i = 0; i < size; i++) {
        delete[] A[i];
        delete[] B[i];
    }
    delete[] A;
    delete[] B;

    // Using 2D float vectors for an even ground with the GPU.
    std::vector<std::vector<float>> vecA(size, std::vector<float>(size, 2));
    std::vector<std::vector<float>> vecB(size, std::vector<float>(size, 4));
    std::vector<std::vector<float>> vecC(size, std::vector<float>(size, 0));
//    std::vector<float> vecA(size*size, 2);
//    std::vector<float> vecB(size*size, 4);
//    std::vector<float> vecC(size*size, 0);

    auto start = std::chrono::high_resolution_clock::now();
    avx_multiply_f(vecA, vecB, vecC, size);
//    avx_multiply_fv(vecA, vecB, vecC, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "AVX (float): " << diff.count() << "s" << std::endl;

// Only for verification purposes.
//    int count = 0;
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            if (D[i][j] != E[i][j] || D[i][j] != C[i][j]) {
//                std::cout << "Not match: " << i << ", " << j << std::endl;
//                std::cout << "Expected result: " << C[i][j] << std::endl;
//                std::cout << "Got: " << D[i][j] << std::endl;
//                count++;
//            }
//        }
//    }
//    std::cout << "Number of mismatches: " << count << std::endl;

    int G[2][2] = {{1, 2},
                   {3, 4}};
    int H[2][2] = {{5, 6},
                   {7, 8}};
    start_time = std::chrono::high_resolution_clock::now();
    int** P = tensor_decomposition(G, H);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    std::cout << "Tensor Decomposition Algorithm: " << elapsed.count() << "s" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    int** Q = norm_multiply(G, H, 2);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    std::cout << "Naive Algorithm 2: " << elapsed.count() << "s" << std::endl;

// Only for verification purposes.
//    for (int i = 0; i < 2; i++) {
//        for (int j = 0; j < 2; j++) {
//            if (P[i][j] != Q[i][j]) {
//                std::cout << "Not match: " << i << ", " << j << std::endl;
//                std::cout << "Expected result: " << P[i][j] << std::endl;
//                std::cout << "Got: " << Q[i][j] << std::endl;
//            }
//        }
//    }
    return 0;
}
