#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h> // For AVX2 intrinsics

void scalar_add(const float* a, const float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simd_add(const float* a, const float* b, float* result, size_t n) {
    size_t i = 0;
    // Process 8 floats at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vsum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result + i, vsum);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    std::vector<float> a(N, 1.0f), b(N, 2.0f), result_scalar(N), result_simd(N);

    // Scalar addition
    auto t1 = std::chrono::high_resolution_clock::now();
    scalar_add(a.data(), b.data(), result_scalar.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();

    // SIMD addition
    auto t3 = std::chrono::high_resolution_clock::now();
    simd_add(a.data(), b.data(), result_simd.data(), N);
    auto t4 = std::chrono::high_resolution_clock::now();

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (result_scalar[i] != result_simd[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Results match: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "Scalar time: "
              << std::chrono::duration<double, std::milli>(t2 - t1).count()
              << " ms\n";
    std::cout << "SIMD time: "
              << std::chrono::duration<double, std::milli>(t4 - t3).count()
              << " ms\n";
    return 0;
}