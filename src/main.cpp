#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <immintrin.h> // For AVX2 intrinsics
#include <thread>
#include <random>

void scalar_add(const float* a, const float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

void simd_add(const float* a, const float* b, float* result, size_t n) {
    size_t i = 0;
    // Process 8 floats at a time (AVX2)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vsum = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result + i, vsum);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

void worker(const float* a, const float* b, float* result, size_t start, size_t end) {
    simd_add(a + start, b + start, result + start, end - start);
}

void simd() {
    const size_t N = 1 << 20; // 1M elements
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk = N / num_threads;
    std::vector<float> a(N, 1.0f), b(N, 2.0f), result_scalar(N), result_simd(N);

    // Scalar addition
    auto t1 = std::chrono::high_resolution_clock::now();
    scalar_add(a.data(), b.data(), result_scalar.data(), N);
    auto t2 = std::chrono::high_resolution_clock::now();

    // SIMD addition
    auto t3 = std::chrono::high_resolution_clock::now();
    simd_add(a.data(), b.data(), result_simd.data(), N);
    auto t4 = std::chrono::high_resolution_clock::now();

    // Threaded SIMD addition
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? N : start + chunk;
        threads.emplace_back(worker, a.data(), b.data(), result_simd.data(), start, end);
    }
    for (auto& th : threads) th.join();
    
    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (result_scalar[i] != result_simd[i]) {
            correct = false;
            break;
        }
    }

    float percentIncrease = 100.0f * (t2 - t1).count() / (t4 - t3).count();
    std::cout << "Results match: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "Scalar time: "
              << std::chrono::duration<double, std::milli>(t2 - t1).count()
              << " ms\n";
    std::cout << "SIMD time: "
              << std::chrono::duration<double, std::milli>(t4 - t3).count()
              << " ms\n";
    std::cout << "Percent increase in speed: " << percentIncrease << "%\n";
}

void cache() {
    //cache friendly
    int size = 1 << 13;
    std::vector<std::vector<int>> a(size, std::vector<int>(size, 1));

    volatile long long sum = 0;
    auto friendStart = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 0; i < size; i++){
        for(unsigned int j = 0; j < size; j++){
            sum += a[i][j];
        }
    }
    std::cout << "sum = " << sum << "\n";
    auto friendEnd = std::chrono::high_resolution_clock::now();
    
    auto friendlyTime = std::chrono::duration<double, std::milli>(friendEnd-friendStart).count();

    auto unfriendStart = std::chrono::high_resolution_clock::now();
    sum = 0;
    for(unsigned int i = 0; i < size; i++){
        for(unsigned int j = 0; j < size; j++){
            sum += a[j][i];
        }
        
    }

    std::cout << "sum = " << sum << "\n";
    auto unfriendEnd = std::chrono::high_resolution_clock::now();
    auto unfriendTime = std::chrono::duration<double, std::milli>(unfriendEnd-unfriendStart).count();

    std::cout << "Cache friendly time: " << friendlyTime
              << "\nCache unfriendly time: " << unfriendTime
              << "\n";
}

void branchprediction(){
    size_t N = 1 << 20;
    std::vector<int> v(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 100);


    for (size_t i = 0; i < N; ++i) {
        v[i] = distrib(gen);
    }

    long long sum = 0;
    auto friendStart = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < N; i++){
        sum += v[i];
    }
    auto friendEnd = std::chrono::high_resolution_clock::now();
    auto friendlyTime = std::chrono::duration<double, std::milli>(friendEnd-friendStart).count();

    std::sort(v.begin(), v.end());
    sum = 0;
    auto unfriendStart = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < N; i++){
        sum += v[i];
    }
    auto unfriendEnd = std::chrono::high_resolution_clock::now();
    auto unfriendTime = std::chrono::duration<double, std::milli>(unfriendEnd-unfriendStart).count();

        std::cout << "Cache friendly time: " << friendlyTime
              << "\nCache unfriendly time: " << unfriendTime
              << "\n";
}

int main() {
    cache();

    return 0;
}