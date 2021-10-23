#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <CL/sycl.hpp>

#define SEED 1234
#define min(x,y) (((x) < (y)) ? (x) : (y))

#if defined DOUBLE
typedef double PREC;
#else
typedef float PREC;
#endif

using namespace cl::sycl; 

template <typename T> inline T random_number(); 
template <typename T> void random_matrix(T *matrix, int m, int n); 
template <typename T> void zero_matrix(T *matrix, int m, int n); 
template <typename T> void print_matrix(T *matrix, int m, int n, std::string name); 

int main(int argc, char *argv[]) {
    int m, n, p; 

    // device selection
    sycl::device device(sycl::default_selector{}); 
    sycl::queue  queue(device); 

    // matrix size
    if (argc != 4) {
        m = 4; n = 4; p = 4;
    } else {
        m = atoi(argv[1]); n = atoi(argv[2]); p = atoi(argv[3]);
    }

    // USM 
    PREC *A_USM = sycl::malloc_shared<PREC>(m * p, queue);
    PREC *B_USM = sycl::malloc_shared<PREC>(p * n, queue);
    PREC *C_USM = sycl::malloc_shared<PREC>(m * n, queue);

    // seed
    srand(SEED); 

    // initialization 
    random_matrix<PREC>(A_USM, m, p); 
    random_matrix<PREC>(B_USM, p, n); 
    zero_matrix<PREC>(C_USM, m, n); 

    // start timing
    auto start = std::chrono::high_resolution_clock::now(); 

    // matmul
    queue.submit([&](auto &handle) {  
            handle.parallel_for(range(m, n), [=](auto index) { 
                    int i = index[0]; 
                    int j = index[1]; 

                    for (int k=0; k<p; k++)
                    C_USM[i*n+j] += A_USM[i*p+k] * B_USM[k*n+j];
                    }); 
            });  
    queue.wait(); 

    // stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // // walltime 
    std::chrono::duration<PREC> walltime = end-start; 

    // // gflops 
    PREC gflops = (2.0*m*p*n - 1.0*m*n)*1E-9/walltime.count(); 

    std::cout << "Performance:" << gflops << " (GFlops)" << std::endl;  

    // debug
    print_matrix<PREC>(A_USM, m, p, "A =");
    print_matrix<PREC>(B_USM, p, n, "B =");
    print_matrix<PREC>(C_USM, m, n, "C =");

    // free memory 
    sycl::free(A_USM, queue);
    sycl::free(B_USM, queue);
    sycl::free(C_USM, queue);

    return 0;
}

// generate random number between [0,1)
template <typename T> inline T random_number() { 
    return (T)rand() / (T)(RAND_MAX);
}

// generate random matrix
template <typename T> void random_matrix(T *matrix, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*n+j] = random_number<T>();
}

// zero matrix
template <typename T> void zero_matrix(T *matrix, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*n+j] = (T) 0.0;
}

// print matrix for debug
template <typename T> void print_matrix(T *matrix, int m, int n, std::string name) {
    std::cout << name << std::endl;

    for (int i=0; i<min(m,4); i++) {
        for (int j=0; j<min(n,4); j++) {
            std::cout << matrix[i*n+j] << " ";
        }
        std::cout << std::endl;
    }
}
