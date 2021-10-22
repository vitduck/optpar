#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SEED 1234
#define min(x,y) (((x) < (y)) ? (x) : (y))

float random_number(); 

void random_matrix(float*, int, int); 
void zero_matrix(float*, int, int) ; 
void print_matrix(float*, int , int, const char*); 
void mat_mul(float*, float*, float*, int, int, int); 

int main(int argc, char **argv) { 
    int    m, n, p; 
    double elapsed_time, gflops; 
    struct timeval t1, t2; 

    // matrix size 
    if (argc != 4) {
        m = 4; n = 4; p = 4; 
    } else {  
        m = atoi(argv[1]); n = atoi(argv[2]); p = atoi(argv[3]); 
    } 

    // allocation
    float* A = (float*) malloc(sizeof(float) * m * p); 
    float* B = (float*) malloc(sizeof(float) * p * n); 
    float* C = (float*) malloc(sizeof(float) * m * n); 

    //initialize A, B 
    srand(SEED); 
    random_matrix(A, m, p); 
    random_matrix(B, p, n); 

    //initialize C
    zero_matrix(C, m, n); 

    // start timing 
    gettimeofday(&t1, NULL);

    // C = A * B 
    mat_mul(A, B, C, m, n, p); 

    // end timing 
    gettimeofday(&t2, NULL);

    // walltime 
    elapsed_time = (t2.tv_usec - t1.tv_usec)*1e-6 + (t2.tv_sec - t1.tv_sec);
    printf("Timing: %10.3f (s)\n", elapsed_time); 

    // gflops 
    gflops = (2.0*m*n*p - 1.0*m*p)*1E-9; 
    printf("Performance: %10.3f (GFlops)\n", gflops/elapsed_time);
   
    // debug
    print_matrix(A, m, p, "A ="); 
    print_matrix(B, p, n, "B ="); 
    print_matrix(C, m, n, "C ="); 
    
    //deallocate 
    free(A); 
    free(B); 
    free(C); 

    return 0;
}

void random_matrix(float *matrix, int m, int n) { 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*n + j] = random_number();
} 

void zero_matrix(float *matrix, int m, int n ) { 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*n + j] = 0.0; 
} 

void mat_mul(float* A, float* B, float* C, int m, int n, int p) { 
    int i, j, k; 
    for (i = 0; i < m; i++) 
        for (k = 0; k < p; k++) 
            # pragma nounroll
            # pragma omp simd reduction(+:C[i*n+j])
            for (j = 0; j < n; j++)  
                C[i*n+j] += A[i*p+k] * B[k*n+j]; 
} 

void print_matrix(float *matrix, int m , int n, const char *name ) { 
    printf("%s\n", name); 
    for (int i=0; i<min(m,4); i++) {
        for (int j=0; j<min(n,4); j++) {
            printf ("%12.5f", matrix[i*n+j]);
        }
        printf ("\n");
    }
} 

float random_number() { 
    return ((float)rand() / (float)RAND_MAX);   
} 
