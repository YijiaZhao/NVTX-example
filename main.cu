#include <math.h>
#include <chrono>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "nvToolsExt.h"

using namespace std;

#define BLOCK_SIZE 32

__host__ void print_info() {

	int num;
	cudaDeviceProp prop;
	cudaError_t cudaStatus = cudaGetDeviceCount(&num);
	printf("deviceCount := %d\n",num);
	for(int i=0;i<num;i++){
		cudaGetDeviceProperties(&prop,i);
		printf("================================\nname:%s\n",prop.name);
		// printf("totalGlobalMem:%zu\n",prop.totalGlobalMem);
		// printf("totalGlobalMem:%zu\n",prop.totalGlobalMem/1024);
		// printf("totalGlobalMem:%zu\n",prop.totalGlobalMem/1024/1024);
		printf("totalGlobalMem:%zu GB\n",prop.totalGlobalMem/1024/1024/1024);
		printf("multiProcessorCount:%d\n",prop.multiProcessorCount);
		printf("maxThreadsPerBlock:%d\n",prop.maxThreadsPerBlock);

		printf("l2CacheSize:%d\n",prop.l2CacheSize);
		printf("warpSize:%d\n",prop.warpSize);
		printf("sharedMemPerBlock:%zu\n",prop.sharedMemPerBlock);
		printf("sharedMemPerMultiprocessor:%zu\n",prop.sharedMemPerMultiprocessor);

        
		printf("maxGridSize:%d %d %d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		printf("major:%d,minor:%d\n================================\n",prop.major,prop.minor);
	}

}

template <typename T>
void matmul_cpu(T *h_a, T *h_b, T *h_result, int m, int n, int k) {
/*
  h_a: 
  row
  for 1:m
    for 1：n

  h_b:
  col
  for 1:k
    for 1：n

  h_r
  for 1:m
    for 1:k

  for 1:m
    for 1:k
      for 1:n
*/

    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            T tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

//one thread one result
template <typename T>
__global__ void matmul_gpu_1(T *a, T *b, T *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < k){
      T sum = 0;
      for(int i = 0; i < n; i++) 
      {
          sum += a[row * n + i] * b[i * k + col];
      }
      c[row * k + col] = sum;
    }
} 

//one thread one part of result
template <typename T>
__global__ void matmul_gpu_2(T *a, T *b, T *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < k){
      T sum = 0;
      for(int i = blockIdx.z*(n/gridDim.z); i < (blockIdx.z + 1) * n/gridDim.z; i++) 
      {
          sum += a[row * n + i] * b[i * k + col];
      }
      atomicAdd(c + row * k + col, sum);
    }
} 



//use share mem as cache
template <typename T>
__global__ void matmul_gpu_3(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // a: 
    // row
    // i_tile * 32 + threadIdx.x

    // b:col
    // (j_tile * 32 + threadIdx.x)
    // col
    // if(row<m && col<k){
    int res = 0;    
    // if(row == 0 && col == 0) printf("i_tile: %d %d %d\n", n, BLOCK_SIZE, gridDim.z);

for(int i_tile=0; i_tile < n/BLOCK_SIZE/gridDim.z; i_tile++){

        tile_a[threadIdx.y][threadIdx.x] = a[row * n + i_tile * BLOCK_SIZE * gridDim.z + blockIdx.z * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
        tile_b[threadIdx.y][threadIdx.x] = b[(i_tile * BLOCK_SIZE * gridDim.z + blockIdx.z * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict

    __syncthreads();

    for(int i=0; i<BLOCK_SIZE; i++) {
        // a one row , b one col
        res += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

}
    atomicAdd(c + row * k + col, res);

}

//use share mem as cache
template <typename T>
__global__ void matmul_gpu_3_(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

int res = 0;    
for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
    int _id_a = row * n + i_tile * BLOCK_SIZE + threadIdx.x;
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y][threadIdx.x] = a[_id_a]; // avoid bank confict
    }    

    int _id_b = (i_tile * BLOCK_SIZE + threadIdx.y) * k + col;
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x] = b[_id_b];  // avoid bank confict
    }

    __syncthreads();

    for(int i=0; i<BLOCK_SIZE; i++) {
        // a one row , b one col
        res += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

    __syncthreads();
}
c[row * k + col] = res;
}

//use share mem as cache
template <typename T>
__global__ void matmul_gpu_44(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

T res[3] = {0};


for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }    
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x + 16) < n) {
        tile_a[threadIdx.y][threadIdx.x+16] = a[row * n + i_tile * BLOCK_SIZE + threadIdx.x + 16]; // avoid bank confict
    }
    if ( row + 16 < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y+16][threadIdx.x] = a[(row + 16) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row + 16 < m && (i_tile * BLOCK_SIZE + threadIdx.x + 16) < n) {
        tile_a[threadIdx.y+16][threadIdx.x+16] = a[(row + 16) * n + i_tile * BLOCK_SIZE + threadIdx.x + 16]; // avoid bank confict
    }


    if (col < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
        // tile_b[threadIdx.y+BLOCK_SIZE][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
    }
    if (col + 16 < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x+16] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col + 16];  // avoid bank confict
    }
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y + 16) < n) {
        tile_b[threadIdx.y+16][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 16) * k + col];  // avoid bank confict
    }
    if ( col + 16 < k && (i_tile * BLOCK_SIZE + threadIdx.y + 16) < n) {
        tile_b[threadIdx.y+16][threadIdx.x+16] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 16) * k + col + 16] ;  // avoid bank confict
    }

    __syncthreads();
    for(int i=0; i<BLOCK_SIZE; i++) {
        // a one row , b one col
        res[0] += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        res[1] += tile_a[threadIdx.y + 16][i] * tile_b[i][threadIdx.x];
        res[2] += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x + 16];
        res[3] += tile_a[threadIdx.y + 16][i] * tile_b[i][threadIdx.x + 16];
    }
    __syncthreads();
}

c[row * k + col] = res[0];
c[(row + 16)* k + col] = res[1];
c[row * k + col + 16] = res[2];
c[(row + 16)* k + col + 16] = res[3];

}


//use share mem as cache
template <typename T>
__global__ void matmul_gpu__4(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x * 2 + threadIdx.x;

T res[4] = {0};


for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x + 16) < n) {
        tile_a[threadIdx.y][threadIdx.x+16] = a[row * n + i_tile * BLOCK_SIZE + threadIdx.x + 16]; // avoid bank confict
    }
    if ( row + 16 < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y+16][threadIdx.x] = a[(row + 16) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row + 16 < m && (i_tile * BLOCK_SIZE + threadIdx.x + 16) < n) {
        tile_a[threadIdx.y+16][threadIdx.x+16] = a[(row + 16) * n + i_tile * BLOCK_SIZE + threadIdx.x + 16]; // avoid bank confict
    }


    if (col < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
        // tile_b[threadIdx.y+BLOCK_SIZE][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
    }
    if (col + 16 < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x+16] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col + 16];  // avoid bank confict
    }
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y + 16) < n) {
        tile_b[threadIdx.y+16][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 16) * k + col];  // avoid bank confict
    }
    if ( col + 16 < k && (i_tile * BLOCK_SIZE + threadIdx.y + 16) < n) {
        tile_b[threadIdx.y+16][threadIdx.x+16] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 16) * k + col + 16] ;  // avoid bank confict
    }

    __syncthreads();
    for(int i=0; i<BLOCK_SIZE; i++) {
        // a one row , b one col
        res[0] += tile_a[threadIdx.y][i]      * tile_b[i][threadIdx.x];
        res[1] += tile_a[threadIdx.y + 16][i] * tile_b[i][threadIdx.x];
        res[2] += tile_a[threadIdx.y][i]      * tile_b[i][threadIdx.x + 16];
        res[3] += tile_a[threadIdx.y + 16][i] * tile_b[i][threadIdx.x + 16];
    }
    __syncthreads();
}

c[row * k + col] = res[0];
c[(row + 16)* k + col] = res[1];
c[row * k + col + 16] = res[2];
c[(row + 16)* k + col + 16] = res[3];

}

//use share mem as cache
template <typename T>
__global__ void matmul_gpu_4(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

T res[4] = {0};


for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
    if ( row < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row + 8 < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y+8][threadIdx.x] = a[(row + 8) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row + 16 < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y+16][threadIdx.x] = a[(row + 16) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }
    if ( row + 24 < m && (i_tile * BLOCK_SIZE + threadIdx.x) < n) {
        tile_a[threadIdx.y+24][threadIdx.x] = a[(row + 24) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
    }


    if (col < k && (i_tile * BLOCK_SIZE + threadIdx.y) < n) {
        tile_b[threadIdx.y][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
        // tile_b[threadIdx.y+BLOCK_SIZE][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y) * k + col];  // avoid bank confict
    }
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y + 8) < n) {
        tile_b[threadIdx.y+8][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 8) * k + col];  // avoid bank confict
    }
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y + 16) < n) {
        tile_b[threadIdx.y+16][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 16) * k + col];  // avoid bank confict
    }
    if ( col < k && (i_tile * BLOCK_SIZE + threadIdx.y + 24) < n) {
        tile_b[threadIdx.y+24][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + 24) * k + col];  // avoid bank confict
    }

    __syncthreads();
    for(int i=0; i<BLOCK_SIZE; i++) {
        // a one row , b one col
        res[0] += tile_a[threadIdx.y][i]      * tile_b[i][threadIdx.x];
        res[1] += tile_a[threadIdx.y + 8][i]  * tile_b[i][threadIdx.x];
        res[2] += tile_a[threadIdx.y + 16][i] * tile_b[i][threadIdx.x];
        res[3] += tile_a[threadIdx.y + 24][i] * tile_b[i][threadIdx.x];
    }
    __syncthreads();
}

c[row * k + col] = res[0];
c[(row + 8)* k + col] = res[1];
c[(row + 16)* k + col] = res[2];
c[(row + 24)* k + col] = res[3];

}

//use share mem as cache
template <typename T>
__global__ void matmul_gpu_4_(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 8 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T res[8] = {0};
    int interval = BLOCK_SIZE/8;

    for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
        for(int j=0; j<BLOCK_SIZE; j+=interval){
            tile_a[threadIdx.y+j][threadIdx.x] = a[(row + j) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
            tile_b[threadIdx.y+j][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + j) * k + col];  // avoid bank confict
        }
        __syncthreads();
        for(int i=0; i<BLOCK_SIZE; i++) {
            // a one row , b one col
            int tmp = tile_b[i][threadIdx.x];
            for(int j=0; j<BLOCK_SIZE; j+=interval){
                res[j/interval] += tile_a[threadIdx.y + j][i] * tmp;
            }
        }
        __syncthreads();
    }

    for(int j=0; j<BLOCK_SIZE; j+=interval){
        c[(row + j)* k + col] = res[j/interval];
    }
}


template <typename T>
__global__ void matmul_gpu_4_4(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T res[4] = {0};
    int interval = BLOCK_SIZE/4;

    for(int i_tile=0; i_tile < n/BLOCK_SIZE; i_tile++){
        for(int j=0; j<BLOCK_SIZE; j+=interval){
            tile_a[threadIdx.y+j][threadIdx.x] = a[(row + j) * n + i_tile * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
            tile_b[threadIdx.y+j][threadIdx.x] = b[(i_tile * BLOCK_SIZE + threadIdx.y + j) * k + col];  // avoid bank confict
        }
        __syncthreads();
        for(int i=0; i<BLOCK_SIZE; i++) {
            // a one row , b one col
            int tmp = tile_b[i][threadIdx.x];
            for(int j=0; j<BLOCK_SIZE; j+=interval){
                res[j/interval] += tile_a[threadIdx.y + j][i] * tmp;
            }
        }
        __syncthreads();
    }

    for(int j=0; j<BLOCK_SIZE; j+=interval){
        c[(row + j)* k + col] = res[j/interval];
    }
}


//use share mem as cache
template <typename T>
__global__ void matmul_gpu_5_16(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 16 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T res[16] = {0};
    int interval = BLOCK_SIZE/16;

    for(int j=0; j<BLOCK_SIZE; j+=interval){
        tile_a[threadIdx.y+j][threadIdx.x] = a[(row + j) * n + blockIdx.z * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
        tile_b[threadIdx.y+j][threadIdx.x] = b[(blockIdx.z * BLOCK_SIZE + threadIdx.y + j) * k + col];  // avoid bank confict
    }
    __syncthreads();
    for(int i=0; i<BLOCK_SIZE; i++) {
        int tmp = tile_b[i][threadIdx.x];
        for(int j=0; j<BLOCK_SIZE; j+=interval){
            res[j/interval] += tile_a[threadIdx.y + j][i] * tmp;
        }
    }
    for(int j=0; j<BLOCK_SIZE; j+=interval){
        atomicAdd(c + (row + j) * k + col, res[j/interval]);
    }
}

//use share mem as cache
template <typename T>
__global__ void matmul_gpu_5_8(T *a, T *b, T *c, int m, int n, int k)
{ 
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // res' row and col
    int row = blockIdx.y * blockDim.y * 8 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T res[8] = {0};
    int interval = BLOCK_SIZE/8;

for(int i_tile=0; i_tile < n/BLOCK_SIZE/gridDim.z; i_tile++){

    for(int j=0; j<BLOCK_SIZE; j+=interval){
        tile_a[threadIdx.y+j][threadIdx.x] = a[(row + j) * n + i_tile * gridDim.z * BLOCK_SIZE + blockIdx.z * BLOCK_SIZE + threadIdx.x]; // avoid bank confict
        tile_b[threadIdx.y+j][threadIdx.x] = b[(i_tile * gridDim.z * BLOCK_SIZE + threadIdx.y + j) * k + col];  // avoid bank confict
    }
    __syncthreads();
    for(int i=0; i<BLOCK_SIZE; i++) {
        int tmp = tile_b[i][threadIdx.x];
        for(int j=0; j<BLOCK_SIZE; j+=interval){
            res[j/interval] += tile_a[threadIdx.y + j][i] * tmp;
        }
    }
    // __syncthreads();
}

for(int j=0; j<BLOCK_SIZE; j+=interval){
    atomicAdd(c + (row + j) * k + col, res[j/interval]);
}

}

template <typename T>
bool varify(T *cpu_r, T *gpu_r, int m, int k, T diff)
{ 
    // for(int i=0; i<m*k; i++) {
    //     if(i > 255 && i < 1024) {
    //         printf("i %d, cpu: %f\tgpu: %f\n", i, cpu_r[i], gpu_r[i]);
    //     }
    // }

    for(int i=0; i<m*k; i++) {
        if(abs(abs((float)cpu_r[i]) - abs((float)gpu_r[i])) > diff) {
            printf("index: %d error: %f vs %f\n", i, (float)cpu_r[i], (float)gpu_r[i]);
            return false;
        }
    }
    printf("check good!\n");
    return true;
}


int main(void)
{
    //print_info();

    // int m=128;
    // int n=128;
    // int k=128;

    // int m=128;
    // int n=256;
    // int k=128;

    // int m=256;
    // int n=256;
    // int k=256;

    // int m=256;
    // int n=512;
    // int k=256;

    // int m=512;
    // int n=512;
    // int k=512;

    // int m=512;
    // int n=1024;
    // int k=512;

    // int m=1024;
    // int n=1024;
    // int k=1024;

    // int m=1024;
    // int n=2048;
    // int k=1024;
    int m,n,k;
    int m_data[7] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int n_data[14] = {128, 256, 256, 512, 512, 1024, 1024, 2048, 2048, 4096, 4096, 8192, 8192, 16384};

    for(int i=0; i<8; i++) {
        m = k = m_data[i/2];
        n = n_data[i];

    printf("=====m: %d, n: %d, k: %d====\n", m, n, k);    

    float *h_a, *h_b, *h_c, *h_c_gpu, *h_c_cub;
    cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_c_gpu, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_c_cub, sizeof(float)*m*k);


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // h_a[i * n + j] = rand() % 5;
            h_a[i * n + j] = 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            // h_b[i * k + j] = rand() % 5;
            h_b[i * k + j] = 1;
        }
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*m*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*k);
    cudaMalloc((void **) &d_c, sizeof(float)*m*k);


    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // matmul_cpu<float>(h_a, h_b, h_c, m, n, k);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // printf("cpu time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    cublasHandle_t handle;
    cublasCreate_v2(&handle);    
    float alpha = 1.0f, beta = 0.0f;
    // cudaEventCreate(&begin2);
    begin = std::chrono::steady_clock::now();
    cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n,
                 &alpha, d_b, CUDA_R_32F, m, d_a, CUDA_R_32F, n, &beta, d_c,
                 CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaMemcpy(h_c_cub, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    end = std::chrono::steady_clock::now();
    printf("cublas time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // varify<float>(h_c, h_c_cub, m, k, 1e-3);


    begin = std::chrono::steady_clock::now();
    cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n,
                 &alpha, d_b, CUDA_R_32F, m, d_a, CUDA_R_32F, n, &beta, d_c,
                 CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    cudaMemcpy(h_c_cub, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    end = std::chrono::steady_clock::now();
    printf("cublas time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    // varify<float>(h_c, h_c_cub, m, k, 1e-3);




    // cudaEvent_t begin1, end1;
    // cudaEventCreate(&begin1);
    // cudaEventCreate(&end1);

    // float elapsed;
    dim3 dimGrid(m/BLOCK_SIZE, k/BLOCK_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // cudaEventRecord(begin1);
    begin = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    matmul_gpu_1<float><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_1 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_1 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);
    

    dim3 dimGrid2(m/BLOCK_SIZE, k/BLOCK_SIZE, n/BLOCK_SIZE);
    cudaMemset(d_c, 0, m*k*4);
    // cudaEventRecord(begin1);
    begin = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    matmul_gpu_2<float><<<dimGrid2, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_2 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_2 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);


    cudaMemset(d_c, 0, m*k*4);
    // cudaEventRecord(begin1);
    dim3 dimGrid2_(m/BLOCK_SIZE, k/BLOCK_SIZE, n/BLOCK_SIZE/2);
    begin = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    matmul_gpu_3<float><<<dimGrid2, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_3 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);


    begin = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    nvtxRangePushA("matmul_gpu_3_");
    matmul_gpu_3_<float><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    nvtxRangePop();
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_3_ time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);


    begin = std::chrono::steady_clock::now();
    cudaMemset(d_c, 0, m*k*4);
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE/8, 1);
    nvtxRangePushA("matmul_gpu_4_");
    matmul_gpu_4_<float><<<dimGrid, dimBlock2>>>(d_a, d_b, d_c, m, n, k);
    nvtxRangePop();
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_4_ time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);

    
    begin = std::chrono::steady_clock::now();
    dim3 dimGrid__(m/BLOCK_SIZE, k/BLOCK_SIZE, n/BLOCK_SIZE/4);
    cudaMemset(d_c, 0, m*k*4);
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    dim3 dimBlock3(BLOCK_SIZE, BLOCK_SIZE/8, 1);
    nvtxRangePushA("matmul_gpu_5_8");
    matmul_gpu_5_8<float><<<dimGrid__, dimBlock3>>>(d_a, d_b, d_c, m, n, k);
    nvtxRangePop();
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_5_8 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);


    begin = std::chrono::steady_clock::now();
    cudaMemset(d_c, 0, m*k*4);
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    dim3 dimBlock_new(BLOCK_SIZE, BLOCK_SIZE/4, 1);
    nvtxRangePushA("matmul_gpu_4_4");
    matmul_gpu_4_4<float><<<dimGrid, dimBlock_new>>>(d_a, d_b, d_c, m, n, k);
    nvtxRangePop();
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_4_4 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);


    begin = std::chrono::steady_clock::now();
    cudaMemset(d_c, 0, m*k*4);
    cudaMemcpyAsync(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    dim3 dimBlock4(BLOCK_SIZE, BLOCK_SIZE/16, 1);
    nvtxRangePushA("matmul_gpu_5_16");
    matmul_gpu_5_16<float><<<dimGrid2, dimBlock4>>>(d_a, d_b, d_c, m, n, k);
    nvtxRangePop();
    cudaMemcpyAsync(h_c_gpu, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    printf("matmul_gpu_5_16 time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // cudaEventRecord(end1);
    // cudaEventSynchronize(end1);
    // cudaEventElapsedTime(&elapsed, begin1, end1);
    // printf("matmul_gpu_3 time: %f ms\n", elapsed);
    varify<float>(h_c_cub, h_c_gpu, m, k, 1e-3);



    // cudaEvent_t begin2, end2;
    // cudaEventCreate(&begin2);
    // cudaEventCreate(&end2);

   
    printf("=================================================\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);    
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_c_gpu);
    cudaFreeHost(h_c_cub);
    // break;
    }


    return 0;
}
