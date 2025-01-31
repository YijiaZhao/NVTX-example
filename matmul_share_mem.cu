#include <math.h>
#include <chrono>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <string>
#include <cuda_fp16.h>

using namespace std;

#define BLOCK_SIZE 32

void kimi_sync() {
    cudaError_t addVectorsErr;
    cudaError_t asyncErr;
    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error1: %s\n", cudaGetErrorString(addVectorsErr));
  
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("sync Error1: %s\n", cudaGetErrorString(asyncErr));
}

__host__ void print_info() {

	int num;
	cudaDeviceProp prop;
	cudaError_t cudaStatus = cudaGetDeviceCount(&num);
	printf("deviceCount := %d\n",num);
	for(int i=0;i<num;i++){
		cudaGetDeviceProperties(&prop,i);
		printf("================================\nname:%s\n",prop.name);
		printf("totalGlobalMem:%zu GB\n",prop.totalGlobalMem/1024/1024/1024);
		printf("multiProcessorCount:%d\n",prop.multiProcessorCount);
		printf("maxThreadsPerBlock:%d\n",prop.maxThreadsPerBlock);
		printf("maxThreadsPerMultiProcessor:%d\n",prop.maxThreadsPerMultiProcessor);

		printf("l2CacheSize:%d\n",prop.l2CacheSize);
		printf("warpSize:%d\n",prop.warpSize);
		printf("sharedMemPerBlock:%zu\n",prop.sharedMemPerBlock);
		printf("sharedMemPerMultiprocessor:%zu\n",prop.sharedMemPerMultiprocessor);

        
		printf("maxGridSize:%d %d %d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		printf("major:%d,minor:%d\n================================\n",prop.major,prop.minor);
	}

}

template <typename T>
void matmul_cpu(const T *h_a, const T *h_b, T *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            T tmp = 0.0;
            for (int h = 0; h < k; ++h) 
            {
                tmp += h_a[i * k + h] * h_b[h * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

template <typename T>
__global__ void matmul_gpu(const T *a, const T *b, T *c, int m, int n, int k)
{ 
    int ele_size = m*n;
    int stride = blockDim.x*gridDim.x;
    int ele_id = blockDim.x*blockIdx.x + threadIdx.x;

    int num_stride = (ele_size-1)/stride + 1;

    for(int wave_id=0; wave_id<num_stride; wave_id++){
        int c_id = wave_id*stride + ele_id;
        if (c_id < ele_size){
            int row = c_id/n;
            int col = c_id%n;
            T sum = 0;
            for(int i = 0; i < k; i++) 
            {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[c_id] = sum;
        }
    }
}

template <typename T>
__global__ void matmul_gpu_1(const T *a, const T *b, T *c, int m, int n, int k)
{
    int stride_x_dim = gridDim.x*BLOCK_SIZE;
    int stride_y_dim = gridDim.y*BLOCK_SIZE;

    int stride_x = (n-1)/BLOCK_SIZE+1;
    int stride_y = (m-1)/BLOCK_SIZE+1;

    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    for(int stride_idx=0; stride_idx<stride_x; stride_idx++){
        for(int stride_idy=0; stride_idy<stride_y; stride_idy++){

            int row = stride_idy * stride_y_dim + blockIdx.y * BLOCK_SIZE + threadIdx.y; 
            int col = stride_idx * stride_x_dim + blockIdx.x * BLOCK_SIZE + threadIdx.x;
            T sum = 0.0;
            for(int i = 0; i < k; i++) 
            {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row*n+col] = sum;
        }
    }
}

template <typename T>
__global__ void cuda_printf_ele(const T *a, int index)
{ 
    printf("cuda print: %f\n", a[index]);
}

//one thread one result
template <typename T>
__global__ void matmul_gpu_2(const T *a, const T *b, T *c, int m, int n, int k)
{
    int stride_x_dim = gridDim.x*BLOCK_SIZE;
    int stride_y_dim = gridDim.y*BLOCK_SIZE;

    int stride_x = (n-1)/stride_x_dim+1;
    int stride_y = (m-1)/stride_y_dim+1;

    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    for(int stride_idx=0; stride_idx<stride_x; stride_idx++){
        for(int stride_idy=0; stride_idy<stride_y; stride_idy++){

            int row = stride_idy * stride_y_dim + blockIdx.y * BLOCK_SIZE + threadIdx.y; 
            int col = stride_idx * stride_x_dim + blockIdx.x * BLOCK_SIZE + threadIdx.x;
            T sum = 0.0;
            for(int tile_id=0; tile_id<=k/BLOCK_SIZE; tile_id++){
                if(row<m && tile_id*BLOCK_SIZE + threadIdx.x<k) {
                    tile_a[threadIdx.x][threadIdx.y] = a[row*k + tile_id*BLOCK_SIZE + threadIdx.x];
                } else {
                    tile_a[threadIdx.x][threadIdx.y] = 0.0;
                }

                if(col<n && tile_id*BLOCK_SIZE+threadIdx.y<k) {
                    tile_b[threadIdx.y][threadIdx.x] = b[(tile_id*BLOCK_SIZE + threadIdx.y)*n + col];
                } else {
                    tile_b[threadIdx.y][threadIdx.x] = 0.0;
                }
                __syncthreads();
                for(int i=0;i<BLOCK_SIZE;i++){
                    // if (i+tile_id*BLOCK_SIZE >=k) {
                    //     break;
                    // }
                    sum += tile_a[i][threadIdx.y] * tile_b[i][threadIdx.x];
                }
                __syncthreads();
            }
            if (row<m && col<n) {
                c[row*n+col] = sum;
            }
        }
    }
}

template <typename T>
bool varify(const T *cpu_r, const T *gpu_r, int m, int n, T diff)
{ 
    for(int i=0; i<m*n; i++) {
        if(abs(abs((float)cpu_r[i]) - abs((float)gpu_r[i])) > diff) {
            printf("index: %d error: ref: %f vs input: %f\n", i, (float)cpu_r[i], (float)gpu_r[i]);
            return false;
        }
    }
    printf("check good!\n");
    return true;
}

// template <typename T>
// __global__ void varify_gpu(const T *cpu_r, const T *gpu_r, int m, int n, T diff)
// { 
//     int ele_size = m*n;
//     int stride = blockDim.x*gridDim.x;
//     int ele_id = blockDim.x*blockIdx.x + threadIdx.x;

//     int num_stride = (ele_size-1)/stride + 1;

//     for(int wave_id=0; wave_id<num_stride; wave_id++){
//         int c_id = wave_id*stride + ele_id;
//         if (c_id < ele_size){
//             int row = c_id/n;
//             int col = c_id%n;
//             T sum = 0;
//             for(int i = 0; i < k; i++) 
//             {
//                 sum += a[row * k + i] * b[i * n + col];
//             }
//             c[c_id] = sum;
//         }
//     }
// }

int main(void)
{
    print_info();

    int m,n,k;
    // m = 20490;
    // n = 30430;
    // k = 1099;
    // m = 321;
    // n = 321;
    // k = 256;
    m = 2049;
    n = 3043;
    k = 1099;

    float *h_a, *h_b, *h_c, *h_c_gpu, *h_c_ref;
    cudaMallocHost(&h_a, sizeof(float)*m*k);
    cudaMallocHost(&h_b, sizeof(float)*k*n);
    cudaMallocHost(&h_c, sizeof(float)*m*n);
    cudaMallocHost(&h_c_gpu, sizeof(float)*m*n);
    cudaMallocHost(&h_c_ref, sizeof(float)*m*n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_a[i * k + j] = rand() % 5;
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_b[i * n + j] = rand() % 5;
        }
    }

    // matmul_cpu(h_a, h_b, h_c, m, n, k);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float)*m*k);
    cudaMalloc((void**)&d_b, sizeof(float)*k*n);
    cudaMalloc((void**)&d_c, sizeof(float)*m*n);

    cudaMemcpy(d_a, h_a, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    dim3 dimGrid(19, 6*6, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    string nvtx_name;
    for(int i=0; i<10; i++){
        string name1 = "share_mem_kernel";
        string name2 = std::to_string(i);
        nvtx_name = name1 + name2;
        nvtxRangePushA(nvtx_name.c_str());
        matmul_gpu_2<float><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
        // cudaDeviceSynchronize();
        kimi_sync();
        nvtxRangePop();
    }
    cudaMemcpy(h_c_gpu, d_c, sizeof(float)*m*n, cudaMemcpyDeviceToHost);


    dim3 dimGrid2(132*4, 1, 1);
    dim3 dimBlock2(1024, 1, 1);
    for(int i=0; i<10; i++){
        string name1 = "no_share_mem_kernel";
        string name2 = std::to_string(i);
        nvtx_name = name1 + name2;
        nvtxRangePushA(nvtx_name.c_str());
        matmul_gpu<float><<<dimGrid2, dimBlock2>>>(d_a, d_b, d_c, m, n, k);
        // cudaDeviceSynchronize();
        kimi_sync();
        nvtxRangePop();
    }
    cudaMemcpy(h_c_ref, d_c, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    varify<float>(h_c_ref, h_c_gpu, m, n, 1e-3);
       
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);    
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_c_gpu);
    cudaFreeHost(h_c_ref);

    return 0;
}
