
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include "utils.hpp"
using namespace std;

#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define FULL_MASK 0xffffffff

#define CUBLAS_SAFE_CALL(func)                                                                  \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t e = (func);                                                              \
        if (e != CUBLAS_STATUS_SUCCESS)                                                         \
        {                                                                                       \
            std::stringstream safe_call_ss;                                                     \
            safe_call_ss << "\nerror: " #func " failed with error"                              \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e; \
            throw std::runtime_error(safe_call_ss.str());                                       \
        }                                                                                       \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                         \
    do                                                                                            \
    {                                                                                             \
        cudaError_t result = (x);                                                                 \
        if (result != cudaSuccess)                                                                \
        {                                                                                         \
            const char *msg = cudaGetErrorString(result);                                         \
            std::stringstream safe_call_ss;                                                       \
            safe_call_ss << "\nerror: " #x " failed with error"                                   \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
            throw std::runtime_error(safe_call_ss.str());                                         \
        }                                                                                         \
    } while (0)

__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}

__device__ __forceinline__ const int* add_ptr_u(const int* src, int offset)      \
{                                                                            \
    const int* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ const float* add_ptr_f(const float* src, int offset)      \
{                                                                            \
    const float* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}
__global__ void convert_bcsr_kernel_fine_1(int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.x; // row
    uint tid = threadIdx.x;
    int sum =0;
    int4 flag;
    for(int _pos = tid; _pos<w/4; _pos +=blockDim.x){
        flag = FETCH_INT4(mask[by * w + _pos*4]);
        sum += flag.x + flag.y + flag.z + flag.w;
    }
    reduce[tid] = sum;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    if(tid==0){
        extra_buffer[by] = reduce[0];
        extra_buffer[by+h] = reduce[0];
        atomicAdd(&row[h], reduce[0]);
    }

}
__global__ void convert_bcsr_kernel_fine_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, float * values, int * extra_buffer)
{
    uint tid = threadIdx.x;
    uint by = blockIdx.x;
    __shared__ int prefix_sum;
    if (tid==0){
        prefix_sum = 0;
        for(int i=0; i<by; i++)
            prefix_sum += extra_buffer[i];
        row[by] = prefix_sum;
    }
    __syncthreads();
    for(int _pos=tid; _pos<w; _pos+=blockDim.x){
        if(mask[by*w+_pos]>0){
            int tmp = atomicSub(&extra_buffer[by+h], 1);
            tmp-=1;
            col[prefix_sum+tmp] = _pos;
            values[prefix_sum+tmp] = dense[by*w+_pos];
        }
    }
}
void convert_csr_fine(int * mask, float* dense, int h, int w,
                        int * row, int * col,
                        float * values, int * extra_buffer)
{
    const int block_h =1, block_w=1;
    // assert(block_w==1);
    // assert(block_h==1);
    // CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    // CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
    dim3 block_dim(512);
    dim3 grid_dim(h/block_h);
    convert_bcsr_kernel_fine_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
    convert_bcsr_kernel_fine_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, values, extra_buffer);
}
template<
    const int H,
    const int W,
    const int BLOCKDIM
>
__global__ void convert_bcsr_kernel_fine_2_template( int * __restrict__  mask, float * __restrict__  dense, int * row, int *col, float * values, int * extra_buffer)
{
    uint tid = threadIdx.x;
    uint wid = tid/32;
    uint wtid = tid%32;
    uint by = blockIdx.x;
    const int row_stride = BLOCKDIM / 32;
    const int rid = wid + by * row_stride; // row id
    assert(blockDim.x==BLOCKDIM);
    int prefix_sum;
    if (wtid%32==0){
        prefix_sum = 0;
        #pragma unroll
        for(int i=0; i<rid; i++)
            prefix_sum += extra_buffer[i];
        row[rid] = prefix_sum;
    }
    prefix_sum = __shfl_sync(FULL_MASK, prefix_sum, 0);
    // __syncthreads();
    #pragma unroll
    for(int round=0; round<W/32/4; round++){
        int _pos_start = round*32*4+wtid*4;
        int4 mask4 = FETCH_INT4(mask[rid*W+_pos_start]);
        int * p_mask = (int*)&mask4;
        #pragma unroll 
        for(int i=0; i<4; i++){
            if(p_mask[i]>0){
                int tmp = atomicSub(&extra_buffer[rid+H], 1);
                tmp-=1;
                col[prefix_sum+tmp] = _pos_start+i;
                values[prefix_sum+tmp] = dense[rid*W+_pos_start+i];
            }
        }
    }

}

void convert_csr_fine_template(int * mask, float* dense, int h, int w,
                        int * row, int * col,
                        float * values, int * extra_buffer)
{
    if(h==4096 && w==4096){
        const int threads = 512;
        const int block_h = threads/32;
        dim3 block_dim(512);
        dim3 grid_dim_1(h);
        dim3 grid_dim_2(h/block_h);
        convert_bcsr_kernel_fine_1<<<grid_dim_1, block_dim>>>(mask, dense, h, w, 1, 1, row, col, values, extra_buffer);
        convert_bcsr_kernel_fine_2_template<4096, 4096, 512><<<grid_dim_2, block_dim>>>(mask, dense, row, col, values, extra_buffer);
    }
}
void convert_csr(int * mask, float * dense, int h, int w,
    int * row, int *col, float*values, int * extra_buffer)
{
    
    // convert_csr_fine(mask, dense, h, w, row, col, values, extra_buffer);
    convert_csr_fine_template(mask, dense, h, w, row, col, values, extra_buffer);
}



int main(int argc, char *argv[]){
    float sparsity_ratio = atof(argv[1]);
    //printf("Sparsity Ratio=%f\n", sparsity_ratio);
    // Calculate the matA(Activation: Shape=mxk) * matB(Weight:Shape=k*n)
    // Specify the random seed here
    srand(1);
    int32_t * row_idx, *col_idx, *d_row, *d_col;
    int nnz;
    float * values, *d_val;
    int * maskA, * d_maskA;
    float * matA, * d_matA, *dBuffer;
    int * ext_buffer, *mask;
    const int m = atoi(argv[2]);
    const int k = atoi(argv[3]);
    
    //int m=1024, k=1024, n=1024;
    float alpha=1.0, beta=0.0;
    float sparsity = sparsity_ratio;

    matA = (float*) malloc(sizeof(float)*m*k);
    maskA = (int *) malloc(sizeof(int)*m*k);
    // init(matA, m*k, sparsity_ratio);
    init_mask_blockwise(maskA, matA, m, k, 1, 1, sparsity_ratio);
    CUDA_SAFE_CALL(cudaMalloc(&d_matA, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_maskA, sizeof(int)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&ext_buffer, sizeof(int)*m*k));
    CUDA_SAFE_CALL(cudaMemcpy(d_matA, matA, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_maskA, maskA, sizeof(int)*m*k, cudaMemcpyHostToDevice));
   
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 3000;


    CUDA_SAFE_CALL(cudaEventRecord(start));

    for(int i = 0; i < nIter; i += 1){
        convert_csr(d_maskA, d_matA, m, k, d_row, d_col, d_val, ext_buffer);
    }

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    //printf("Time= %f msec\n", msecPerMatrixMul);
    printf("%d,%d,1,1,%.2f,%.2f\n",m, k, sparsity_ratio, msecPerMatrixMul);
    return 0;
}
