#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024

//Device function for recursive Merge
__device__ void Merge(int* arr, int* temp, int left, int middle, int right) 
{
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

//GPU Kernel for Merge Sort
__global__ void MergeSortGPU(int* arr, int* temp, int n, int width) 
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = left + width / 2;
    int right = left + width;

    if (left < n && middle < n) 
    {
        Merge(arr, temp, left, middle, right);
    }
}

//CPU Merge Recursive Call function
void merge(int* arr, int* temp, int left, int mid, int right) 
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}

//CPU Implementation of Merge Sort
void mergeSortCPU(int* arr, int* temp, int left, int right) 
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    merge(arr, temp, left, mid, right);
}

// 共享内存归并排序 - 每个块内进行插入排序
__global__ void sharedMemoryMergeSort(int* arr, int n) 
{
    extern __shared__ int sharedArr[]; // 共享内存声明
    int local_tid = threadIdx.x; // 计算块内索引
    int block_start = blockIdx.x * blockDim.x; // 当前块的起始索引

    // 1. 加载数据到共享内存
    if (block_start + local_tid < n) 
    {
        sharedArr[local_tid] = arr[block_start + local_tid];
    }
    __syncthreads(); // 确保所有数据加载完成

    // 2. 共享内存内部执行插入排序
    for (int i = 1; i < blockDim.x && block_start + i < n; i++) 
    {
        int key = sharedArr[i];
        int j = i - 1;
        while (j >= 0 && sharedArr[j] > key) 
        {
            sharedArr[j + 1] = sharedArr[j];
            j--;
        }
        sharedArr[j + 1] = key;
    }
    __syncthreads(); // 确保排序完成

    // 3. 写回全局内存
    if (block_start + local_tid < n) 
    {
        arr[block_start + local_tid] = sharedArr[local_tid];
    }
}

//Function to print array
void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

//Automated function to check if array is sorted
bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

//Function to check if given number is a power of 2
bool isPowerOfTwo(int num) 
{
    return num > 0 && (num & (num - 1)) == 0;
}


//MAIN PROGRAM
int main()
{   
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "MERGE SORT IMPLEMENTATION" << std::endl;
    std::cout << "A Performance Comparison of These 4 Sorts in CPU vs GPU vs sharedMemory vs ？？？ " << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    int size;
    std::cout << "\n\nEnter the size of the array. Must be a power of 2:\n ";
    std::cin>>size;

    while (!isPowerOfTwo(size))
    {
        if (!isPowerOfTwo(size))
        {
            std::cout << "\nWrong Size, must be power of 2. Try again:\n ";
            std::cin>>size;
        }
        else
            break;
    }
    
    //Create CPU based Arrays
    int* arr = new int[size];
    int* arr2 = new int[size];
    //int* arr3 = new int[size];？？？
    int* carr = new int[size];
    int* temp = new int[size];

    //Create GPU based arrays
    int* gpuArrmerge;
    int* gpuArrmerge2;
    //int* gpuArrmerge3; ？？？
    int* gpuTemp;

    // Initialize the array with random values
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size; ++i) 
    {
        arr[i] = rand() % 100;
        carr[i] = arr[i];
    }

    //Print unsorted array 
    std::cout << "\n\nUnsorted array: ";
    if (size <= 100) 
    {
        printArray(arr, size);
    }
    else 
    {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&gpuArrmerge, size * sizeof(int));
    cudaMalloc((void**)&gpuTemp, size * sizeof(int));
    cudaMalloc((void**)&gpuArrmerge2, size * sizeof(int));
    //cudaMalloc((void**)&gpuArrmerge3, size * sizeof(int)); ？？？

    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuArrmerge2, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(gpuArrmerge3, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;
    
    // Perform sharedMemory merge sort and measure time
    cudaEvent_t startGPU2, stopGPU2;
    cudaEventCreate(&startGPU2);
    cudaEventCreate(&stopGPU2);
    float millisecondsGPU2 = 0;

    // // Perform sharedMemory merge sort and measure time？？？
    // cudaEvent_t startGPU3, stopGPU3;
    // cudaEventCreate(&startGPU3);
    // cudaEventCreate(&stopGPU3);
    // float millisecondsGPU3 = 0;

    //Initialize CPU clock counters
    clock_t startCPU, endCPU;








     //sharedMemory
    // 确定线程块大小
    int threadsPerBlock2 = 256; // 每个线程块的线程数
    int blocksPerGrid2 = (size + threadsPerBlock2 - 1) / threadsPerBlock2; // 计算线程块数量
     // 设置共享内存大小
    int sharedMemSize = sizeof(int) * threadsPerBlock2; 
    cudaEventRecord(startGPU2);
    // 调用 GPU 共享内存归并排序
    sharedMemoryMergeSort << < blocksPerGrid2, threadsPerBlock2, sharedMemSize >> > (gpuArrmerge2, size);
    cudaEventRecord(stopGPU2);

    //Transfer sorted array back to CPU
    cudaMemcpy(arr2, gpuArrmerge2, size * sizeof(int), cudaMemcpyDeviceToHost);

    //Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU2);
    cudaEventElapsedTime(&millisecondsGPU2, startGPU2, stopGPU2);


    //Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

     //Call GPU Merge Kernel and time the run
    cudaEventRecord(startGPU);
    for (int wid = 1; wid < size; wid *= 2)
    {
        MergeSortGPU << <threadsPerBlock, blocksPerGrid >> > (gpuArrmerge, gpuTemp, size, wid * 2);
    }
    cudaEventRecord(stopGPU);

    //Transfer sorted array back to CPU
    cudaMemcpy(arr, gpuArrmerge, size * sizeof(int), cudaMemcpyDeviceToHost);

    //Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);


    //Time the CPU and call CPU Merge Sort
    startCPU = clock();
    mergeSortCPU(carr, temp, 0, size - 1);
    endCPU = clock();
    //Calculate Elapsed CPU time
    double millisecondsCPU = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);

   







    //Display sorted CPU array
    std::cout << "\nSorted CPU array: ";
    if (size <= 100) 
    {
        printArray(carr, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Display sorted GPU array
    std::cout << "\n\nSorted GPU array: ";
    if (size <= 100) 
    {
        printArray(arr, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Display sorted sharedMemory array
    std::cout << "\n\nSorted sharedMemory array: ";
    if (size <= 100) 
    {
        printArray(arr2, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }
    
    //Run the array with the automated isSorted checker
    
   
    if (isSorted(carr, size))
        std::cout << "SORT CHECKER RUNNING - SUCCESFULLY SORTED CPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    if (isSorted(arr, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;
    
    if (isSorted(arr2, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    //Print the time of the runs
    std::cout << "\n\nCPU Time: " << millisecondsCPU << " ms" << std::endl;
    std::cout << "GPU Time: " << millisecondsGPU << " ms" << std::endl;
    std::cout << "sharedMemory Time: " << millisecondsGPU2 << " ms" << std::endl;

    //Destroy all variables
    delete[] carr;
    delete[] arr;
    delete[] arr2;
    delete[] temp;

    //End
    cudaFree(gpuArrmerge);
    cudaFree(gpuArrmerge2);
    cudaFree(gpuTemp);

    std::cout << "\n------------------------------------------------------------------------------------\n||||| END. YOU MAY RUN THIS AGAIN |||||\n------------------------------------------------------------------------------------";
    return 0;
}