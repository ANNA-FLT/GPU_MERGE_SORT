#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024

// Method 1 串行
// CPU Merge Recursive Call function
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

// CPU Implementation of Merge Sort
void mergeSortCPU(int* arr, int* temp, int left, int right) 
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    merge(arr, temp, left, mid, right);
}

// Method 2 并行
// Device function for recursive Merge
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

// GPU Kernel for Merge Sort
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

// Method 3 共享内存并行
// 共享内存归并排序 - 每个块内进行归并排序
__global__ void sharedMemoryMergeSort(int* arr, int* temp, int n) 
{
    extern __shared__ int sharedArr[]; // 共享内存声明
    int local_tid = threadIdx.x; // 计算块内索引
    int block_start = blockIdx.x * blockDim.x; // 当前块的起始索引

    // 1. 加载数据到共享内存
    if (block_start + local_tid < n) 
    {
        sharedArr[local_tid] = arr[block_start + local_tid];
    }else {
        sharedArr[local_tid] = INT_MAX; // 填充无效值
    }
    __syncthreads();
    
    // 2. 共享内存内部执行归并排序
    for (int width = 1; width < blockDim.x; width *= 2) {
        int left = local_tid * width * 2;
        int middle = left + width;
        int right = min(left + width * 2, blockDim.x);

        if (middle < blockDim.x) {
            Merge(sharedArr, temp, left, middle, right);
        }
        __syncthreads();
    }
    
    // 3. 写回全局内存
    if (block_start + local_tid < n) {
        arr[block_start + local_tid] = sharedArr[local_tid];
    }
}


// Method 4 co-rank归并
//输入k，数组A的长度m， 数组B的长度n
__device__ int co_rank(int k, const int *A, int m, const int *B, int n) { 
	int i = k <m ? k : m;	//初始化为k和m的较小值，确保初始猜测不会超出数组A的范围
	int j = k - i;	// corresponding j
	int i_low = 0 > (k-n) ? 0 : k-n;//lower bound on i，确保后续调整不会越界
	int j_low = 0 >(k-m) ? 0 : k-m;//lower bound on j
	int delta;	//缩小范围的步长
	//通过二分法逐步调整i和j，直到满足分割条件
	while(true){
		if(i > 0 && j < n && A[i-1] > B[j]){	//如果A[i-1] > B[j]，说明当前i过大，需减小i
		// first excluded B comes before last included A
			delta = ((i - i_low + 1)>>1);//即除2
			j_low=j;
			j = j + delta;
			i = i - delta;
		} else if(j > 0 && i < m && B[j-1] >= A[i]){	//如果B[j-1] >= A[i]，说明当前j过大，需增大i
			// first excluded A comes before last included B
			delta=((j - j_low + 1)>>1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		}else{
			break;
		}
	}
	return i;	//返回最终的i值，用于确定合并后的前k小元素的分割位置
}

__device__  
void merge(const int *A, int m, const int *B, int n, int *C) {  
    int i = 0; // Index into A  
    int j = 0; // Index into B  
    int k = 0; // Index into C  

    // merge the initial overlapping sections of A and B  
    while ((i < m) && (j < n)) {  
        if (A[i] <= B[j]) {  
            C[k++] = A[i++];  
        } else {  
            C[k++] = B[j++];  
        }  
    }  
    if (i == m) {  
        // done with A, place the rest of B  
        for (; j < n; j++) {  
            C[k++] = B[j];  
        }  
    } else {  
        // done with B, place the rest of A  
        for (; i < m; i++) {  
            C[k++] = A[i];  
        }  
    }  
}

__global__ void CoRankMergeSort(int *arr, int *temp, int n, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ThreadNum = blockDim.x * gridDim.x;
    const int GroupNum = (n + 2*step - 1) / (2*step); // 向上取整计算组数
    const int GThreadNum = (ThreadNum + GroupNum - 1) / GroupNum; // 每組至少1线程
    const int tidG = tid / GThreadNum;
    const int tid_in_group = tid % GThreadNum;

    // 边界检查：确保线程属于有效组
    if (tidG >= GroupNum) return;

    // 计算当前组的范围
    const int group_start = tidG * 2 * step;
    const int group_end = min(group_start + 2*step, n);
    const int group_size = group_end - group_start;

    // 组内任务划分
    const int sectionSize = (group_size + GThreadNum - 1) / GThreadNum;
    const int thisK_local = tid_in_group * sectionSize;
    const int nextK_local = min(thisK_local + sectionSize, group_size);
    const int thisK = group_start + thisK_local;
    const int nextK = group_start + nextK_local;

    // 处理A和B的实际长度
    const int startA = group_start;
    const int startB = startA + step;
    const int lenA = (startA < n) ? min(step, n - startA) : 0;
    const int lenB = (startB < n) ? min(step, n - startB) : 0;

    // 获取子数组指针
    int *A = &arr[startA];
    int *B = &arr[startB];

    // 计算合并分割点（需确保co_rank处理lenA/lenB为0的情况）
    int thisI = co_rank(thisK_local, A, lenA, B, lenB);
    int nextI = co_rank(nextK_local, A, lenA, B, lenB);
    int thisJ = thisK_local - thisI;
    int nextJ = nextK_local - nextI;

    // 边界修正（防止越界）
    thisJ = max(0, min(thisJ, lenB));
    nextJ = max(0, min(nextJ, lenB));

    // 合并到临时数组
    if (thisK < n) {
        merge(&A[thisI], nextI - thisI, 
             &B[thisJ], nextJ - thisJ, 
             &temp[thisK]);
    }
}


// 功能函数
// Function to print array
void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Automated function to check if array is sorted
bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

// Function to check if given number is a power of 2
bool isPowerOfTwo(int num) 
{
    return num > 0 && (num & (num - 1)) == 0;
}


// MAIN PROGRAM
int main()
{   
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "MERGE SORT IMPLEMENTATION" << std::endl;
    std::cout << "A Performance Comparison of These 4 Sorts in CPU vs GPU vs sharedMemory vs co-rank " << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    

    // 输入大小
    int size;
    std::cout << "\n\nEnter the size of the array. Must be a power of 2:\n ";
    std::cin>>size;
    // 判断是否为2的幂
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
    

    // 初始化数组
    //Create CPU based Arrays
    int* arr = new int[size];
    int* arr2 = new int[size];
    int* arr3 = new int[size];
    int* carr = new int[size];
    int* temp = new int[size];


    //Create GPU based arrays
    int* gpuArrmerge;
    int* gpuArrmerge2;
    int* gpuArrmerge3;
    int* gpuTemp;
    int* gpuTemp2;
    int* gpuTemp3;

    // Initialize the array with random values
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size; ++i) 
    {
        arr[i] = rand() % 256 + 1;
        arr2[i] = arr[i];
        arr3[i] = arr[i];
        carr[i] = arr[i];
    }

    //Print unsorted array 
    std::cout << "\n\nUnsorted array: ";
    if (size <= 256) 
    {
        printArray(arr, size);
    }
    else 
    {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }


    // 排序


    // CPU
    // 初始化时间变量
    // Initialize CPU clock counters
    clock_t startCPU, endCPU;
    // Time the CPU and call CPU Merge Sort
    startCPU = clock();
    mergeSortCPU(carr, temp, 0, size - 1);
    endCPU = clock();
    // Calculate Elapsed CPU time
    double millisecondsCPU = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);
    
    
    // GPU
    // 分配GPU内存
    // Allocate memory on GPU
    cudaMalloc((void**)&gpuArrmerge, size * sizeof(int));
    cudaMalloc((void**)&gpuTemp, size * sizeof(int));
    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge, arr, size * sizeof(int), cudaMemcpyHostToDevice); 
    
    // 初始化时间变量
    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;
    
    // Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    //Call GPU Merge Kernel and time the run
    cudaEventRecord(startGPU);
    for (int wid = 1; wid < size; wid *= 2)
    {
        MergeSortGPU << <threadsPerBlock, blocksPerGrid >> > (gpuArrmerge, gpuTemp, size, wid * 2);
    }
    cudaEventRecord(stopGPU);
    
    // 传送结果
    //Transfer sorted array back to CPU
    cudaMemcpy(arr, gpuArrmerge, size * sizeof(int), cudaMemcpyDeviceToHost);
    //Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
    //End
    cudaFree(gpuArrmerge);
    cudaFree(gpuTemp);


    // 共享内存
    // 分配GPU内存
    // Allocate memory on GPU
    cudaMalloc((void**)&gpuArrmerge2, size * sizeof(int));
    cudaMalloc((void**)&gpuTemp2, size * sizeof(int));
    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge2, arr2, size * sizeof(int), cudaMemcpyHostToDevice);   
    
    // 初始化时间变量
    // Perform sharedMemory merge sort and measure time
    cudaEvent_t startGPU2, stopGPU2; 
    cudaEventCreate(&startGPU2);
    cudaEventCreate(&stopGPU2);
    float millisecondsGPU2 = 0;  

    // 确定线程块大小
    int threadsPerBlock2 = 256; // 每个线程块的线程数
    int blocksPerGrid2 = (size + threadsPerBlock2 - 1) / threadsPerBlock2; // 计算线程块数量        
    // 计算共享内存的大小：每个线程块需要至少存储一个元素
    int sharedMemSize = threadsPerBlock2 * sizeof(int);
    // 记录开始时间
    cudaEventRecord(startGPU2);
    // 启动归并排序内核
    sharedMemoryMergeSort << < blocksPerGrid2, threadsPerBlock2, sharedMemSize >> > (gpuArrmerge2, gpuTemp2, size);
    // 记录结束时间
    cudaEventRecord(stopGPU2);

    // 传送结果
    //Transfer sorted array back to CPU
    cudaMemcpy(arr2, gpuArrmerge2, size * sizeof(int), cudaMemcpyDeviceToHost);
    //Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU2);
    cudaEventElapsedTime(&millisecondsGPU2, startGPU2, stopGPU2);
    //End
    cudaFree(gpuArrmerge2);
    cudaFree(gpuTemp2);



    // co-rank归并
    // 分配GPU内存
    // Allocate memory on GPU
    cudaMalloc((void**)&gpuArrmerge3, size * sizeof(int));
    cudaMalloc((void**)&gpuTemp3, size * sizeof(int));
    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge3, arr3, size * sizeof(int), cudaMemcpyHostToDevice); 

    // 初始化时间变量
    // Perform sharedMemory merge sort and measure time
    cudaEvent_t startGPU3, stopGPU3; 
    cudaEventCreate(&startGPU3);
    cudaEventCreate(&stopGPU3);
    float millisecondsGPU3 = 0;   

    // 确定线程块大小
    int threadsPerBlock3 = 256; // 每个线程块的线程数
    int blocksPerGrid3 = (size + threadsPerBlock3 - 1) / threadsPerBlock3; // 计算线程块数量        
    
    //Call GPU Merge Kernel and time the run
    cudaEventRecord(startGPU3);
    for (int wid = 1; wid < size; wid *= 2)
    {
        CoRankMergeSort << <threadsPerBlock3, blocksPerGrid3 >> >(gpuArrmerge3, gpuTemp3, size, wid);
        std::swap(gpuArrmerge3, gpuTemp3); // 交换输入输出
    }
    cudaEventRecord(stopGPU3);

    // 传送结果
    //Transfer sorted array back to CPU
    cudaMemcpy(arr3, gpuArrmerge3, size * sizeof(int), cudaMemcpyDeviceToHost);
    //Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU3);
    cudaEventElapsedTime(&millisecondsGPU3, startGPU3, stopGPU3);
    //End
    cudaFree(gpuArrmerge3);
    cudaFree(gpuTemp3);
        


    // 输出结果
    // Display sorted CPU array
    std::cout << "\n\nSorted CPU array: ";
    if (size <= 256) 
    {
        printArray(carr, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Display sorted GPU array
    std::cout << "\n\nSorted GPU array: ";
    if (size <= 256) 
    {
        printArray(arr, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Display sorted sharedMemory array
    std::cout << "\n\nSorted sharedMemory array: ";
    if (size <= 256) 
    {
        printArray(arr2, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Display co-rank array
    std::cout << "\n\nco-rank array: ";
    if (size <= 256) 
    {
        printArray(arr3, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }
    
    //Run the array with the automated isSorted checker
    if (isSorted(carr, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED CPU ARRAY" << std::endl;
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

    if (isSorted(arr3, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    //Print the time of the runs
    std::cout << "\n\nCPU Time: " << millisecondsCPU << " ms" << std::endl;
    std::cout << "GPU Time: " << millisecondsGPU << " ms" << std::endl;
    std::cout << "sharedMemory Time: " << millisecondsGPU2 << " ms" << std::endl;
    std::cout << "co-rank Time: " << millisecondsGPU3 << " ms" << std::endl;

    //Destroy all variables
    delete[] carr;
    delete[] arr;
    delete[] arr2;
    delete[] arr3;
    delete[] temp;

    std::cout << "\n------------------------------------------------------------------------------------\n||||| END. YOU MAY RUN THIS AGAIN |||||\n------------------------------------------------------------------------------------\n";
    return 0;
}
