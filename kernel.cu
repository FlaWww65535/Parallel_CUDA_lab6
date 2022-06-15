#include<iostream>
#include<iomanip>
#include<Windows.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"device_functions.h"

using namespace std;
const int MAXN = 2048;
const int BLOCK_SIZE = 1024;
float elm[MAXN][MAXN] = { 0 };
float ans[MAXN][MAXN] = { 0 };

const float eps = 1e-3;

__global__ void division_kernel(float* data, int k, int N) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;//计算线程索引
	float element = data[k * N + k];
	float temp = data[k * N + tid];
	data[k * N + tid] = (float)temp / element;
	return;
}

__global__ void eliminate_kernel(float* data, int k, int N) {
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tx == 0)data[k * N + k] = 1.0;//对角线元素设为 1
	int row = k + 1 + blockIdx.x;//每个块负责一行
	while (row < N) {
		int tid = threadIdx.x;
		while (k + 1 + tid < N) {
			int col = k + 1 + tid;
			float temp_1 = data[(row * N) + col];
			float temp_2 = data[(row * N) + k];
			float temp_3 = data[k * N + col];
			data[(row * N) + col] = temp_1 - temp_2 * temp_3;
			tid = tid + blockDim.x;
		}
		__syncthreads(); 
		if (threadIdx.x == 0) {
			data[row * N + k] = 0;
		}
		row += gridDim.x;
	}
	return;
}


int main() {
	freopen("input2048.dat", "r", stdin);
	for (int i = 0; i < MAXN; i++) {
		for (int j = 0; j < MAXN; j++) {
			cin >> elm[i][j];
		}
	}
	//for (int i = 0; i < MAXN; i++) {
	//	for (int j = 0; j < MAXN; j++) {
	//		cin >> ans[i][j];
	//	}
	//}
	float* temp = new float[MAXN * MAXN];
	for (int i = 0; i < MAXN; i++) {
		for (int j = 0; j < MAXN; j++) {
			temp[i * MAXN + j] = elm[i][j];
		}
	}
	cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess
	float* gpudata;
	float* result = new float[MAXN * MAXN];
	int size = MAXN * MAXN * sizeof(float);
	ret = cudaMalloc(&gpudata, size);//分配显存空间
	if (ret != cudaSuccess) {
		printf("cudaMalloc gpudata failed!\n");
	}
	
	ret = cudaMemcpy(gpudata, temp, size, cudaMemcpyHostToDevice);//将数据传输至 GPU 端
	
	if (ret != cudaSuccess) {
		printf("cudaMemcpyHostToDevice failed!\n");
	}

	dim3 dimBlock(BLOCK_SIZE, 1);//线程块
	dim3 dimGrid(1, 1);//线程网格
	cudaEvent_t start, stop;//计时器
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);//开始计时

	for (int k = 0; k < MAXN; k++) {
		division_kernel << <dimGrid, dimBlock >> > (gpudata,k,MAXN);
		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		ret = cudaGetLastError();
		if (ret != cudaSuccess) {
			printf("division_kernel failed, %s\n", cudaGetErrorString(ret));
		
		}
		
		eliminate_kernel << <dimGrid, dimBlock >> > (gpudata, k, MAXN);//负责消去任务的核函数
		cudaDeviceSynchronize();
		ret = cudaGetLastError();
		if (ret != cudaSuccess) {
			printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
		}
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("GPU_LU:%f ms\n", elapsedTime);
	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	}

	ret = cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost);//将数据传回 CPU 端
	if (ret != cudaSuccess) {
		printf("cudaMemcpyDeviceToHost failed!\n");
	}


	cudaFree(gpudata);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放
	cudaEventDestroy(start);
	cudaEventDestroy(stop);




	LARGE_INTEGER timeStart;	//开始时间
	LARGE_INTEGER timeEnd;		//结束时间

	LARGE_INTEGER frequency;	//计时器频率
	QueryPerformanceFrequency(&frequency);
	double quadpart = (double)frequency.QuadPart;//计时器频率

	QueryPerformanceCounter(&timeStart);
	for (int k = 0; k < MAXN; k++) {
		for (int j = k + 1; j < MAXN; j++) {
			elm[k][j] = elm[k][j] / elm[k][k];
		}
		elm[k][k] = 1.0f;
		for (int i = k + 1; i < MAXN; i++) {
			for (int j = k + 1; j < MAXN; j++) {
				elm[i][j] = elm[i][j] - elm[i][k] * elm[k][j];
			}
			elm[i][k] = 0;
		}
	}
	QueryPerformanceCounter(&timeEnd);

	//得到两个时间的耗时
	double elapsed = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;

	printf("Trivial_LU:%f ms\n", elapsed*1000);

	bool flag = 0;
	for (int i = 0; i < MAXN; i++) {
		for (int j = 0; j < MAXN; j++) {
			if (abs(result[i * MAXN + j] - elm[i][j]) > eps) { if (flag == 0)cout << result[i * MAXN + j] << " " << elm[i][j]<<endl; flag = 1; }
		}
	}

}