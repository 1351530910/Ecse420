#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>

template <typename T>
class vector
{
private:

	bool host = true;

	vector()
	{

	}
public:
	int length = 0;
	T* arr = 0;
	vector* cudavector = 0;
	T* cudaarr = 0;
	vector(int length):length(length)
	{
		arr = (T*)calloc(length,sizeof(T));
		vector v;
		v.host = false;
		v.length = length;
		cudaMalloc(&cudavector, sizeof(vector));
		cudaMalloc(&cudaarr, length * sizeof(T));
		v.arr = cudaarr;
		cudaMemcpy(cudavector, &v, sizeof(vector), cudaMemcpyHostToDevice);
	}
	vector(int length,T* data):length(length)
	{
		arr = (T*)calloc(length, sizeof(T));
		memcpy(arr, data,length*sizeof(T));
		vector v;
		v.host = false;
		v.length = length;
		cudaMalloc(&cudavector, sizeof(vector));
		cudaMalloc(&cudaarr, length * sizeof(T));
		v.arr = cudaarr;
		cudaMemcpy(cudavector, &v, sizeof(vector), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaarr, arr, sizeof(length*sizeof(T)), cudaMemcpyHostToDevice);
	}
	~vector()
	{
		if (host)
		{
			if (cudavector)
				cudaFree(cudavector);
			if (cudaarr)
				cudaFree(cudaarr);
			
		}
	}

	__host__ __device__ inline T& operator[](int index) {
		return arr[index];
	}

	__host__ inline vector<T>& getCudaVector() {
		return *cudavector;
	}
	__host__ inline void toCuda() {
		cudaMemcpy(cudaarr, arr, length * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ inline void fromCuda() {
		cudaMemcpy(arr, cudaarr, length * sizeof(T), cudaMemcpyDeviceToHost);
	}
};

