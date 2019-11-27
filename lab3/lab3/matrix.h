#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>

template <typename T>
class matrix
{
public:
	bool host = true;
	matrix<T>* cudamatrix = 0;
	T* cudaarr = 0;

	float cudathreads = 1.0f;
	int width = 0;
	int height = 0;
	int length1d  = 0;
	T* arr = 0;
	
	matrix() {
		
	}
	matrix(int width,int height):width(width),height(height),length1d(width*height)
	{
		arr = new T[width*height];
		memset(arr, 0,width * height * sizeof(T));
		matrix m;
		m.host = false;
		m.width = width;
		m.height = height;
		m.length1d = width * height;
		cudaMalloc(&cudamatrix, sizeof(matrix));
		cudaMalloc(&cudaarr, width * height * sizeof(T));
		m.arr = cudaarr;
		cudaMemcpy(cudamatrix, &m, sizeof(matrix), cudaMemcpyHostToDevice);
	}

	matrix(int width, int height,T* data) :width(width), height(height), length1d(width* height)
	{
		arr = new T[width * height];
		memcpy(arr, data, width * height * sizeof(T));
		if (host)
		{
			matrix m;
			m.host = false;
			m.width = width;
			m.height = height;
			cudaMalloc(&cudamatrix, sizeof(matrix));
			cudaMalloc(&cudaarr, width * height * sizeof(T));
			m.arr = cudaarr;
			cudaMemcpy(cudamatrix, &m, sizeof(matrix), cudaMemcpyHostToDevice);
			cudaMemcpy(cudaarr, arr, width * height * sizeof(T), cudaMemcpyHostToDevice);
		}
	}

	~matrix()
	{
		if (host)
		{
			if (cudamatrix)
				cudaFree(cudamatrix);
			if (cudaarr)
				cudaFree(cudaarr);
			delete[] arr;
			
		}
		arr = 0;
		cudamatrix = 0;
		cudaarr = 0;
	}
	inline size_t memsize() {
		return length1d * sizeof(T);
	}

	__host__ inline matrix<T>& getCudaMatrix() {
		return *cudamatrix;
	}
	__host__ inline cudaError_t toCuda() {
		return cudaMemcpy(cudaarr,arr, width * height * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ inline cudaError_t fromCuda() {
		return cudaMemcpy(arr, cudaarr, width * height * sizeof(T),cudaMemcpyDeviceToHost);
	}
	__device__ __host__ inline T* operator[](int index) {
		return &arr[index * width];
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& os,matrix<T>& m)
{
	for (int i = 0; i < m.height; i++)
	{
		for (int j = 0; j < m.width; j++)
		{
			os << m[i][j] << ",";
		}
		os << std::endl;
	}
	os << std::endl;
	return os;
}