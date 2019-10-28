#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>




template <typename T>
class matrix
{
private:
	bool host = true;

	

public:
	int width;
	int height;
	int length1d;
	T* arr;
	matrix* cudamatrix;
	T* cudaarr;
	matrix() {}
	matrix(int width,int height):width(width),height(height),length1d(width*height)
	{
		arr = new T[width * height];
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
		int count = width * height;
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
		
	}

	__host__ matrix<T>* transpose() {
		auto m = new matrix<T>(height, width);

		int index = 0;
		while (index<length1d)
		{
			d_transpose<<<width,height>>>(m->getCudaMatrix(), getCudaMatrix(), index);
			auto err = cudaGetLastError();
			index += 1024;
		}
		m->fromCuda();
		cudaDeviceSynchronize();
		return m;
	}
	__host__ float determinant() {

	}

	__host__ matrix<T>* getCudaMatrix() {
		return cudamatrix;
	}
	__host__ void toCuda() {
		cudaMemcpy(cudaarr,arr, width * height * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ void fromCuda() {
		cudaMemcpy(arr, cudaarr, width * height * sizeof(T),cudaMemcpyDeviceToHost);
	}
	__device__ __host__ inline T* operator[](int index) {
		return &arr[index * width];
	}
	

};

template <typename T>
__global__ void d_transpose(matrix<T>* dest, matrix<T>* src, int index) {
	
	if (index >= dest->length1d)
		return;

	int x = threadIdx.x+index;
	int y = blockIdx.x;
	//check out of bound
	if (x >= src->width||y>=src->height) return;

	(*dest)[x][y] = (*src)[y][x];
}

template <typename T>
__global__ void d_determinant(matrix<T>* dest, matrix<T>* src, int index) {

	if (index >= dest->length1d)
		return;

	int x = threadIdx.x + index;
	int y = blockIdx.x;
	int sign = ((x + y) % 2) * 2 - 1;

	//check out of bound
	if (x >= dest->width || y >= dest->height) return;

	
}