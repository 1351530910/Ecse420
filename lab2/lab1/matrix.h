#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include "vector.h"

template <typename T>
class matrix
{
private:
	bool host = true;
	matrix<T>* cudamatrix = 0;
	T* cudaarr = 0;
public:
	int width = 0;
	int height = 0;
	int length1d  = 0;
	T* arr = 0;
	
	matrix() {
		
	}
	matrix(int width,int height):width(width),height(height),length1d(width*height)
	{
		arr = new T[width*height];
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

	__host__ bool invert() {
		//construct an identity matrix
		matrix<T> Identity(width, height);
		for (size_t i = 0; i < Identity.length1d; i++)
		{
			Identity.arr[i] = 0;
		}
		for (size_t i = 0; i < Identity.width; i++)
			Identity[i][i] = 1;
		Identity.toCuda();

		//a temp clone of the two
		matrix<T> clone(width, height, arr);
		matrix<T> IC(width, height, Identity.arr);

		for (size_t i = 0; i < Identity.width; i++)
		{
			d_inverse << <height, width >> > (getCudaMatrix(), clone.getCudaMatrix(), Identity.getCudaMatrix(), IC.getCudaMatrix(), i);
			cudaDeviceSynchronize();
			copy<<<height,width>>>(clone.getCudaMatrix(), getCudaMatrix());
			copy << <height, width >> > (IC.getCudaMatrix(), Identity.getCudaMatrix());
			cudaDeviceSynchronize();
#if DEBUG
			fromCuda();
			Identity.fromCuda();
			print();
			Identity.print();
#endif 
		}

		//normalize inverse matrix
		d_normalize << <height, width >> > (getCudaMatrix(), IC.getCudaMatrix());
		cudaDeviceSynchronize();

		//verify if success
		T n;
		cudaMemcpy(&n, cudaarr, sizeof(T), cudaMemcpyDeviceToHost);

		if (n)
		{
			IC.fromCuda();
			cudaMemcpy(arr, IC.cudaarr, sizeof(T), cudaMemcpyDeviceToHost);
			toCuda();
			return true;
		}
		else {
			toCuda();
			return false;
		}
	}

	void print() {
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
			{
				std::cout << (*this)[i][j] << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	__host__ inline matrix<T>& getCudaMatrix() {
		return *cudamatrix;
	}
	__host__ inline void toCuda() {
		cudaMemcpy(cudaarr,arr, width * height * sizeof(T), cudaMemcpyHostToDevice);
	}
	__host__ inline void fromCuda() {
		cudaMemcpy(arr, cudaarr, width * height * sizeof(T),cudaMemcpyDeviceToHost);
	}
	__device__ __host__ inline T* operator[](int index) {
		return &arr[index * width];
	}
};


template <typename T>
__global__ void d_inverse(matrix<T>& m,matrix<T>& clone,matrix<T>& inv,matrix<T>& IC,const int index) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int cy = index;
	
	//find the cy
	while (m[cy][index]==0)
	{
		//all 0, matrix not invertible
		if (++cy == m.height) {
			clone[y][x] = 0;
			return;
		}
	}

	if (y!=cy&& m[y][index]!=0)
	{
		clone[y][x] = m[y][x] * m[cy][index] / m[y][index] - m[cy][x];
		IC[y][x] = inv[y][x] * m[cy][index] / m[y][index] - inv[cy][x];
	}
	else {
		clone[y][x] = m[y][x];
		IC[y][x] = inv[y][x];
	}
}

template <typename T>
__global__ void d_normalize(matrix<T>& m, matrix<T>& inv) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	if (m[y][y]!=0)
	{
		inv[y][x] = inv[y][x] / m[y][y];
	}
}

template <typename T>
__global__ void copy(matrix<T>& src, matrix<T>& dest) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	dest[y][x] = src[y][x];
}

template <typename T>
vector<T>& operator*(matrix<T>& m, vector<T>& v) {
	vector<T> temp(v.length,v.arr);
	multiply << <m.height, m.width >> > (m.getCudaMatrix(), v.getCudaVector());
	cudaDeviceSynchronize();
	toVector << <1, v.length >> > (m.getCudaMatrix(), temp.getCudaVector());
	cudaDeviceSynchronize();
	temp.fromCuda();
	m.toCuda();
	return temp;
}
template <typename T>
vector<T>& operator*(vector<T>& v,matrix<T>& m) {
	return m * v;
}
template <typename T>
__global__ void multiply(matrix<T>& m, vector<T>& v) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	m[y][x] = m[y][x] * v[x];
}

template <typename T>
__global__ void toVector(matrix<T>& m, vector<T>& v) {
	int x = threadIdx.x;
	v[x] = 0;
	for (size_t i = 0; i < m.width; i++)
	{
		v[x] += m[x][i];
	}
}