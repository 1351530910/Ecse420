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
	__host__ matrix<T>* inverse() {

		//construct an identity matrix
		auto Identity = new matrix<T>(width, height);
		for (size_t i = 0; i < Identity->length1d; i++)
		{
			Identity->arr[i] = 0;
		}
		for (size_t i = 0; i < Identity->width; i++)
			(*Identity)[i][i] = 1;
		Identity->toCuda();

		//a temp clone of the two
		auto clone = new matrix<T>(width, height, arr);
		auto IC = new matrix<T>(width, height, Identity->arr);

		for (size_t i = 0; i < Identity->width; i++)
		{
			d_inverse << <height, width >> > (getCudaMatrix(), clone->getCudaMatrix(), Identity->getCudaMatrix(), IC->getCudaMatrix(), i);
			cudaDeviceSynchronize();
			copy<<<height,width>>>(clone->getCudaMatrix(), getCudaMatrix());
			copy << <height, width >> > (IC->getCudaMatrix(), Identity->getCudaMatrix());
			cudaDeviceSynchronize();
#if DEBUG
			fromCuda();
			Identity->fromCuda();
			print();
			Identity->print();
#endif // DEBUG

			
		}

		//normalize inverse matrix
		d_normalize << <height, width >> > (getCudaMatrix(), IC->getCudaMatrix());
		cudaDeviceSynchronize();

		//verify if success
		T n;
		cudaMemcpy(&n, cudaarr, sizeof(T), cudaMemcpyDeviceToHost);

		//restore cuda matrix
		toCuda();

		delete clone;
		delete Identity;

		if (n==0)
		{
			delete IC;
			return 0;
		}
		else {
			IC->fromCuda();
			return IC;
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

	__host__ matrix<T>* getCudaMatrix() {
		return cudamatrix;
	}
	__host__ void toCuda() {
		auto err = cudaMemcpy(cudaarr,arr, width * height * sizeof(T), cudaMemcpyHostToDevice);
		if (err)
			std::cout << "tocuda error " << err;
	}
	__host__ void fromCuda() {
		auto err = cudaMemcpy(arr, cudaarr, width * height * sizeof(T),cudaMemcpyDeviceToHost);
		if (err)
			std::cout << "tocuda error " << err;
	}
	__device__ __host__ T* operator[](int index) {
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
__global__ void d_inverse(matrix<T>* m,matrix<T>* clone,matrix<T>* inv,matrix<T>* IC,const int index) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int cy = index;
	
	//find the cy
	while ((*m)[cy][index]==0)
	{
		//all 0, matrix not invertible
		if (++cy == m->height) {
			(*clone)[y][x] = 0;
			return;
		}
	}

	if (y!=cy&& (*m)[y][index]!=0)
	{
		(*clone)[y][x] = (*m)[y][x] * (*m)[cy][index] / (*m)[y][index] - (*m)[cy][x];
		(*IC)[y][x] = (*inv)[y][x] * (*m)[cy][index] / (*m)[y][index] - (*inv)[cy][x];
	}
	else {
		(*clone)[y][x] = (*m)[y][x];
		(*IC)[y][x] = (*inv)[y][x];
	}
}

template <typename T>
__global__ void d_normalize(matrix<T>* m, matrix<T>* inv) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	if ((*m)[y][y]!=0)
	{
		(*inv)[y][x] = (*inv)[y][x] / (*m)[y][y];
	}
}

template <typename T>
__global__ void copy(matrix<T>* src, matrix<T>* dest) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	(*dest)[y][x] = (*src)[y][x];
}