
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <time.h>
#include <string>

#include "lodepng.h"
#include "matrix.h"
#include "constants.h"

__global__ void d_convolution(matrix<float>& input, matrix<float>& transformation, matrix<float>& output, int i) {

	int index = i + threadIdx.x + blockIdx.x * 1024;
	
	//check for out of bound
	if (index >= output.length1d) return;

	//compute the pixel x
	int x = index / output.height;
	int y = index % output.height;

	//do not modify opacity
	if (x%4==3)
	{
		output[y][x] = 255;
		return;
	}
	float sum = 0;
	
	for (int i = 0; i < transformation.width; i++)
	{
		for (int j = 0; j < transformation.height; j++)
		{
			sum += input[y + j][x + i * 4] * transformation[j][i];
		}
	}

	if (sum < 0) sum = 0;
	if (sum > 255) sum = 255;
	output[y][x] = sum;
}

double h_convolution(const char* input_filename, const char* output_filename, const int nthreads, const int weightMatrixSize) {

	//note the time
	clock_t StartTime = clock();

	unsigned error;
	unsigned char* image;
	unsigned int width, height;
	cudaError_t cudaError;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);

	if (error) {
		std::cerr << "error " << error << ": " << lodepng_error_text(error) << std::endl;;
		return 0;
	}

	//rgba to pixel
	width *= 4;

	//initialize cuda
	if (cudaError = cudaSetDevice(0)) goto Error;

	//initialize matrices
	auto input = new matrix<float>(width, height);
	auto output = new matrix<float>(width - weightMatrixSize*4 + 4, height - weightMatrixSize + 1);
	auto wm = new matrix<float>(weightMatrixSize, weightMatrixSize);
	unsigned char* outImage = new unsigned char[output->length1d];

	//define the weight matrix
	switch (weightMatrixSize)
	{
	case 3:
		memcpy(wm->arr, w3, 9 * sizeof(float)); break;
	case 5:
		memcpy(wm->arr, w5, 25 * sizeof(float)); break;
	case 7:
		memcpy(wm->arr, w7, 49 * sizeof(float)); break;
	default:
		goto End;
	}


	//copy image to the float matrix
	for (size_t i = 0; i < width * height; i++)
	{
		input->arr[i] = image[i];
	}
	input->toCuda();
	wm->toCuda();

	int index = 0;
	const int end = output->length1d;

	int blocks = 1;	//nblocks
	int threads = nthreads;	//nthreads
	int left = 0;

	if (nthreads > 1024)
	{
		blocks = std::floor(nthreads / 1024.0 - 0.000001);
		threads = 1024;
		left = nthreads % 1024;
	}

	//loop
	while (index < end)
	{
		d_convolution << < blocks, threads >> > (input->getCudaMatrix(), wm->getCudaMatrix(), output->getCudaMatrix(), index);
		index += threads * blocks;
		if (left)
		{
			d_convolution << <1, left >> > (input->getCudaMatrix(), wm->getCudaMatrix(), output->getCudaMatrix(), index);
			index += left;
		}
	}

	//check for error
	if (cudaError = cudaGetLastError()) goto Error;
	if (cudaError = cudaDeviceSynchronize()) goto Error;

	//load from cuda
	output->fromCuda();

	//convert float matrix to uchar
	for (size_t i = 0; i < output->length1d; i++)
		outImage[i] = output->arr[i];

	lodepng_encode32_file(output_filename, outImage, output->width/4, output->height);

	goto End;

Error:
	//print error
	std::cerr << "cuda error: " << cudaGetErrorString(cudaError) << std::endl;

End:
	//free memories allocated
	free(image);
	delete input;
	delete output;
	delete wm;
	delete[] outImage;
	return (clock() - StartTime) / double(CLOCKS_PER_SEC);
}

void print(matrix<int>* m) {
	for (size_t i = 0; i < m->height; i++)
	{
		for (size_t j = 0; j < m->width; j++)
		{
			std::cout << (*m)[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void main(int argc, const char* argv[]) {


	//const char* image = "original.png";

	//std::cout << h_convolution(image, "333.png", 1024, 3) << std::endl;
	//std::cout << h_convolution(image, "555.png", 1024, 5) << std::endl;
	//std::cout << h_convolution(image, "777.png", 1024, 7) << std::endl;


	cudaSetDevice(0);
	std::ofstream result;
	result.open("result.txt");
	srand(time(0));
	for (size_t i = 1; i < 100; i++)
	{
		int s = i*10;
		double* arr = new double[s * s];
		for (size_t j = 0; j < s*s; j++)
		{
			arr[j] = rand()%10;
		}
		
		matrix<double> m(s, s, arr);
		
		clock_t StartTime = double(clock());
		auto n = m.inverse();
		result <<s<<"\t"<< std::to_string((double(clock()) - StartTime) / double(CLOCKS_PER_SEC))<<std::endl;
		n[1][1] = 1;

		//vector<double> v(s, arr);
		//vector<double> res = v * m;
		//vector<double> org = res * m.inverse();

		delete[] arr;
	}
	
	
	return;
}