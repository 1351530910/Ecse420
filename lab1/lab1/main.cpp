
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <time.h>

#include "lodepng.h"

__global__ void poll(unsigned char* arr1, unsigned char* arr2, unsigned char* out) {
	int x = threadIdx.x;

	out[x] = arr1[x];
	if (arr1[1 + x] > out[x])
		out[x] = arr1[1 + x];
	if (arr2[x] > out[x])
		out[x] = arr2[x];
	if (arr2[x + 1] > out[x])
		out[x] = arr2[x + 1];
}

void polling(const char* input_filename, const char* output_filename, const int maxThreads) {
	unsigned error;
	unsigned char* image, * new_image;
	unsigned char* upImage, * downImage;
	unsigned int width, height;
	unsigned char* cudaUpImage,* cudaDownImage, * cudaNewImage;

	cudaError_t cudaError;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);


	if (error) {
		std::cerr << "error " << error << ": " << lodepng_error_text(error);
		goto Error;
	}

	int length = width * height * 2;

	new_image = new unsigned char[length/2];
	upImage = new unsigned char[length];
	downImage = new unsigned char[length];

	
	for (int i = 0; i < height/2; i++)
	{
		memcpy(upImage + (i * width*4), image + (2 * i * width*4), width *4* sizeof(unsigned char));
		memcpy(downImage + (i * width*4), image + (2 * i * width*4+width*4), width*4 * sizeof(unsigned char));
	}

	std::cout << "polling started with threads:" << maxThreads << "\t";
	clock_t StartTime = clock();

	if (cudaError = cudaSetDevice(0)) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaUpImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaDownImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaNewImage, length/2 * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMemcpy(cudaUpImage,upImage,length*sizeof(unsigned char),cudaMemcpyHostToDevice)) goto Error;
	if (cudaError = cudaMemcpy(cudaDownImage, downImage, length * sizeof(unsigned char), cudaMemcpyHostToDevice)) goto Error;

	unsigned char* upImageCpy = cudaUpImage,* downImageCpy = cudaDownImage,*outImage = cudaNewImage;
	int count = length / 2;
	while (count>maxThreads)
	{
		poll <<<1, maxThreads >>> (upImageCpy, downImageCpy, cudaNewImage);
		if (cudaError = cudaGetLastError()) 
			goto Error;
		upImageCpy += maxThreads*2;
		downImageCpy += maxThreads*2;
		outImage += maxThreads;
		count -= maxThreads;
	}
	poll <<<1, count >>> (upImageCpy, downImageCpy, outImage);
	if (cudaError = cudaGetLastError())
		goto Error;

	
	if (cudaError = cudaDeviceSynchronize()) goto Error;
	if (cudaError = cudaMemcpy(upImage, cudaUpImage, length * sizeof(unsigned char), cudaMemcpyDeviceToHost)) goto Error;
	if (cudaError = cudaMemcpy(downImage, cudaDownImage, length * sizeof(unsigned char), cudaMemcpyDeviceToHost)) goto Error;


	std::cout << "time used:" << (clock() - StartTime) / double(CLOCKS_PER_SEC) << std::endl;
	lodepng_encode32_file(output_filename, new_image, width/2, height/2);
	goto End;

Error:
	std::cerr << "cuda error: " << cudaGetErrorString(cudaError) << std::endl;

End:

	delete[] upImage;
	delete[] downImage;
	free(image);
	delete[] new_image;
	cudaFree(cudaUpImage);
	cudaFree(cudaDownImage);
	cudaFree(cudaNewImage);
}

__global__ void rectify(unsigned const char* in, unsigned char* out) {
	int i = threadIdx.x;
	if (in[i] < 127) out[i] = 127;
	else out[i] = in[i];
}

void rectification(const char* input_filename, const char* output_filename,const int maxThreads) {
	unsigned error;
	unsigned char* image, * new_image;
	unsigned int width, height;
	unsigned char* cudaImage, * cudaNewImage;
	cudaError_t cudaError;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	
	if (error) {
		std::cerr << "error " << error << ": " << lodepng_error_text(error);
		goto Error;
	}

	int length = width * height * 4;

	new_image = new unsigned char[length];

	std::cout << "rectification started with threads:" << maxThreads << "\t";
	clock_t StartTime = clock();


	if (cudaError = cudaSetDevice(0)) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaImage,length*sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaNewImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMemcpy(cudaImage,image,length*sizeof(unsigned char),cudaMemcpyHostToDevice)) goto Error;

	unsigned char* cudaImageCpy = cudaImage, * cudaNewImageCpy = cudaNewImage;
	int left = width * height * 4-1;
	while (left> maxThreads)
	{
		rectify <<< 1, maxThreads >>> (cudaImageCpy, cudaNewImageCpy);
		if (cudaError = cudaGetLastError()) goto Error;
		cudaImageCpy += maxThreads;
		cudaNewImageCpy += maxThreads;
		left -= maxThreads;
		
	}
	rectify << < 1, left >> > (cudaImageCpy, cudaNewImageCpy);
	if (cudaError = cudaGetLastError()) goto Error;
	if (cudaError = cudaDeviceSynchronize()) goto Error;
	if (cudaError = cudaMemcpy(new_image, cudaNewImage, length * sizeof(unsigned char), cudaMemcpyDeviceToHost)) goto Error;


	std::cout << "time used:" << (clock() - StartTime) / double(CLOCKS_PER_SEC) << std::endl;
	lodepng_encode32_file(output_filename, new_image, width, height);
	goto End;

Error:
	std::cerr << "cuda error: " << cudaGetErrorString(cudaError) <<std::endl;
	
End:
	cudaFree(cudaImage);
	cudaFree(cudaNewImage);
	free(image);
	delete[] new_image;
}

void main(int argc, const char* argv[]) {
	
	const char* input = "C:/Users/Administrator/Desktop/test.png";
	const char* rect = "C:/Users/Administrator/Desktop/rectification.png";
	const char* poll = "C:/Users/Administrator/Desktop/test2.png";

	std::cout << "question 1" << std::endl;
	for (size_t exp = 0; exp < 11; exp++)
	{
		rectification(
			input,
			rect,
			int(std::pow(2, exp))
		);
	}
	
	std::cout << std::endl << std::endl << "question 2" << std::endl;

	polling(
		input,
		poll,
		64
	);
}