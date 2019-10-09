
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <time.h>

#include "lodepng.h"

#define MAX_MSE 0.00001f

float get_MSE(const char* input_filename_1, const char* input_filename_2)
{
	unsigned error1, error2;
	unsigned char* image1, * image2;
	unsigned width1, height1, width2, height2;

	error1 = lodepng_decode32_file(&image1, &width1, &height1, input_filename_1);
	error2 = lodepng_decode32_file(&image2, &width2, &height2, input_filename_2);
	if (error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));
	if (error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));
	if (width1 != width2) printf("images do not have same width\n");
	if (height1 != height2) printf("images do not have same height\n");

	// process image
	float im1, im2, diff, sum, MSE;
	sum = 0;
	for (int i = 0; i < width1 * height1; i++) {
		im1 = (float)image1[i];
		im2 = (float)image2[i];
		diff = im1 - im2;
		sum += diff * diff;
	}
	MSE = sqrt(sum) / (width1 * height1);

	free(image1);
	free(image2);

	return MSE;
}

__global__ void poll(unsigned char* upImage, unsigned char* downImage, unsigned char* out) {
	int x = threadIdx.x;
	for (int i = 0; i < 4; i++)	//loop for RGBA
	{
		out[x*4+i] = upImage[x*8+i]; //assume first one is max
		if (upImage[x*8+4+i] > out[x*4+i])	//replace if the value is higher
			out[x*4+i] = upImage[x*8+4+i];
		if (downImage[x*8+i] > out[x*4+i])
			out[x*4+i] = downImage[x*8+i];
		if (downImage[x*8+4+i] > out[x*4+i])
			out[x*4+i] = downImage[x*8+4+i];
	}
}

void polling(const char* input_filename, const char* output_filename, const int maxThreads) {
	
	//initial variables
	unsigned error;
	unsigned char* image, * new_image;
	unsigned char* upImage, * downImage;
	unsigned int width, height;
	unsigned char* cudaUpImage,* cudaDownImage, * cudaNewImage;
	cudaError_t cudaError;

	//load image file
	error = lodepng_decode32_file(&image, &width, &height, input_filename);

	//if error go to error
	if (error) {
		std::cerr << "error " << error << ": " << lodepng_error_text(error);
		goto Error;
	}

	//compute the length of the arrays
	int length = width * height * 2;
	new_image = new unsigned char[length/2];
	upImage = new unsigned char[length];
	downImage = new unsigned char[length];

	//divide the image to two new images 
	for (int i = 0; i < height/2; i++)
	{
		memcpy(upImage + (i * width*4), image + (2 * i * width*4), width *4* sizeof(unsigned char));
		memcpy(downImage + (i * width*4), image + (2 * i * width*4+width*4), width*4 * sizeof(unsigned char));
	}

	//for debug purpose
	clock_t StartTime = clock();

	//allocate memory on graphics card
	if (cudaError = cudaSetDevice(0)) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaUpImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaDownImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaNewImage, length/2 * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMemcpy(cudaUpImage,upImage,length*sizeof(unsigned char),cudaMemcpyHostToDevice)) goto Error;
	if (cudaError = cudaMemcpy(cudaDownImage, downImage, length * sizeof(unsigned char), cudaMemcpyHostToDevice)) goto Error;

	//save a copy of pointers
	unsigned char* cudaUpImageCpy = cudaUpImage,
		* cudaDownImageCpy = cudaDownImage,
		*cudaOutImage = cudaNewImage;
	
	//total number of pixels
	int npixels = width*height/4;
	
	//cuda runtime loop
	while (npixels > maxThreads)
	{
		poll << <1, maxThreads >> > (cudaUpImageCpy, cudaDownImageCpy, cudaOutImage);
		if (cudaError = cudaGetLastError()) goto Error;

		//compute the next array base size
		cudaUpImageCpy += 8 * maxThreads;
		cudaDownImageCpy += 8 * maxThreads;
		cudaOutImage += 4 * maxThreads;
		npixels -= maxThreads;
	}

	//compute the last remaining pixels
	poll << <1, npixels >> > (cudaUpImageCpy, cudaDownImageCpy, cudaOutImage);
	if (cudaError = cudaGetLastError()) goto Error;

	//wait for synchronization
	if (cudaError = cudaDeviceSynchronize()) goto Error;

	//copy back the result
	if (cudaError = cudaMemcpy(new_image, cudaNewImage, length/2 * sizeof(unsigned char), cudaMemcpyDeviceToHost)) goto Error;


	std::cout << (clock() - StartTime) / double(CLOCKS_PER_SEC);
	lodepng_encode32_file(output_filename, new_image, width/2, height/2);
	goto End;

Error:
	//print the cuda errors if there are any
	std::cerr << "cuda error: " << cudaGetErrorString(cudaError) << std::endl;

End:
	//free all pointers
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

	//variables
	unsigned error;
	unsigned char* image, * new_image;
	unsigned int width, height;
	unsigned char* cudaImage, * cudaNewImage;
	cudaError_t cudaError;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	
	if (error) {
		std::cerr << "error " << error << ": " << lodepng_error_text(error) << std::endl;;
		goto Error;
	}

	//allocate the output image
	int length = width * height * 4;
	new_image = new unsigned char[length];

	//note the time
	clock_t StartTime = clock();

	//initialize cuda
	if (cudaError = cudaSetDevice(0)) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaImage,length*sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMalloc((void**)&cudaNewImage, length * sizeof(unsigned char))) goto Error;
	if (cudaError = cudaMemcpy(cudaImage,image,length*sizeof(unsigned char),cudaMemcpyHostToDevice)) goto Error;

	//save a copy of the initial pointers 
	unsigned char* cudaImageCpy = cudaImage,
		* cudaNewImageCpy = cudaNewImage;

	//number of pixels left
	int left = width * height * 4-1;
	while (left> maxThreads)
	{
		rectify <<< 1, maxThreads >>> (cudaImageCpy, cudaNewImageCpy);
		if (cudaError = cudaGetLastError()) goto Error;
		cudaImageCpy += maxThreads;
		cudaNewImageCpy += maxThreads;
		left -= maxThreads;
		
	}

	//compute the remaining pixels
	rectify << < 1, left >> > (cudaImageCpy, cudaNewImageCpy);
	if (cudaError = cudaGetLastError()) goto Error;
	if (cudaError = cudaDeviceSynchronize()) goto Error;
	if (cudaError = cudaMemcpy(new_image, cudaNewImage, length * sizeof(unsigned char), cudaMemcpyDeviceToHost)) goto Error;


	std::cout <<  (clock() - StartTime) / double(CLOCKS_PER_SEC);
	lodepng_encode32_file(output_filename, new_image, width, height);
	goto End;

Error:
	//print error
	std::cerr << "cuda error: " << cudaGetErrorString(cudaError) <<std::endl;
	
End:
	//free memories allocated
	cudaFree(cudaImage);
	cudaFree(cudaNewImage);
	free(image);
	delete[] new_image;
}


void main(int argc, const char* argv[]) {
	
	const char* input = "test.png";
	const char* rect = "rect.png";
	const char* poll = "poll.png";

	for (size_t exp = 12; exp < 14; exp++)
	{
		std::cout << int(std::pow(2, exp)) << " threads\tquestion 1:";
		rectification(
			input,
			rect,
			int(std::pow(2, exp))
		);
		std::cout << "\tquestion 2:";
		polling(
			input,
			poll,
			int(std::pow(2, exp))
		);
		std::cout << "\t" << get_MSE("rect.png", "test_rectify.png");
		std::cout << "\t" << get_MSE("poll.png", "test_pooling.png");
		std::cout << std::endl;
	}
	

	//const char* input = argv[1];
	//const char* output = argv[2];

	//std::string str(argv[0]);
	//auto program = str.substr(str.find_last_of("/\\") + 1);

	//if (!program.compare("rectify") || !program.compare("rectify.exe"))
	//	rectification(input, output, std::stoi(argv[3]));
	//else if (!program.compare("pool") || !program.compare("pool.exe"))
	//	polling(input, output, std::stoi(argv[3]));
	//else if (!program.compare("mse") || !program.compare("mse.exe"))
	//	std::cout << get_MSE(input, output) << std::endl;
	//else
	//	std::cout << "unable to identify program rectify.exe or pool.exe or mse.exe" << std::endl;
}