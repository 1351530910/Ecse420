
#include <iostream>
#include <fstream>
#include <chrono>

#include "matrix.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define rho 0.5
#define upeta 0.0002
#define G 0.75
#define minusUpeta (1-upeta)
#define invUpeta (1.0 / (1 + upeta))


//formula from handout
//normal case
#define normalCase (u[i][j] = invUpeta * (rho * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j - 1] + u1[i][j + 1] - 4 * u1[i][j]) + 2 * u1[i][j] - minusUpeta * u2[i][j]))

//boundaries
#define rightBoundary (u[N - 1][j] = G * u[N - 2][j])
#define leftBoundary u[0][j] = G * u[1][j]
#define topBoundary u[i][0] = G * u[i][1]
#define bottomBoundary u[i][N - 1] = G * u[i][N - 2]
//corners
#define topleft u[0][0] = G * u[1][0]
#define topright u[N - 1][0] = G * u[N - 2][0]
#define bottomleft u[0][N - 1] = G * u[0][N - 2]
#define bottomright u[N - 1][N - 1] = G * u[N - 1][N - 2]

using namespace std;

__global__ void d_inside(matrix<double>& u, matrix<double>& u1, matrix<double>& u2 ) {
	int i = threadIdx.x+1;
	int j = blockIdx.x+1;
	normalCase;
}
__global__ void d_border(matrix<double>& u) {
	int i = threadIdx.x+1;
	int j = i;
	int k = blockIdx.x;
	int N = u.height;
	switch (k)
	{
	case 0:	//left boundary
		leftBoundary;
		return;
	case 1:	//right boundary
		rightBoundary;
		return;
	case 2:	//top boundary
		topBoundary;
		return;
	case 3:	//bottom boundary
		bottomBoundary;
		return;
	default:
		break;
	}
}
__global__ void d_corner(matrix<double>& u) {
	int i = threadIdx.x;
	int N = u.height;
	switch (i)
	{
	case 0:	
		topleft;
		return;
	case 1:	
		topright;
		return;
	case 2:
		bottomleft;
		return;
	case 3:
		bottomright;
		return;
	default:
		break;
	}
}

//per grid of 16*16
__global__ void d_decomposition(matrix<double>& u, matrix<double>& u1, matrix<double>& u2) {
	const int N = u.height;
	const int x = threadIdx.x;
	const int y = blockIdx.x;
	if (!x) {//if x == 0
		if (!y) {	//if y == 0
			for (int i = 15 + x * 16; i >= x * 16; i--)
				for (int j = 15 + y * 16; j >= y * 16; j--) {
					if (i && j)			normalCase;
					else if (i && !j)	topBoundary;
					else if (!i && j)	leftBoundary;
					else				topleft;
				}
		}
		else
			for (int i = 15 + x * 16; i >= x * 16; i--)
				for (int j = y * 16; j < y * 16 + 16; j++)
					if (i)
						if (j == N - 1) bottomBoundary;
						else			normalCase;
					else
						if (j == N - 1)	bottomleft;
						else			leftBoundary;
	}
	else { 	//if i > 0
		if (!y) { 	//if j == 0
			for (int i = x * 16; i < 16 + x * 16; i++)
				for (int j = 15 + y * 16; j >= y * 16; j--)
					if (j)
						if (i == N - 1)	rightBoundary;
						else			normalCase;
					else
						if (i == N - 1)	topright;
						else			topBoundary;
		}
		else
			for (int i = x * 16; i < 16 + x * 16; i++)
				for (int j = y * 16; j < y * 16 + 16; j++)
					if (j == N - 1)
						if (i == N - 1)	bottomright;
						else			bottomBoundary;
					else
						if (i == N - 1)	rightBoundary;
						else			normalCase;
	}
}

void h_decomposition(const long t, ostream* os) {
	const int N = 512;
	matrix<double> u(N, N);
	matrix<double> u1(N, N);
	matrix<double> u2(N, N);
	u[N / 2][N / 2] = 1;
	CUDACHECK(u.toCuda());

	for (size_t _ = 0; _ < t; _++)
	{
		CUDACHECK(cudaMemcpy(u2.cudaarr, u1.cudaarr, u1.memsize(), cudaMemcpyDeviceToDevice));
		CUDACHECK(cudaMemcpy(u1.cudaarr, u.cudaarr, u1.memsize(), cudaMemcpyDeviceToDevice));
		
		d_decomposition << <N/16, N/16 >> > (*u.cudamatrix, *u1.cudamatrix, *u2.cudamatrix);
		CUDACHECK(cudaDeviceSynchronize());
		CUDACHECK(u.fromCuda());
		if(os) *os << u << endl;
	}
}

void h_parallel(const long t,const long N, ostream* os) {
	matrix<double> u(N, N);
	matrix<double> u1(N, N);
	matrix<double> u2(N, N);
	u[N / 2][N / 2] = 1;
	CUDACHECK(u.toCuda());

	for (size_t _ = 0; _ < t; _++)
	{
		CUDACHECK(cudaMemcpy(u2.cudaarr, u1.cudaarr, u1.memsize(),cudaMemcpyDeviceToDevice));
		CUDACHECK(cudaMemcpy(u1.cudaarr, u.cudaarr, u1.memsize(),cudaMemcpyDeviceToDevice));
		d_inside << <N - 2, N - 2 >> > (*u.cudamatrix, *u1.cudamatrix, *u2.cudamatrix);
		CUDACHECK(cudaDeviceSynchronize());
		d_border << <4, N - 2 >> > (*u.cudamatrix);
		CUDACHECK(cudaDeviceSynchronize());
		d_corner << <1, 4 >> > (*u.cudamatrix);
		CUDACHECK(cudaDeviceSynchronize());

		CUDACHECK(u.fromCuda());
		if (os) *os << u << endl;
	}
}

void h_sequential(const long t,const long N,ostream* os) {
	matrix<double> u(N, N);
	matrix<double> u1(N, N);
	matrix<double> u2(N, N);
	u[N / 2][N / 2] = 1;

	for (size_t _ = 0; _ < t; _++)
	{
		memcpy(u2.arr, u1.arr, u1.memsize());
		memcpy(u1.arr, u.arr, u1.memsize());

		for (size_t i = 1; i < N - 1; i++)
			for (size_t j = 1; j < N - 1; j++)
				normalCase;

		for (size_t i = 1; i < N - 1; i++)
		{
			int j = i;
			rightBoundary;
			leftBoundary;
			topBoundary;
			bottomBoundary;
		}

		topleft;
		topright;
		bottomleft;
		bottomright;

		if (os) *os << u << endl;
	}
	
}

void main(int argc,const char** argv) {

	ofstream time;
	time.open("result.csv");

	for (size_t i = 0; i < 30; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		h_sequential(3, 512, 0);
		auto seq = std::chrono::high_resolution_clock::now();
		h_parallel(3, 512, 0);
		auto par = std::chrono::high_resolution_clock::now();
		h_decomposition(3, 0);
		auto dec = std::chrono::high_resolution_clock::now();

		time
			<< i <<","
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(seq - start).count() << ","
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(par - seq).count() << ","
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(dec - par).count() << endl;
	}

	time.close();

	//to check if all 3 has same output, use excel to compare those 3 files
	/*ofstream seq;
	ofstream par;
	ofstream dec;
	seq.open("sequential.csv", ios::out);
	par.open("parallel.csv", ios::out);
	dec.open("decomposition.csv", ios::out);
	h_sequential(3, 512, &seq);
	h_parallel(3, 512,&par);
	h_decomposition(3, &dec);
	seq.close();
	par.close();
	dec.close();*/
}