/**
 * Lorenzo Mario Amorosa - APAI 2020/21
 *
 * compile: nvcc neural_network_cuda.cu -o neural_network_cuda
 * run: ./neural_network_cuda 30000 3000
 **/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hpc.h"

#define R 3
#define BLKDIM 128

/*
 * Return a random float between -5 and 5
 */
float rnd_float()
{
	return (float) rand() / (float) (RAND_MAX/10.0) - 5.0;
}

/*
 * Return a random array of N elements
 */
float* rnd_array(int N)
{
	float *array = (float *)malloc(N * sizeof(float));
	int i;
	for(i = 0; i < N; i++) {
		array[i] = rnd_float();
	}
	return array;
}

/*
 * Return K-1 arrays of weights, each one for a distinct output layer
 */
float** generate_weights(int N, int K)
{
	float **weights = (float **)malloc((K - 1) * sizeof(float *));
	int i, rows = N - R + 1;
	for(i = 0; i < K - 1; i++) {
		weights[i] = rnd_array(rows * R);
		rows -= R - 1;
	}
	return weights;
}

__device__ float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

/*
 * This kernel function computes the output layer Y using the input layer X, the weights W 
 * and the bias b. The computation of a single output neuron is assigned to a distinct cuda core.
 * The inputs X are firstly moved to shared memory (considering halo as well), then it 
 * follows the computation of the output neuron, done by products between X and W, summed with b
 * and passed through the sigmoid.
 */
__global__ void compute_output_layer(float* X, int N, float* W, float b, float* Y) 
{
	__shared__ float shared_X[BLKDIM + R - 1];
	__shared__ float shared_b;
	const unsigned int gindex = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int lindex = threadIdx.x;
	float result = 0;
	int offset;

	if(gindex < N) {
		// copy X data from global to shared memory only for valid input
		shared_X[lindex] = X[gindex];
		// copy needed right valid halo
		if(lindex < R - 1 && gindex + BLKDIM < N) {
			shared_X[lindex + BLKDIM] = X[gindex + BLKDIM];
		}
		// copy the bias to shared memory
		if(lindex == 0) {
			shared_b = b;
		}
		
		__syncthreads(); // copy all data before computing Y

		// accumulate inputs and weights products and apply sigmoid only for valid output
		if(gindex < N - R + 1) {
			for(offset = 0; offset < R; offset++){
				result += shared_X[lindex + offset] * W[gindex * R + offset];
			}
			Y[gindex] = sigmoid(result + shared_b);
		}
	}
}

int main( int argc, char* argv[] )
{
	int N = 20000, K = 2000, i;
	float *h_X, **h_W, *h_b, *h_Y, *tmp;
	float *d_X, *d_W, *d_Y;
	double tstart, tstop;
	srand(time(NULL));

	if ( 3 == argc ) {
		N = atoi(argv[1]);
		K = atoi(argv[2]);
		if (N < 1 || K < 2 || N - (K - 1) * (R - 1) < 1) {
			printf("Illegal arguments: N=%d, K=%d\n", N, K);
			return EXIT_FAILURE;
		}
	}

	h_X = rnd_array(N);
	h_W = generate_weights(N, K);
	h_b = rnd_array(K - 1);

	cudaSafeCall(cudaMalloc((void **)&d_X, N * sizeof(float)));
	cudaSafeCall(cudaMalloc((void **)&d_W, (N - R + 1) * R * sizeof(float)));
	cudaSafeCall(cudaMalloc((void **)&d_Y, N * sizeof(float)));

	/*
	 * The for loop iterates K-1 times to compute the K-1 output layers. In order to allocate
	 * less memory, at each iteration (apart from the last one) d_X and d_Y are swapped, so that
	 * only the needed input and the current output are kept in device memory
	 */
	tstart = hpc_gettime();
	// copy input h_X to device memory
	cudaSafeCall(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
	for(i = 0; i < K - 1; i++) {
		// copy weights h_W[i] to device memory
		cudaSafeCall(cudaMemcpy(d_W, h_W[i], (N - R + 1) * R * sizeof(float), cudaMemcpyHostToDevice));
		// compute i-th output layer
		compute_output_layer<<<(N + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_X, N, d_W, h_b[i], d_Y);
		cudaCheckError(); // implicit cudaDeviceSynchronize()
		if(i < K - 2) {
			N -= R - 1;
			tmp = d_X;
			d_X = d_Y;
			d_Y = tmp;
		}
	}
	h_Y = (float *)malloc((N - R + 1) * sizeof(float));
	// copy last output layer d_Y to host memory
	cudaSafeCall(cudaMemcpy(h_Y, d_Y, (N - R + 1) * sizeof(float), cudaMemcpyDeviceToHost));
	tstop = hpc_gettime();
	printf("Elapsed time: %f\n", tstop - tstart);

	cudaSafeCall(cudaFree(d_X));
	cudaSafeCall(cudaFree(d_W));
	cudaSafeCall(cudaFree(d_Y));

	free(h_X);
	free(h_Y);
	free(h_b);
	for(i = 0; i < K - 1; i++) {
		free(h_W[i]);
	}
	free(h_W);

	return EXIT_SUCCESS;
}

