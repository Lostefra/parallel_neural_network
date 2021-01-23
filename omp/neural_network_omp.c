/**
 * Lorenzo Mario Amorosa - APAI 2020/21
 *
 * compile: gcc -std=c99 -Wall -Wpedantic -fopenmp neural_network_omp.c -o neural_network_omp -lm
 * run: OMP_NUM_THREADS=8 ./neural_network_omp 30000 3000
 **/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define R 3

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

float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

/*
 * This function computes the output layer Y using the input layer X, the weights W and the bias b.
 * The outer for loop iterates over Y: it is firstly initialized the current output cell Y_i, then
 * Y_i is updated with the products between X and W in the inner for loop, in the end Y_i is summed
 * with b and passed through the sigmoid function
 */
void compute_output_layer(float* X, int N, float* W, float b, float* Y)
{
        int i, j, out_dim = N - R + 1;

#pragma omp parallel for default(none) private(j) shared(X, W, Y, out_dim, b) schedule(static)
	for(i = 0; i < out_dim; i++) {
		Y[i] = 0.0;
		for(j = 0; j < R; j++) {
			Y[i] += X[i + j] * W[i * R + j];
		}
		Y[i] = sigmoid(Y[i] + b);
	}
}

int main( int argc, char* argv[] )
{
	int N = 20000, K = 2000, i;
	float *X, **W, *b, *Y, *tmp;
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

	X = rnd_array(N);
	W = generate_weights(N, K);
	b = rnd_array(K - 1);
	Y = (float *)malloc(N * sizeof(float));

	/*
	 * The for loop iterates K-1 times to compute the K-1 output layers. In order to allocate
	 * less memory, at each iteration (apart from the last one) X and Y are swapped, so that
	 * only the needed input and the current output are kept in memory
	 */
	tstart = omp_get_wtime();
	for(i = 0; i < K - 1; i++) {
		compute_output_layer(X, N, W[i], b[i], Y);
		if(i < K - 2) {
			tmp = X;
			X = Y;
			Y = tmp;
			N -= R - 1;
		}
	}
	tstop = omp_get_wtime();
	printf("Elapsed time: %f\n", tstop - tstart);

	free(X);
	free(Y);
	free(b);
	for(i = 0; i < K - 1; i++) {
		free(W[i]);
	}
	free(W);

	return EXIT_SUCCESS;
}

