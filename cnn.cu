#include <curand_kernel.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <algorithm>

void swap (int &i);
int read_int(int fd);
void output_pgm(const char *fn, const float (&img)[28][28]);
template <int N>
void read_mnist_images(const std::string &fn, float (&imgs)[N][28][28]);
template <int N>
void read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]);

typedef float img[28][28];

__global__ void kernel (img * images)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, x, y;
	curandState_t state;
	curand_init(0, id, 0, &state);

	__shared__ float CL1_input[28][28];
	__shared__ float CL1_output[28][28][32];
	__shared__ float CL1_filters[5][5][32];

	// Initialize CL1 filters
	for (i = 0; i < 5; i++) for (j = 0; j < 5; j++) CL1_filters[i][j][id] = ((curand_uniform(&state) * 2) - 1) / 25; 
	if (id == 0)
	{
		printf("\nFilter:\n");		
		for (i = 0; i < 5; i++) 
		{
			for (j = 0; j < 5; j++) printf("%.2f ", CL1_filters[i][j][id]);
			printf("\n");
		}
	}
	
	// Initialize CL1 input - first 28x28 image
	if (id < 28) for (i = 0; i < 28; i++) CL1_input[id][i] = images[0][id][i];
	if (id == 0)
	{
		printf("\nInput:\n");		
		for (i = 0; i < 28; i++) 
		{
			for (j = 0; j < 28; j++) printf("%.2f ", CL1_input[i][j]);
			printf("\n");
		}
	}

	__syncthreads();	

	float res = 0;
	int index_x, index_y;

	for (x = 0; x < 28; x++)
	{
		for (y = 0; y < 28; y++)
		{
			res = 0;

			for (i = 0; i < 5; i++)
			{
				for (j = 0; j < 5; j++)
				{
					index_x = x + i - 2;
					index_y = y + j - 2;
					
					if (index_y < 0 || index_x < 0) continue;
					//res = res + CL1_filters[i][j][id] * CL1_input[index_x][index_y]; 
				}
			}

			//CL1_output[x][y][id] = res;
		}
	}

	__syncthreads();

	if (id == 0)
	{
		printf("\nOutput:\n");		
		for (i = 0; i < 28; i++) 
		{
			//for (j = 0; j < 28; j++) printf("%.2f ", CL1_output[i][j][id]);
			printf("\n");
		}
	}	
}

int main ()
{
	cudaError_t e;
	static float training_images[60000][28][28];
    	read_mnist_images("train-images-idx3-ubyte", training_images);
        //output_pgm("img0.pgm", training_images[0]);
	//output_pgm("img59999.pgm", training_images[59999]);

	//static unsigned char training_labels[60000];
	//read_mnist_labels("train-labels-idx1-ubyte", training_labels);
	//assert(training_labels[0] == 5);
	//assert(training_labels[59999] == 8);

	//static float test_images[10000][28][28];
	//read_mnist_images("t10k-images-idx3-ubyte", test_images);
	//static unsigned char test_labels[10000];
	//read_mnist_labels("t10k-labels-idx1-ubyte", test_labels);

	img * d_input;
	e = cudaMalloc((void **) &d_input, (60000 * 28 * 28) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMemcpy(d_input, training_images, (60000 * 28 * 28) * sizeof(float), cudaMemcpyHostToDevice); assert(e == cudaSuccess);
	
	// Number of threads in each block
	int t_block = 32;

	// Number of thread blocks in grid
	int b_grid = 1;

	// Start the GPU
	kernel<<<b_grid, t_block>>>(d_input);
	assert(cudaPeekAtLastError() == cudaSuccess);
	assert(cudaDeviceSynchronize() == cudaSuccess);

	// Release device memory
	cudaFree(d_input);
	return 0;
}

template <int N>
void
read_mnist_labels(const std::string &fn, unsigned char (&labels)[N])
{
	int rv, fd;
	fd = open(fn.c_str(), O_RDONLY);
	assert(fd >= 0);
	
	int magic = read_int(fd);
        assert(magic == 0x801);

	int n_labels = read_int(fd);
        assert(n_labels == N);

	rv = read(fd, labels, N); assert(rv == N);
	for (int i = 0; i < N; i++) assert(labels[i] <= 9);
	rv = close(fd); assert(rv == 0);
}

template <int N>
void
read_mnist_images(const std::string &fn, float (&imgs)[N][28][28])
{
	int rv, fd;
	fd = open(fn.c_str(), O_RDONLY);
	assert(fd >= 0);

        int magic = read_int(fd);
	assert(magic == 0x803);
	
	int n_images = read_int(fd);
        assert(n_images == N);

        int n_rows = read_int(fd);
	assert(n_rows == 28);

        int n_cols = read_int(fd);
	assert(n_cols == 28);

	for (int i = 0; i < N; i++) 
	{
		unsigned char tmp[28][28];
	        rv = read(fd, tmp, 28*28); assert(rv == 28*28);
	        for (int r = 0; r < 28; r++) 
		{
			for (int c = 0; c < 28; c++) 
			{
	                	// Make go from -1 to 1.
          	                imgs[i][r][c] = double(tmp[r][c])/127.5 - 1;
	                }
	        }
        }

	rv = close(fd); assert(rv == 0);
}

void swap (int &i) 
{
	// Some of the & are superfluous.
	i =
        (0xff&(i >> 24)) |
        (0xff00&(i >> 8)) |
        (0xff0000&(i << 8)) |
        (0xff000000&(i << 24));
}

int read_int(int fd) 
{
	int rv;
	int i;
	rv = read(fd, &i, 4); assert(rv == 4);
        swap(i);
	return i;
}

void output_pgm(const char *fn, const float (&img)[28][28]) 
{
	std::ofstream ofs(fn, std::fstream::out|std::fstream::trunc);

        ofs << "P2\n";
    	ofs << "28 28\n";
        ofs << "255\n";
	for (int i = 0; i < 28; i++) 
	{
		for (int j = 0; j < 28; j++) 
		{
                	if (j > 0) ofs << " ";
         	        ofs << 255 - int(std::ceil(127.5*(img[i][j] + 1)));
		}						            
	ofs << "\n";
        }
}

