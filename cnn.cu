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
#include <math.h>

void swap (int &i);
int read_int(int fd);
void output_pgm(const char *fn, const float (&img)[28][28]);
template <int N>
void read_mnist_images(const std::string &fn, float (&imgs)[N][28][28]);
template <int N>
void read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]);

typedef float img[28][28];
typedef float img_5[5][5];
typedef float img_7[14][14];

__global__ void kernel (img * images, float * CL_input, int p_cli, float * DL_filters, int p_dlf, 
			img_5 * CL_filters, img * CL_output, img_7 * PL_output, unsigned char * labels, 
			img_7 * deriv_pl_dl, float * dl_filter_deriv, int p_dlfd, img * deriv_cl_pl,
			img_5 * cl_filter_deriv)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, x, y, index_x, index_y, counter = 0;
	const int dl_size = 14 * 14 * 32; // 6272
	const float rate = 0.01;
	const int batch = 100;
	const int training_size = 1000;
	curandState_t state;
	curand_init(0, id, 0, &state);
	__shared__ float classes[10];
	__shared__ float dl_bias[10];
	__shared__ float cl_bias[32];
	__shared__ float softmax_vec[10];
	__shared__ float deriv_softmax_cross[10];
	__shared__ float deriv_dl_softmax[10];
	__shared__ float dl_bias_deriv[10];
	__shared__ float cl_bias_deriv[10];
	int mul = 32 * 14;
	float _class, res, max, current_data;
	//float nom = 1, denom = 0;

	// Initialize DL and CL biases, derivatives
        cl_bias[id] = 0;	
	if (id < 10) 
	{
		dl_bias[id] = 0;
		dl_bias_deriv[id] = 0;
		cl_bias_deriv[id] = 0;
	}

	// Initialize CL1 filters
	for (i = 0; i < 5; i++) for (j = 0; j < 5; j++) CL_filters[i][j][id] = ((curand_uniform(&state) * 2) - 1) / 25; 

	/*
	for (i = 0; i < 5; i++) for (j = 0; j < 5; j++) 
	{
		if (id % 2) CL_filters[i][j][id] = curand_uniform(&state) / 25;
		else CL_filters[i][j][id] = -curand_uniform(&state) / 25;
	}
	*/

	// Check filter 0
	/*
	if (id == 0)
	{
		printf("\nFilter:\n");		
		for (i = 0; i < 5; i++) 
		{
			for (j = 0; j < 5; j++) printf("%.2f ", CL_filters[i][j][id]);
			printf("\n");
		}
	}
	*/

	// Initialize DL1 filters
	if (id < 10) for (i = 0; i < dl_size; i++) 
	{
		float * row = (float *) ((char *) DL_filters + i * p_dlf);

		//if (id % 2) row[id] = curand_uniform(&state);
		//else row[id] = -curand_uniform(&state);

		row[id] = ((curand_uniform(&state) * 2) - 1);
	}

	// Core training images loop
	for (counter = 0; counter < training_size; counter++)
	{
		if (id == 0 && counter % batch == 0) printf("------------ Counter: %d -----------\n", counter);
		// Initialize CL1 input - first 28x28 image
		if (id < 28) for (i = 0; i < 28; i++) 
		{
			float * row = (float *) ((char *) CL_input + id * p_cli);
			row[i] = images[counter][id][i];
		}
		__syncthreads();
		
		/*
		if (id == 0)
		{
			printf("\nImage:\n");		
			for (i = 0; i < 28; i++) 
			{
				for (j = 0; j < 28; j++) 
				{
					float * row = (float *) ((char *) CL_input + i * p_cli);
					float p = row[j];
					printf("%.2f ", p);
				}		
				printf("\n");
			}
		}
		*/
		
		/*
		if (id == 0)
		{
			printf("\nCL Filter:\n");		
			for (i = 0; i < 5; i++) 
			{
				for (j = 0; j < 5; j++) printf("%f ", CL_filters[i][j][id]);
				printf("\n");
			}
		}
		*/
		/*
		if (id == 0)
		{
			int filter_no = 1;
			printf("\nDL Filter %d:\n", filter_no);		
			for (i = 0; i < 14; i++) 
			{
				for (j = 0; j < 14; j++) 
				{
					int r = i * mul + j * 32 + id;
					float * _row = (float *) ((char *) DL_filters + r * p_dlf);
						
					printf("%.3f ", _row[filter_no]);
				}
				printf("\n");
			}
		}
		if (id == 0) printf("\n\n");
		*/

		// Convolution Layer 1
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
					
						if (index_y < 0 || index_x < 0 || index_x > 27 || index_y > 27) continue;

						float * row = (float *) ((char *) CL_input + index_x * p_cli);
						float elem = row[index_y]; 
					
						res += CL_filters[i][j][id] * elem; 
					}
				}

				float _out = res + cl_bias[id];

				if (_out > 0) CL_output[x][y][id] = _out;
				else CL_output[x][y][id] = 0;
			}
		}
		__syncthreads();
		// Check output of CL
		/*
		if (id == 0)
		{
			printf("\nOutput of CL:\n");		
			for (i = 0; i < 28; i++) 
			{
				for (j = 0; j < 28; j++) printf("%.1f ", CL_output[i][j][id]);
				printf("\n");
			}
		}
		*/

		// Pooling Layer
		for (x = 0; x < 14; x++)
		{
			for (y = 0; y < 14; y++)
			{
				// Or is that supposed to be 0 ?? Because of Relu but PL does not have Relu !!
				max = CL_output[x * 2][y * 2][id];

				for (i = 0; i < 2; i++)
				{
					for (j = 0; j < 2; j++)
					{
						index_x = x * 2 + i;
						index_y = y * 2 + j;
					
						current_data = CL_output[index_x][index_y][id];
						if (current_data > max) max = current_data;
					}
				}

				PL_output[x][y][id] = max;
			}
		}
		__syncthreads();
		// Check output of PL
		/*
		if (id == 0)
		{
			printf("\nOutput of PL:\n");		
			for (i = 0; i < 14; i++) 
			{
				for (j = 0; j < 14; j++) printf("%.1f ", PL_output[i][j][id]);
				printf("\n");
			}
		}
		*/
	
		// Dense Layer
		if (id < 10)
		{
			_class = 0;
			for (i = 0; i < 14; i++)
			{
				for (j = 0; j < 14; j++)
				{
					for (x = 0; x < 32; x++)
					{
						int r = i * mul + j * 32 + x;
						float * row = (float *) ((char *) DL_filters + r * p_dlf);
						float elem = row[id];
					
						//if (id == 0 && x == 0 && i == 0 && j == 0) printf("\nPoints:\n");
						//if (id == 0 && x == 0) printf("%f ", PL_output[i][j][0]);
						
						float point = elem * PL_output[i][j][x];
						
						// Or is Relu here ? !!!!
						//if (id == 1 && point > 0) printf("%f ", point);
						//if (point > 0) _class += point;
						_class += point;
					}
				}
			}
			
			// Relu of Dense Layer - Maybe !!!
			//if (_class > 0)
			{
				float out = _class + dl_bias[id];
				
				if (out > 0) classes[id] = out; 
				else classes[id] = 0; 
				
				//if (id == 0) printf("%f, %f", _class, dl_bias[id]);
				//printf("class %d: %f\n", id, _class);
			}
		}
		__syncthreads();
		// Print classes
		//if (id < 10) printf("classes[%d]: %f\n", id, classes[id]);
		//if (id == 0) printf("\nLabel: %d\n", labels[counter]);

		if (counter % batch == 0)
		{
			if (id == 0) printf("Classes: \n");
			if (id == 0) for (i = 0; i < 10; i++) printf("%f ", classes[i]);
			if (id == 0) printf("\n");
		}

		// Softmax 
		/*
		if (id < 10)
		{
			denom = 0;
			nom = powf(M_E, classes[id]);

			if (counter % batch == 0)
			{
				if (id == 0) printf("Classes: \n");
				if (id == 0) for (i = 0; i < 10; i++) printf("%f ", classes[i]);
				if (id == 0) printf("\n");
			}

			// !!!! BUG: classes become huge here so powf results inf and nan !!!!
			for (i = 0; i < 10; i++) 
			{
				//if (id == 0) printf("\nDenom: %f", denom);
				denom += powf(M_E, classes[i]);
			}
	
			//if (id == 0) printf("Nom - Denom:\n%f, %f\n", nom, denom);		
			softmax_vec[id] = nom / denom;
		}
		*/

		if (id == 0)
		{
			float s_max = 0;
			for (i = 0; i < 10; i++) if (classes[i] > s_max) s_max = classes[i];
		
			float s_sum = 0;

			for (i = 0; i < 10; i++)
			{
				softmax_vec[i] = powf(M_E, classes[i] - s_max);
				s_sum += softmax_vec[i];
			}

			for (i = 0; i < 10; i++) softmax_vec[i] = softmax_vec[i] / s_sum;
		}
		__syncthreads();
		
		// Print softmax
		/*
		if (counter % batch == 0)
		{
			if (id == 0) printf("Softmax:\n");
			if (id == 0) for (i = 0; i < 10; i++) printf("%f ", softmax_vec[i]);
			if (id == 0) printf("\n");
		}
		*/

		// Cross Entropy
		// Loss is not really necessary, just to debug
		if (id == 0 && (counter % batch) == 0)
		{
			//printf("Softmax[correct_label]: %f\n", softmax_vec[labels[counter]]);
			//float loss = -1 * logf(softmax_vec[labels[counter]]);
			//printf("Loss: %f\n", loss);
		}

		// Back Propagation Starts from Cross Entropy
		//if (id == 0) printf("\n\nSoftmax vec:\n");
		//if (id < 10) printf("%f ", softmax_vec[id]);
		
		if (id < 10) deriv_softmax_cross[id] = 0;
		__syncthreads();	
		if (id == 0) deriv_softmax_cross[labels[counter]] = -1 / softmax_vec[labels[counter]];
		__syncthreads();

		// Back Propagation - Softmax
		if (id < 10)
		{
			deriv_dl_softmax[id] = 0;
			//if (id == 0) printf("deriv softmax cross:\n");

			for (i = 0; i < 10; i++)
			{
				//if (id == 0) printf("%f ", deriv_softmax_cross[i]);

				if (i == id) deriv_dl_softmax[id] += deriv_softmax_cross[i] *
								     (softmax_vec[i] * (1 - softmax_vec[id]));
				else deriv_dl_softmax[id] += deriv_softmax_cross[i] *
							     (-softmax_vec[id] * softmax_vec[i]);	
			}

			//if (id == 0) printf("\n%f\n", deriv_dl_softmax[id]);
		}
		__syncthreads();

		// Back Propagation - DL
		for (i = 0; i < 14; i++) for (j = 0; j < 14; j++) deriv_pl_dl[i][j][id] = 0;
		__syncthreads();
		
		//if (id == 0) printf("deriv dl softmax:\n");
		for (x = 0; x < 10; x++)
		{
			//if (id == 0) printf("%f ", deriv_dl_softmax[x]);
			//if (id == 0 && x == 9) printf("\n\n");

			//if (classes[x] != 0)
			if (classes[x] > 0)
			{
				//if (id == 0) printf("%f ", deriv_dl_softmax[x]);
				
				for (i = 0; i < 14; i++)
				{
					for (j = 0; j < 14; j++)
					{
						int r = i * mul + j * 32 + id;
						float * row = (float *) ((char *) DL_filters + r * p_dlf);
						float elem = row[x];


						//if (id == 0 && x == 0) printf("%f ", elem);
						
						//if (id == 0) printf("deriv_pl_dl: %f\n", deriv_pl_dl[i][j][id]);
						
						deriv_pl_dl[i][j][id] += deriv_dl_softmax[x] * elem;
						//if (id == 0) printf("%f ", deriv_pl_dl[i][j][id]);
						
						float * _row = (float *) ((char *) dl_filter_deriv + r * p_dlfd);
						_row[x] += deriv_dl_softmax[x] * PL_output[i][j][id] / batch;		
					}
				}
				
				if (id == 0)
				{
					//printf("deriv_dl_softmax: %f\n", deriv_dl_softmax[x]);
					dl_bias_deriv[x] += deriv_dl_softmax[x] / batch;
				}
			}
		}
		__syncthreads();
		//if (id == 0) for (i = 0; i < 14; i++) for (j = 0; j < 14; j++) printf("%f ", deriv_pl_dl[i][j][id]);

		// Update filter and bias of DL
		if (counter != 0 && counter % batch == 0)
		{
			for (x = 0; x < 10; x++)
			{
				for (i = 0; i < 14; i++)
				{
					for (j = 0; j < 14; j++)
					{
						int r = i * mul + j * 32 + id;
						float * row = (float *) ((char *) dl_filter_deriv + r * p_dlfd);
						float elem = row[x];
							
						//if (id == 0 && x == 0 && i == 0 && j == 0) printf("dl filter deriv:\n");

						// DL filters corrupt so will trash classes !!!!!!
						float * _row = (float *) ((char *) DL_filters + r * p_dlf);
						
						//if (id == 0 && x == 0) printf("%f ", elem);
						
						_row[x] -= rate * elem;

						row[x] = 0;
					}
				}
				
				if (id == 0)
				{
					float update = rate * dl_bias_deriv[x];
					//printf("Update of dl_bias_deriv: %f\n", dl_bias_deriv[x]);
					//printf("Update of dl_bias: %f\n", update);
					//printf("DL_bias before: %f\n", dl_bias[x]);
					dl_bias[x] -= update;
					//printf("DL_bias after: %f\n\n", dl_bias[x]);
					dl_bias_deriv[x] = 0;
				}	
			}
		}

		// Remove sync
		//__syncthreads();
		//if (id == 0)
		{
			//printf("\nDL Filter:\n");		
			//for (i = 0; i < 14; i++) 
			{
				//for (j = 0; j < 14; j++) 
				{
					//int r = i * mul + j * 32 + id;
					//float * _row = (float *) ((char *) DL_filters + r * p_dlf);		
					//printf("%f ", _row[0]);
				}
			}
		}


		// Back Propagation - PL
		for (i = 0; i < 14; i++)
		{
			for (j = 0; j < 14; j++)
			{
				max = PL_output[i][j][id];
				//if (id == 0) printf("%f ", deriv_pl_dl[i][j][id]);
				
				for (x = 0; x < 2; x++)
				{
					for (y = 0; y < 2; y++)
					{
						index_x = i * 2 + x;	
						index_y = j * 2 + y;

						if (CL_output[index_x][index_y][id] == max)	
							deriv_cl_pl[index_x][index_y][id] = deriv_pl_dl[i][j][id];
						else deriv_cl_pl[index_x][index_y][id] = 0; 
						
						//if (id == 0) printf("%f ", deriv_cl_pl[index_x][index_y][id]);
					}
				}
			}
		}
		__syncthreads();

		// Back Propagation - CL
		for (i = 0; i < 28; i++)
		{
			for (j = 0; j < 28; j++)
			{
				if (CL_output[i][j][id] > 0)
				{
					for (x = 0; x < 5; x++)
					{
						for (y = 0; y < 5; y++)
						{
							index_x = x + i - 2;
							index_y = y + j - 2;
					
							if (index_y < 0 || index_x < 0 || index_x > 27 || index_y > 27) continue;

							float * row = (float *) ((char *) CL_input + index_x * p_cli);
							float elem = row[index_y];

							cl_filter_deriv[x][y][id] += (elem * deriv_cl_pl[i][j][id]) / batch;	
						}
					}

					//if (id == 0) printf("deriv cl pl: %f\n", deriv_cl_pl[i][j][id]);
					cl_bias_deriv[id] += deriv_cl_pl[i][j][id] / batch;
				}
			}
		}
		//if (id == 0) printf("\n");
		__syncthreads();

		// Update CL filter and bias
		if (counter != 0 && counter % batch == 0)
		{
			//if (id == 0) printf("CL bias deriv: %f\n", cl_bias_deriv[id]);
			cl_bias[id] -= rate * cl_bias_deriv[id];
			cl_bias_deriv[id] = 0;
			
			for (i = 0; i < 5; i++)
			{
				for (j = 0; j < 5; j++)
				{
					//if (id == 0) printf("CL filter deriv: %f\n", cl_filter_deriv[i][j][id]);
					CL_filters[i][j][id] -= rate * cl_filter_deriv[i][j][id];
					cl_filter_deriv[i][j][id] = 0;	
				}
			}
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

	static unsigned char training_labels[60000];
	read_mnist_labels("train-labels-idx1-ubyte", training_labels);
	assert(training_labels[0] == 5);
	assert(training_labels[59999] == 8);

	//static float test_images[10000][28][28];
	//read_mnist_images("t10k-images-idx3-ubyte", test_images);
	//static unsigned char test_labels[10000];
	//read_mnist_labels("t10k-labels-idx1-ubyte", test_labels);

	img * d_input;
	e = cudaMalloc((void **) &d_input, (60000 * 28 * 28) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMemcpy(d_input, training_images, (60000 * 28 * 28) * sizeof(float), cudaMemcpyHostToDevice); assert(e == cudaSuccess);
	
	unsigned char * d_labels;
	e = cudaMalloc(&d_labels, 60000 * sizeof(unsigned char)); assert(e == cudaSuccess);
	e = cudaMemcpy(d_labels, training_labels, 60000 * sizeof(unsigned char), cudaMemcpyHostToDevice); assert(e == cudaSuccess);

	// Number of threads in each block and number of thread blocks in grid
	int t_block = 32;
	int b_grid = 1;

	float * cl_input;
	size_t pitch_cli;
	float * dl_filters;
	size_t pitch_dlf;
	float * dl_filter_deriv;
	size_t pitch_dlfd;
	
	e = cudaMallocPitch((void **) &cl_input, &pitch_cli, 28 * sizeof(float), 28); assert(e == cudaSuccess);
	e = cudaMallocPitch((void **) &dl_filters, &pitch_dlf, 6272 * sizeof(float), 10); assert(e == cudaSuccess);
	e = cudaMallocPitch((void **) &dl_filter_deriv, & pitch_dlfd, 6272 * sizeof(float), 10); assert(e == cudaSuccess);

	img_5 * cl_filters;
	img * cl_output;
	img_7 * pl_output;
	img_7 * deriv_pl_dl;
	img * deriv_cl_pl;
	img_5 * cl_filter_deriv;

	e = cudaMalloc((void **) &cl_filters, (5 * 5 * 32) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMalloc((void **) &cl_output, (28 * 28 * 32) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMalloc((void **) &pl_output, (14 * 14 * 32) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMalloc((void **) &deriv_pl_dl, (14 * 14 * 32) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMalloc((void **) &deriv_cl_pl, (28 * 28 * 32) * sizeof(float)); assert(e == cudaSuccess);
	e = cudaMalloc((void **) &cl_filter_deriv, (5 * 5 * 32) * sizeof(float)); assert(e == cudaSuccess);

	// Start the GPU
	std::cout << "\n";
	kernel<<<b_grid, t_block>>>(d_input, cl_input, pitch_cli, dl_filters, pitch_dlf, cl_filters, 
				    cl_output, pl_output, d_labels, deriv_pl_dl, dl_filter_deriv, 
				    pitch_dlfd, deriv_cl_pl, cl_filter_deriv);
	assert(cudaPeekAtLastError() == cudaSuccess);
	
	e = cudaDeviceSynchronize();
	//std::cout << e << std::endl;
	assert(e == cudaSuccess);
	std::cout << "\n";

	// Release device memory
	cudaFree(d_input);
	cudaFree(dl_filters);
	cudaFree(cl_input);
	cudaFree(cl_filters);
	cudaFree(cl_output);
	cudaFree(pl_output);
	cudaFree(deriv_pl_dl);
	cudaFree(dl_filter_deriv);
	cudaFree(deriv_cl_pl);
	cudaFree(cl_filter_deriv);
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
