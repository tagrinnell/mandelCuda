
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>


#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#define X 1920
#define Y 1080

#define numBlocks 16 
#define numThreads 16 

struct Point {
    int x;
    int y;
    int iteration;
    int sizeX;
    int sizeY;
};

cudaError_t mandelBrotCalc(struct Point* pointArray, int* numIterations, unsigned long size);

__global__ void computeSet(struct Point* returnPointArr, int* numIterations) {
    int iterations = 0;

    for (int i = X / threadIdx.x; i < X && i < X / (threadIdx.x + 1); i++) {
        for (int j = Y / threadIdx.y; j < Y && j < Y / (threadIdx.y + 1); j++) {

            float x0 = i / (double)X * 2.47 - 2;
            float y0 = j / (double)Y * 2.24 - 1.12;
            float x = 0.0;
            float y = 0.0;

            int iteration = 0;
            int max_iteration = 1000;

            while (x * x + y * y <= (2 * 2) && iteration < max_iteration) {
                float xtemp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = xtemp;

                iteration++;
            }

            struct Point newPoint = { i, j, iteration, 0, 0 };

            returnPointArr[];

            iterations++;
        }

        iterations++;
    }

    numIterations[threadIdx.x / numThreads + threadIdx.y % 16] = iterations;
}

/*
*
* Unoptimized escape for calculating the Mandelbrot Set
* 
*/
int main()
{
    struct Point pointArray[X * Y];
    
    int* numIterations = (int*) malloc(sizeof(int) * numThreads * numBlocks);

    // Add vectors in parallel.
    cudaError_t cudaStatus = mandelBrotCalc(pointArray, numIterations, (unsigned long) X * Y );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::ofstream file("MandelSetOut_Parallel.csv");

    file << "X,Y,Iteration,sizeX,sizeY" << std::endl;

    for (int i = 0; i < X * Y; i++) {
        if (i == 0) {
            file << pointArray[i].x << "," << pointArray[i].y << "," << pointArray[i].iteration
                << X << ","
                << Y << "," << std::endl;
        }

        file << pointArray[i].x << "," << pointArray[i].y << "," << pointArray[i].iteration << std::endl;
    }

    // Output results

    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to calculate mandelBrot set in parallel.
cudaError_t mandelBrotCalc (struct Point pointArray[], int* numIterations, unsigned long size)
{
    struct Point *dev_points = 0;
    int* dev_iterations = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for output vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_points, size * sizeof(struct Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_iterations, numThreads * numBlocks * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Launch a kernel on the GPU with 16 threads for each element.
    // Num blocks, numThreads
    CUDA_KERNEL (16, 16) computeSet (dev_points, dev_iterations);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&pointArray, dev_points, size * sizeof(struct Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(&numIterations, dev_iterations, size * sizeof(struct Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_points);
    
    return cudaStatus;
}
