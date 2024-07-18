# mandelCuda
Mandelbrot Cuda Core Programming Project.

Personal Notes for Cuda Programming:

// Workflow for Cuda Programming:

/*

	Use Wrapper to set:
		Device to use 
			--- cudaSetDevice ()
			For sys with many GPUs
		malloc arrays to use on GPU
			---   cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
		copy arrays to GPU

		Call Kernel
			KERNELDEF <<<numBlocks, numThreads>>> (inputArgs);
			KERNELDEF CUDA_KERNEL(numBlocks, numThreads) (inputArgs); 

		Check for kernel launch errors
			--- cudaStatus = cudaGetLastError();
		DeviceSync + check for errors			
			--- cudaSTatus = cudaDeviceSynchronize();
		Export arrays from GPU
									array pased from main, device array used in kernel
			--- cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		
		free arrays
			--- cudaFree(array);

		Check for errors when doing this
			--- if (cudaStatus != cudaSuccess) { // Handle }


		IN MAIN: 
		call cudaDeviceReset:
			--- cudaStatus = cudaDeviceReset();
*/

### Cuda Presentation
https://www3.nd.edu/~zxu2/acms60212-40212-S12/Lec-12-02.pdf

dim3 gridDim;
	Dimenstions of the grid in blocks (gridDim.z unused)
dim3 blockDim;
	Dimensions of the block in threads
dim3 blockIdx;
	Block index within the grid
dim3 threadIdx;
	Thread index within the box

Specifying 1D Block => Just straight up call the kernel with integer vals
e.g.
	kernel<<<3, 4>>>();

 Calls a Kernel of 3 blocks with 4 threads for each, organized in a 1d space:

 Block# | Thread#
 0 | 	0	1	2	3
 1 | 	0	1	2	3
 2 | 	0	1	2	3

threadIdx.x ranges from 0 - blockDim.x-1

dim3 nthreads(16, 4); 
	Defines threads in a 16 x 4 area in each block


### Calculating ThreadID
1d blocks, with 1d threads
int tid = threadIdx.x + blockDim.x * blockIdx.x;








