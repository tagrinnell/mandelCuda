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