#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <cuda.h>

#include "../include/utils.h"

#define NUM_BLOCKS 128
#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK


//return is number of iterarions
__global__ void k_means(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,int *points,const int N,const int K){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		int lid = threadIdx.x; 

		//__shared__ int points[NUM_THREADS_PER_BLOCK]; 

		__shared__ float mean_x[10],mean_y[10]; //keep values to calculate new centroid

		//__shared__ int local_elem_cluster[K];

		__shared__ float local_arr_x[NUM_THREADS_PER_BLOCK],local_arr_y[NUM_THREADS_PER_BLOCK];

		local_arr_x[lid] = arr_x[id];
		local_arr_y[lid] = arr_y[id];

		//reset arrays
		for(int i = 0; i < K;i++){
				mean_x[i] = 0;	
				mean_y[i] = 0;
				//local_elem_cluster[i] = 0;
		}



		
		float dist[10]; 
		int ind = 0; //lower distance index
		for(int i = 0; i < K;i++){//calculate distance between points and centroids 
			float x = (cluster_x[i] - local_arr_x[lid]);
			float y = (cluster_y[i] - local_arr_y[lid]);
			dist[i]  = x * x;  
			dist[i] += y * y;
			DEBUG("cluster_x[%d]=%f | cluster_y[%d]=%f | local_arr_x[%d]=%f | local_arr_x[%d]=%f | dist[%d]=%f\n",i,cluster_x[i],i,cluster_y[i],i,local_arr_x[i],i,local_arr_y[i],i,dist[i]);
		}

		//find minimum value for the distance between centroid and point
		for(int i = 1; i < K;i++){
			ind = (dist[i] < dist[ind])?i:ind;
		}
		
		DEBUG("ind:%d\n",ind);

		points[id] = ind; // assigns the new lowest distance centroid to the point 








		/*
		if(threadIdx.x == 0){
			for(int i = 0; i < NUM_THREADS_PER_BLOCK; i++){
				int ind = points[i];

				mean_x[ind] += local_arr_x[i]; // add this point to the sum of points belonging to cluster
				mean_y[ind] += local_arr_y[i];
				local_elem_cluster[ind]++; //update number of elements in cluster
			}

			for(int i = 0;i < K;i++){

			}
		}*/
	}
}

int launch_kernel(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,int *n_elem_cluster,const int N,const int K,const int T){

	int ret = 0;

	const int size_clusters = K * sizeof(float);
	const int size_points = N * sizeof(float);
	int *points = (int*)malloc(N * sizeof(int));
	float mean_x[K],mean_y[K];

	float *d_cluster_x;
	float *d_cluster_y;
	float *d_arr_x;
	float *d_arr_y;
	int *d_points;
	int *d_n_elem_cluster;

	//allocate space in the device
	cudaMalloc((void**) &d_cluster_x, size_clusters);
	cudaMalloc((void**) &d_cluster_y, size_clusters);
	cudaMalloc((void**) &d_arr_x, size_points);
	cudaMalloc((void**) &d_arr_y, size_points);
	//cudaMalloc((void**) &d_n_elem_cluster, K * sizeof(int));
	cudaMalloc((void**) &d_points, N * sizeof(int));

	//copy data from host to device
	cudaMemcpy (d_cluster_x,cluster_x,size_clusters,cudaMemcpyHostToDevice);
	cudaMemcpy (d_cluster_y,cluster_y,size_clusters,cudaMemcpyHostToDevice);
	cudaMemcpy (d_arr_x,arr_x,size_points,cudaMemcpyHostToDevice);
	cudaMemcpy (d_arr_y,arr_y,size_points,cudaMemcpyHostToDevice);

	//call Kernel
	for(int ite = 0; ite < 20;ite++){
		startKernelTime ();
		k_means <<< N/NUM_THREADS_PER_BLOCK + 1, NUM_THREADS_PER_BLOCK >>> (d_cluster_x,d_cluster_y,d_arr_x,d_arr_y,d_points,N,K);
		stopKernelTime ();
		cudaMemcpy (points,d_points, N * sizeof(int),cudaMemcpyDeviceToHost);

		for(int i = 0;i < K;i++){
			mean_x[i] = 0; // add this point to the sum of points belonging to cluster
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}

		for(int i = 0; i < N; i++){
			int ind = points[i];
			DEBUG("point[%d]=%d\n",i,ind);
			mean_x[ind] += arr_x[i]; // add this point to the sum of points belonging to cluster
			mean_y[ind] += arr_y[i];
			n_elem_cluster[ind]++; //update number of elements in cluster

		}

		for(int i = 0; i < K;i++){
			cluster_x[i] = mean_x[i] / (n_elem_cluster[i]);
			cluster_y[i] = mean_y[i] / (n_elem_cluster[i]);
			DEBUG("new cluster_x[%d]=%f | cluster_y[%d]=%f | mean_x[%d]=%f |  mean_y[%d]=%f | elem[%d]=%d\n",i,cluster_x[i],i,cluster_y[i],i,mean_x[i],i,mean_y[i],i,n_elem_cluster[i]);
		}

		cudaMemcpy (d_cluster_x,cluster_x,size_clusters,cudaMemcpyHostToDevice);
		cudaMemcpy (d_cluster_y,cluster_y,size_clusters,cudaMemcpyHostToDevice);

		ret++;
	}

	//Retrive information from device
	//cudaMemcpy (cluster_x,d_cluster_x,size_clusters,cudaMemcpyDeviceToHost);
	//cudaMemcpy (cluster_y,d_cluster_y,size_clusters,cudaMemcpyDeviceToHost);
	//cudaMemcpy (n_elem_cluster,d_n_elem_cluster, K * sizeof(int),cudaMemcpyDeviceToHost);

	return ret;
}

int main(int argc, char const *argv[]){
	int N = atoi(argv[1]);
	int K = atoi(argv[2]);
	int T = atoi(argv[3]);

	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	arr_x = (float*) malloc(sizeof(float) * N);
	arr_y = (float*) malloc(sizeof(float) * N);

	init(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = launch_kernel(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster,N,K,T);

	free(arr_x);
	free(arr_y);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}