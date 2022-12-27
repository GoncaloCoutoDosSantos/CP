#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <cuda.h>

#include "../include/utils.h"

#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK
#ifndef MAX
#define MAX 128
#endif


//return is number of iterarions
__global__ void k_means(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,float *r_mean_x,float *r_mean_y,int *r_n_elem_cluster,int *r_old_points,char * r_flag,const int N,const int K){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		int lid = threadIdx.x; 
		int mid = blockIdx.x * K;

		__shared__ int points[NUM_THREADS_PER_BLOCK]; 

		__shared__ float mean_x[MAX],mean_y[MAX]; //keep values to calculate new centroid

		__shared__ int local_elem_cluster[MAX];

		__shared__ float local_arr_x[NUM_THREADS_PER_BLOCK],local_arr_y[NUM_THREADS_PER_BLOCK];

		__shared__ float local_cluster_x[MAX],local_cluster_y[MAX];

		__shared__ char flag[NUM_THREADS_PER_BLOCK];

		local_arr_x[lid] = arr_x[id];
		local_arr_y[lid] = arr_y[id];
		int old_point = r_old_points[id];

		if(lid == 0){
			for(int i = 0; i < K;i++){
				local_cluster_x[i] = cluster_x[i];
				local_cluster_y[i] = cluster_y[i];
			}
		}

		if(lid == 1){
			for(int i = 0; i < K;i++){
				mean_x[i] = 0;	
				mean_y[i] = 0;
				local_elem_cluster[i] = 0;
			}
		}

		__syncthreads();

		
		float dist[MAX]; 
		int ind = 0; //lower distance index
		for(int i = 0; i < K;i++){//calculate distance between points and centroids 
			float x = (local_cluster_x[i] - local_arr_x[lid]);
			float y = (local_cluster_y[i] - local_arr_y[lid]);
			dist[i]  = x * x;  
			dist[i] += y * y;
			DEBUG("cluster_x[%d]=%f | cluster_y[%d]=%f | local_arr_x[%d]=%f | local_arr_x[%d]=%f | dist[%d]=%f\n",i,cluster_x[i],i,cluster_y[i],i,local_arr_x[i],i,local_arr_y[i],i,dist[i]);
		}

		//find minimum value for the distance between centroid and point
		for(int i = 1; i < K;i++){
			ind = (dist[i] < dist[ind])?i:ind;
		}
		
		//printf("ind:%d\n",ind);

		flag[lid] = (ind != old_point);
		points[lid] = ind; // assigns the new lowest distance centroid to the point 

		__syncthreads();
		//printf("lid = %d | mid = %d\n",lid,mid);
		if(lid == 0){
			r_flag[blockIdx.x] = false;

			for(int i = 0; i < NUM_THREADS_PER_BLOCK && id + i < N; i++){
				int ind = points[i];

				mean_x[ind] += local_arr_x[i]; // add this point to the sum of points belonging to cluster
				mean_y[ind] += local_arr_y[i];
				local_elem_cluster[ind]++; //update number of elements in cluster
				r_flag[blockIdx.x] |= flag[i];
				//printf("%d\n",local_elem_cluster[ind]);
			}
			for(int i = 0;i < K;i++){
				r_mean_x[mid + i] = mean_x[i];
				r_mean_y[mid + i] = mean_y[i];
				r_n_elem_cluster[mid + i] = local_elem_cluster[i];
				//printf("mean_x[%d][%d]=%f | mean_y[%d][%d]=%f | elem[%d][%d]=%d\n",mid,i,mean_x[i],mid,i,mean_y[i],mid,i,local_elem_cluster[i]);
			}
		}
		r_old_points[id] = ind;
	}
}

int launch_kernel(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,int *n_elem_cluster,const int N,const int K){

	int ret = -1;

	const int size_clusters = K * sizeof(float);
	const int size_points = N * sizeof(float);
	const int n_blocks = N/NUM_THREADS_PER_BLOCK + 1;
	
	float *mean_x,*mean_y;
	int *aux;
	char flags[n_blocks];

	mean_x = (float *) malloc(K * n_blocks * sizeof(float));
	mean_y = (float *) malloc(K * n_blocks * sizeof(float));
	aux = (int *) malloc(K * n_blocks * sizeof(int));

	float *d_cluster_x;
	float *d_cluster_y;
	float *d_arr_x;
	float *d_arr_y;
	int   *d_old_points;
	int   *d_n_elem_cluster;
	float *d_mean_x,*d_mean_y;
	char  *d_flag;


	//allocate space in the device
	cudaMalloc((void**) &d_cluster_x, size_clusters);
	cudaMalloc((void**) &d_cluster_y, size_clusters);
	cudaMalloc((void**) &d_arr_x, size_points);
	cudaMalloc((void**) &d_arr_y, size_points);
	cudaMalloc((void**) &d_old_points,N * sizeof(int));
	cudaMalloc((void**) &d_n_elem_cluster, K * sizeof(int) * n_blocks);
	cudaMalloc((void**) &d_mean_x, K * sizeof(float) * n_blocks);
	cudaMalloc((void**) &d_mean_y, K * sizeof(float) * n_blocks);
	cudaMalloc((void**) &d_flag, sizeof(char) * n_blocks);

	//copy data from host to device
	cudaMemcpy (d_cluster_x,cluster_x,size_clusters,cudaMemcpyHostToDevice);
	cudaMemcpy (d_cluster_y,cluster_y,size_clusters,cudaMemcpyHostToDevice);
	cudaMemcpy (d_arr_x,arr_x,size_points,cudaMemcpyHostToDevice);
	cudaMemcpy (d_arr_y,arr_y,size_points,cudaMemcpyHostToDevice);

	char flag = true;

	//call Kernel
	while (flag){
		//startKernelTime ();
		k_means <<< N/NUM_THREADS_PER_BLOCK + 1, NUM_THREADS_PER_BLOCK >>> (d_cluster_x,d_cluster_y,d_arr_x,d_arr_y,d_mean_x,d_mean_y,d_n_elem_cluster,d_old_points,d_flag,N,K);
		cudaDeviceSynchronize();
		//stopKernelTime ();
		
		cudaMemcpy (mean_x,d_mean_x, K * sizeof(float) * n_blocks,cudaMemcpyDeviceToHost);
		cudaMemcpy (mean_y,d_mean_y, K * sizeof(float) * n_blocks,cudaMemcpyDeviceToHost);
		cudaMemcpy (aux,d_n_elem_cluster, K * sizeof(int) * n_blocks,cudaMemcpyDeviceToHost);
		cudaMemcpy (flags,d_flag, sizeof(char) * n_blocks,cudaMemcpyDeviceToHost);

		flag = flags[0];

		for(int i = 1; i < n_blocks; i++){
			for(int j = 0; j < K;j++){
				mean_x[j] += mean_x[j + i*K]; // add this point to the sum of points belonging to cluster
				mean_y[j] += mean_y[j + i*K];
				aux[j] += aux[j + i*K]; //update number of elements in cluster
			}
			flag |= flags[i];
		}

		for(int i = 0; i < K;i++){
			cluster_x[i] = mean_x[i] / (aux[i]);
			cluster_y[i] = mean_y[i] / (aux[i]);
			//printf("new cluster_x[%d]=%f | cluster_y[%d]=%f | mean_x[%d]=%f |  mean_y[%d]=%f | elem[%d]=%d\n",i,cluster_x[i],i,cluster_y[i],i,mean_x[i],i,mean_y[i],i,n_elem_cluster[i]);
		}

		cudaMemcpy (d_cluster_x,cluster_x,size_clusters,cudaMemcpyHostToDevice);
		cudaMemcpy (d_cluster_y,cluster_y,size_clusters,cudaMemcpyHostToDevice);

		ret++;
		//printf("flag: %d | ret: %d\n",flag,ret);
	}

	

	for(int i = 0;i < K;i++){
		n_elem_cluster[i] = aux[i];
	}

	cudaFree((void**) &d_cluster_x);
	cudaFree((void**) &d_cluster_y);
	cudaFree((void**) &d_arr_x);
	cudaFree((void**) &d_arr_y);
	cudaFree((void**) &d_old_points);
	cudaFree((void**) &d_n_elem_cluster);
	cudaFree((void**) &d_mean_x);
	cudaFree((void**) &d_mean_y);
	cudaFree((void**) &d_flag);


	free(mean_x);
	free(mean_y);
	free(aux);

	return ret;
}

int main(int argc, char const *argv[]){
	int N = atoi(argv[1]);
	int K = atoi(argv[2]);

	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	arr_x = (float*) malloc(sizeof(float) * N);
	arr_y = (float*) malloc(sizeof(float) * N);

	printf("%d",MAX);

	init(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = launch_kernel(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster,N,K);

	free(arr_x);
	free(arr_y);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}