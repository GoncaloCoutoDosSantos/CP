#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#include "../include/utils.h"

//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,int *n_elem_cluster,const int N,const int K,const int T){
	int *points; // arrays that save the new and previous allocation of points to clusters
	int ret = 0; //keep number of iterations(starts at -1 not considering the setup iteration)

	points = malloc(sizeof(int) * N);
	float mean_x[K],mean_y[K]; //keep values to calculate new centroid

	for(int ite = 0; ite < 20;ite++){
		//reset arrays
		for(int i = 0; i < K;i++){
				mean_x[i] = 0;	
				mean_y[i] = 0;
				n_elem_cluster[i] = 0;
		}


		#pragma omp parallel
		{

			#pragma omp for
			//primary loop that assigns points to clusters
			for(int i = 0; i < N;i++){ 
				float dist[K]; //auxiliar vector to calculate distance between centroid and a point 
				int ind = 0; //lower distance index
				for(int j = 0; j < K;j++){//calculate distance between points and centroids 
					float x = (cluster_x[j] - arr_x[i]);
					float y = (cluster_y[j] - arr_y[i]);
					dist[j] = x * x;  
					dist[j] += y * y;
				}

				//find minimum value for the distance between centroid and point
				for(int j = 1; j < K;j++){
					ind = (dist[j] < dist[ind])?j:ind;
				}
				
				points[i] = ind; // assigns the new lowest distance centroid to the point 
			}

			#pragma omp for reduction(+:mean_x[:K]) reduction(+:mean_y[:K]) reduction(+:n_elem_cluster[:K])
			for(int i = 0; i < N; i++){
				int ind = points[i];

				mean_x[ind] += arr_x[i]; // add this point to the sum of points belonging to cluster
				mean_y[ind] += arr_y[i];
				n_elem_cluster[ind]++; //update number of elements in cluster

			}

		}
		
			
		//new centroids calculations 
		for(int i = 0; i < K;i++){
			cluster_x[i] = mean_x[i] / (n_elem_cluster[i]);
			cluster_y[i] = mean_y[i] / (n_elem_cluster[i]);
		}

		ret++;

	}



	free(points);


	return ret;
}

int main(int argc, char const *argv[]){
	int N = atoi(argv[1]);
	int K = atoi(argv[2]);
	int T = atoi(argv[3]);

	omp_set_num_threads(T);

	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	arr_x = malloc(sizeof(float) * N);
	arr_y = malloc(sizeof(float) * N);

	init(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = k_means(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster,N,K,T);

	free(arr_x);
	free(arr_y);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}