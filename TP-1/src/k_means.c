#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

#define N 10000000
#define K 4

//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,float *arr_x,float *arr_y,int *n_elem_cluster){
	float mean_x[K],mean_y[K]; //keep values to calculate new centroid
	float dist[K]; //auxiliar vector to calculate distance between centroid and a point 
	float *points; // arrays that save the new and previous allocation of points to clusters
	int ret = -1; //keep number of iterations(starts at -1 not considering the setup iteration)
	char flag = 1; //flag of k_means loop 

	points = malloc(sizeof(float) * N);

	while(flag){
		//reset means and n_elem_cluster for calculations
		for(int i = 0; i < K;i++){
			mean_x[i] = 0;
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}

		flag = 0; // assume that the final condition is met 

		//primary loop that assigns points to clusters
		for(int i = 0; i < N;i++){ 

			int ind = 0; //lower distance index
			for(int j = 0; j < K;j++){//calculate distance between points and centroids 
				float x = (cluster_x[j] - arr_x[i]);
				float y = (cluster_y[j] - arr_y[i]);
				dist[j] = x * x;  
				dist[j] += y * y;
				ind = (dist[j] < dist[ind]) ?j:ind; //saves the index of the lower distance centroid
			}


			flag = (flag || points[i] != ind); // check if point centroid allocation is different for the point  

			mean_x[ind] += arr_x[i]; // add this point to the sum of points belonging to cluster
			mean_y[ind] += arr_y[i];
			n_elem_cluster[ind]++; //update number of elements in cluster
			
			points[i] = ind; // assigns the new lowest distance centroid to the point 
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

int main(){
	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	arr_x = malloc(sizeof(float) * N);
	arr_y = malloc(sizeof(float) * N);

	init(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = k_means(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster);

	free(arr_x);
	free(arr_y);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}