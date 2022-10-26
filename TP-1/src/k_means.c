#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

#define N 10000000
#define K 4

//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,float *arr_x,float *arr_y,int *n_elem_cluster){
	float mean_x[K],mean_y[K]; //keep values to calculate new centroid
	float dist[K]; //auxiliar vector for calculate distance betwen centroid and a point 
	float *new_points,*old_points;
	int ret = 0;
	char flag = 1;

	new_points = malloc(sizeof(float) * N);
	old_points = calloc(sizeof(float) , N);

	while(flag){
		//reset means and n_elem_cluster for cluster calculation
		for(int i = 0; i < K;i++){
			mean_x[i] = 0;
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}

		flag = 0;

		//primary loop calculate the 
		for(int i = 0; i < N;i++){ 

			int ind = 0; //indice da menor distancia
			for(int j = 0; j < K;j++){//calculate distance betewn point and centroids 
				dist[j]  = (cluster_x[j] - arr_x[i]) * (cluster_x[j] - arr_x[i]); 
				dist[j] += (cluster_y[j] - arr_y[i]) * (cluster_y[j] - arr_y[i]);
				ind = (dist[j] < dist[ind]) ?j:ind;
			}

			new_points[i] = ind;

			if(!flag)flag = (old_points[i] != ind)?1:flag;

			mean_x[ind] += arr_x[i];
			mean_y[ind] += arr_y[i];
			n_elem_cluster[ind]++; 
		}

		//new centroids calculations 
		for(int i = 0; i < K;i++){
			cluster_x[i] = mean_x[i] / ((float)n_elem_cluster[i]);
			cluster_y[i] = mean_y[i] / ((float)n_elem_cluster[i]);
		}

		float *aux = new_points;
		new_points = old_points;
		old_points = aux;

		ret++;

	}


	free(new_points);
	free(old_points);


	return ret;
}

int main(){
	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	if(!(arr_x = malloc(sizeof(float) * N))) printf("erro allcar x\n");
	if(!(arr_y = malloc(sizeof(float) * N))) printf("erro allcar y\n");

	init(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = k_means(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster);

	free(arr_x);
	free(arr_y);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}