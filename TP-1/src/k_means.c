#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

#define N 100000
#define K 4

//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,float *arr_x,float *arr_y,int *n_elem_cluster){
	float mean_x[K],mean_y[K];
	float dist_x[K],dist_y[K];
	int ret = 0;
	char flag = 1;

	while(flag){
		//reset means and n_elem_cluster for cluster calculation
		for(int i = 0; i < K;i++){
			mean_x[i] = 0;
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}

		//primary loop calculate the 
		for(int i = 0; i < N;i++){ 
			for(int j = 0; j < K;j++){//calculate distance betewn point and centroids 
				dist_x[j] = (arr_x[i] < cluster_x[j])? cluster_x[j] - arr_x[i]: arr_x[i] - cluster_x[j]; 
				dist_y[j] = (arr_y[i] < cluster_y[j])? cluster_y[j] - arr_y[i]: arr_y[i] - cluster_y[j];
			}

			int ind = 0; //indice da menor distancia
			for(int j = 1; j < K;j++) ind = (dist_x[j] + dist_y[j] < dist_x[ind] + dist_y[ind]) ?j:ind;
			mean_x[ind] += arr_x[i];
			mean_y[ind] += arr_y[i];
			n_elem_cluster[ind]++; 
		}

		//new centroids calculations 
		for(int i = 0; i < K;i++){
			mean_x[i] = mean_x[i] / ((float)n_elem_cluster[i]);
			mean_y[i] = mean_y[i] / ((float)n_elem_cluster[i]);
		}

		flag = 0;
		for(int i = 0; i < K;i++){
			if(mean_x[i] != cluster_x[i] || mean_y[i] != cluster_y[i]) {flag = 1;}
			cluster_x[i] = mean_x[i];
			cluster_y[i] = mean_y[i];
		}

		ret++;

	}
	return ret;
}

int main(){
	float arr_x[N],arr_y[N];
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	inicio(N,K,arr_x,arr_y,cluster_x,cluster_y);

	int iterarion = k_means(cluster_x,cluster_y,arr_x,arr_y,n_elem_cluster);

	print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);
	return 0;
}