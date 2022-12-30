#include <stdlib.h>
#include <stdio.h>

void init(int n_elem,int n_k,float *arr_x,float *arr_y,float *cluster_x,float *cluster_y) {
	 srand(10);
	 for(int i = 0; i < n_elem; i++) {
		 arr_x[i] = (float) rand() / RAND_MAX;
		 arr_y[i] = (float) rand() / RAND_MAX;
	 }
	 for(int i = 0; i <  n_k; i++) {
		 cluster_x[i] = arr_x[i];
		 cluster_y[i] = arr_y[i];
	 }
}

void print_ret(float *cluster_x,float *cluster_y,int *n_elem_cluster,int size,int n_k,int iterarion){
	printf("N = %d, K = %d\n",size,n_k);
	for(int i = 0;i < n_k;i++){
		printf("Center: (%.3f, %.3f) : Size: %d\n",cluster_x[i],cluster_y[i],n_elem_cluster[i]);
	}
	printf("Iterations: %d\n",iterarion);
}