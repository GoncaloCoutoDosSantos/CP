#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#include "../include/utils.h"

//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,const float *arr_x,const float *arr_y,int *n_elem_cluster,const int N,const int K,const int T){
	float *points; // arrays that save the new and previous allocation of points to clusters
	int ret = -1; //keep number of iterations(starts at -1 not considering the setup iteration)
	char flag = 1; //flag of k_means loop 

	points = malloc(sizeof(float) * N);

	//while(flag){
	for(int ite = 0; ite< 21;ite++){
		float mean_x[T][K],mean_y[T][K]; //keep values to calculate new centroid
		int n_elem_thread[T][K];
		
		DEBUG("------------------------------------------------------------------\n");
		//reset means and n_elem_cluster for calculations
		for(int i = 0; i < T;i++){
			memset(mean_x[i],0,K * sizeof(float));
			memset(mean_y[i],0,K * sizeof(float));
			memset(n_elem_thread[i],0,K * sizeof(int));
			DEBUG("t[%d]:%d,%d,%d,%d\n",i,n_elem_thread[i][0],n_elem_thread[i][1],n_elem_thread[i][2],n_elem_thread[i][3]);
		}

		flag = 0; // assume that the final condition is met 
		//primary loop that assigns points to clusters
		DEBUG("------------------------------------------------------------------\n");

		#pragma omp parallel
		{

			#pragma omp for   
			for(int i = 0; i < N;i++){ 
				int id =  omp_get_thread_num();
				for(int j = 0;j < T;j++) DEBUG("[%d]:t[%d]:%d,%d,%d,%d\n",i,j,n_elem_thread[j][0],n_elem_thread[j][1],n_elem_thread[j][2],n_elem_thread[j][3]);
				DEBUG("i:%d\n",i);
				float min_dist = 10;
				float dist[K]; //auxiliar vector to calculate distance between centroid and a point 
				int ind = 0; //lower distance index
				for(int j = 0; j < K;j++){//calculate distance between points and centroids 
					float x = (cluster_x[j] - arr_x[i]);
					float y = (cluster_y[j] - arr_y[i]);
					dist[j] = x * x;  
					dist[j] += y * y;
					ind = (dist[j] < min_dist) ?j:ind; //saves the index of the lower distance centroid
					min_dist = dist[ind];
				}


				//flag = (flag || points[i] != ind); // check if point centroid allocation is different for the point  

				mean_x[id][ind] += arr_x[i]; // add this point to the sum of points belonging to cluster
				mean_y[id][ind] += arr_y[i];
				n_elem_thread[id][ind]++; //update number of elements in cluster
				DEBUG("add count n_thread[%d][%d]:%d\n",id,ind,n_elem_thread[id][ind]);
				
				points[i] = ind; // assigns the new lowest distance centroid to the point 
			}

			#pragma omp single
			{
				DEBUG("%d\n",omp_get_thread_num());
				for(int i = 1; i < T;i++){
					for(int j = 0; j < K;j++){
						DEBUG("n_elem[0][%d]:%d\n",j,n_elem_thread[0][j]);
						mean_x[0][j] += mean_x[i][j];
						mean_y[0][j] += mean_y[i][j];
						n_elem_thread[0][j] += n_elem_thread[i][j];
						DEBUG("n_elem[%d][%d]:%d\n",i,j,n_elem_thread[i][j]);

					}
				}

				for(int i = 0; i < K;i++) n_elem_cluster[i] = n_elem_thread[0][i];

				//new centroids calculations 
				for(int i = 0; i < K;i++){
					cluster_x[i] = mean_x[0][i] / (n_elem_cluster[i]);
					cluster_y[i] = mean_y[0][i] / (n_elem_cluster[i]);
				}


				DEBUG("------------------------------------------------------------------\n");
				//reset means and n_elem_cluster for calculations
				for(int i = 0; i < T;i++){
					memset(mean_x[i],0,K * sizeof(float));
					memset(mean_y[i],0,K * sizeof(float));
					memset(n_elem_thread[i],0,K * sizeof(int));
					DEBUG("t[%d]:%d,%d,%d,%d\n",i,n_elem_thread[i][0],n_elem_thread[i][1],n_elem_thread[i][2],n_elem_thread[i][3]);
				}

				flag = 0; // assume that the final condition is met 
				//primary loop that assigns points to clusters
				DEBUG("------------------------------------------------------------------\n");
			}
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