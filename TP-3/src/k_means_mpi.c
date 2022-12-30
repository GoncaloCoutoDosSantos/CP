#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "../include/utils.h"

#define N 10000000
#define K 4

/*
//return is number of iterarions
int k_means(float *cluster_x,float *cluster_y,float *arr_x,float *arr_y,int *n_elem_cluster){
	float mean_x[K],mean_y[K]; //keep values to calculate new centroid
	float dist[K]; //auxiliar vector to calculate distance between centroid and a point 
	float *points; // arrays that save the new and previous allocation of points to clusters
	int ret = -1; //keep number of iterations(starts at -1 not considering the setup iteration)
	char flag = 1; //flag of k_means loop 

	points = malloc(sizeof(float) * N);

    // Initialize MPI
    int rank, size;
	MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	float m = N/size;

	while(flag){

		//reset means and n_elem_cluster for calculations
		for(int i = m*rank; i < m*rank + m; i++){
			mean_x[i] = 0;
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}


		flag = 0; // assume that the final condition is met 

		//primary loop that assigns points to clusters
		for(int i = m*rank; i < m*rank + m;i++){ 

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

		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Reduce(mean_x,mean_x,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(mean_y,mean_y,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(n_elem_cluster,n_elem_cluster,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

		if(rank==0){
	
			//new centroids calculations 
			for(int i = 0; i < K;i++){
				cluster_x[i] = mean_x[i] / (n_elem_cluster[i]);
				cluster_y[i] = mean_y[i] / (n_elem_cluster[i]);
			}
			
		}
		MPI_Bcast(cluster_x,K,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(cluster_y,K,MPI_FLOAT,0,MPI_COMM_WORLD);

		ret++;

	}
    MPI_Finalize();

	free(points);


	return ret;
}
*/
int main(){
    // Initialize MPI
    int rank, size;
	MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	float *arr_xx,*arr_yy;
	float *arr_x,*arr_y;
	float cluster_x[K],cluster_y[K];
	int n_elem_cluster[K];

	if(rank==0){
		arr_xx = malloc(sizeof(float) * N);
		arr_yy = malloc(sizeof(float) * N);

		init(N,K,arr_xx,arr_yy,cluster_x,cluster_y);
	}

	MPI_Bcast(cluster_x,K,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(cluster_y,K,MPI_FLOAT,0,MPI_COMM_WORLD);

	int m = N/size;

	MPI_Scatter(arr_xx,N,MPI_FLOAT,arr_x,m,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Scatter(arr_yy,N,MPI_FLOAT,arr_y,m,MPI_FLOAT,0,MPI_COMM_WORLD);


	float mean_x[K],mean_y[K]; //keep values to calculate new centroid
	float dist[K]; //auxiliar vector to calculate distance between centroid and a point 
	float *points; // arrays that save the new and previous allocation of points to clusters
	int ret = -1; //keep number of iterations(starts at -1 not considering the setup iteration)
	char flag = 1; //flag of k_means loop 

	points = malloc(sizeof(float) * m);


	while(flag){

		//reset means and n_elem_cluster for calculations
		for(int i = 0; i <m; i++){
			mean_x[i] = 0;
			mean_y[i] = 0;
			n_elem_cluster[i] = 0;
		}

		flag = 0; // assume that the final condition is met 

		//primary loop that assigns points to clusters
		for(int i = 0; i < m;i++){ 

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

		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Reduce(mean_x,mean_x,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(mean_y,mean_y,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(n_elem_cluster,n_elem_cluster,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(flag,flag,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

		if(rank==0){
			if(flag!=0) flag=1;
			//new centroids calculations 
			for(int i = 0; i < K;i++){
				cluster_x[i] = mean_x[i] / (n_elem_cluster[i]);
				cluster_y[i] = mean_y[i] / (n_elem_cluster[i]);
			}
			
		}

		MPI_Bcast(cluster_x,K,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(cluster_y,K,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(flag,1,MPI_INT,0,MPI_COMM_WORLD);

		ret++;

	}

	free(points);

	free(arr_x);
	free(arr_y);

	if(rank==0) print_ret(cluster_x,cluster_y,n_elem_cluster,N,K,iterarion);

    MPI_Finalize();

	return 0;
}