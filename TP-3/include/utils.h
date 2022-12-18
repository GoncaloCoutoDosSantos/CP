#ifndef UTILS_H
#define UTILS_H

void init(int n_elem,int n_k,float *arr_x,float *arr_y,float *cluster_x,float *cluster_y);

void print_ret(float *cluster_x,float *cluster_y,int *n_elem_cluster,int size,int n_k,int iterarion);

void stopKernelTime (void);

void startKernelTime (void);

void checkCUDAError (const char *msg);

#ifdef TEST
	#define DEBUG(x...) printf(x)
#else 
	#define DEBUG(x...) 
#endif

#endif