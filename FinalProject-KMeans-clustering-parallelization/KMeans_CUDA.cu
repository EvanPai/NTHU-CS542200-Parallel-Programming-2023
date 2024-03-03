#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

#define BLOCK_DIM 1000

#define numPoints 100000
#define numDimensions 256
#define numCentroids 1024
#define maxIterations 15
#define convergenceThreshold 0.0001

int hasConverged(double* oldCentroids, double* newCentroids) {
    for (int i = 0; i < numCentroids * numDimensions; ++i) {
        if (fabs(oldCentroids[i] - newCentroids[i]) > convergenceThreshold) {
            return 0;  // Not converged
        }
    }
    return 1;  // Converged
}

__global__ void assignToCentroidsKernel(double* data, double* centroids, int* assignments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    double minDistance = DBL_MAX;
    double distance = 0.0;
    int clusterID = 0;
    
    for (int j = 0; j < numCentroids; ++j) {    
        distance = 0.0;

        for (int k = 0; k < numDimensions; ++k) {
            distance += (data[i * numDimensions + k] - centroids[j * numDimensions + k]) * (data[i * numDimensions + k] - centroids[j * numDimensions + k]);
        }
        distance = sqrt(distance);
        if (distance < minDistance) {
            minDistance = distance;
            clusterID = j;
        }
    }

    assignments[i] = clusterID;
}


void updateCentroids(double* data, int* assignments, double* centroids) {
    int* clusterSizes = (int*) calloc(numCentroids, sizeof(int));
    double* clusterSums = (double*) calloc(numCentroids * numDimensions, sizeof(double));

    // Accumulate values for each cluster
    for (int i = 0; i < numPoints; ++i) {
        int clusterID = assignments[i];
        clusterSizes[clusterID]++;
        for (int j = 0; j < numDimensions; ++j) {
            clusterSums[clusterID * numDimensions + j] += data[i * numDimensions + j];
        }
    }

    // Update centroids
    for (int i = 0; i < numCentroids; ++i) {
        for (int j = 0; j < numDimensions; ++j) {
            centroids[i * numDimensions + j] = (clusterSizes[i] > 0) ? clusterSums[i * numDimensions + j] / clusterSizes[i] : centroids[i * numDimensions + j];
        }
    }

    free(clusterSizes);
    free(clusterSums);

}



int main() {
    // Set random seed for reproducibility
    srand(42);

    // Generate random data points and centroids
    double* data = (double*)malloc(numPoints * numDimensions * sizeof(double));
    double* centroids = (double*)malloc(numCentroids * numDimensions * sizeof(double));
    int* assignments = (int*)malloc(numPoints * sizeof(int));


    for (int i = 0; i < numPoints * numDimensions; ++i) {
        data[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < numCentroids * numDimensions; ++i) {
        centroids[i] = (double)rand() / RAND_MAX;
    }

    // Output the begining
    printf("Begining centroids:\n");
    for (int i = 0; i < numCentroids; ++i) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < numDimensions; ++j) {
            printf("%f ", centroids[i * numDimensions + j]);
        }
        printf("\n");
    }

    // Allocate device memory
    double* d_data;
    double* d_centroids;
    int* d_assignments;

    cudaMalloc(&d_data, numPoints * numDimensions * sizeof(double) );
    cudaMalloc(&d_centroids, numCentroids * numDimensions * sizeof(double) );
    cudaMalloc(&d_assignments, numPoints * sizeof(int) );


    // Copy data from host to device
    cudaMemcpy(d_data, data, numPoints * numDimensions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, numCentroids * numDimensions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignments, assignments, numPoints * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid dim and block dim
    dim3 grid_dim((numPoints + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);

    // Launch CUDA kernels
    double* oldCentroids = (double*)malloc(numCentroids * numDimensions * sizeof(double));


    for (int iter = 0; iter < maxIterations; ++iter) {
        memcpy(oldCentroids, centroids, numCentroids * numDimensions * sizeof(double));

        assignToCentroidsKernel<<<grid_dim, block_dim>>>(d_data, d_centroids, d_assignments);
        cudaDeviceSynchronize(); // Ensure kernel completion before proceeding

        cudaMemcpy(assignments, d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

        updateCentroids(data, assignments, centroids);
        cudaMemcpy(d_centroids, centroids, numCentroids * numDimensions * sizeof(double), cudaMemcpyHostToDevice);

        
        if (hasConverged(oldCentroids, centroids)) {
            printf("Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);

    // Output the results
    printf("Final centroids:\n");
    printf("Centroid[0] = %d\n", centroids[0]);
    for (int i = 0; i < numCentroids; ++i) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < numDimensions; ++j) {
            printf("%f ", centroids[i * numDimensions + j]);
        }
        printf("\n");
    }

    free(data);
    free(centroids);
    free(assignments);
    free(oldCentroids);

    return 0;
}