#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>

// #define numPoints 1000
// #define numDimensions 10000 
// #define numCentroids 50
// #define maxIterations 50
// #define convergenceThreshold 0.0001

#define numPoints 450000000
#define numDimensions 2
#define numCentroids 4
#define maxIterations 15
#define convergenceThreshold 0.0001

// double time_count = 0;

// Function to calculate Euclidean distance between two points
double calculateDistance(double* point1, double* point2) {
    double distance = 0.0;
    for (int i = 0; i < numDimensions; ++i) {
        //distance += pow(point1[i] - point2[i], 2);
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

int hasConverged(double* oldCentroids, double* newCentroids) {
    for (int i = 0; i < numCentroids * numDimensions; ++i) {
        if (fabs(oldCentroids[i] - newCentroids[i]) > convergenceThreshold) {
            return 0;  // Not converged
        }
    }
    return 1;  // Converged
}

// Function to assign each data point to the nearest centroid
void assignToCentroids(double* data, double* centroids, int* assignments) {
    // struct timespec start, end, temp;
    // double time_used;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < numPoints; ++i) {
        double minDistance = DBL_MAX;
        int clusterID = 0;

        for (int j = 0; j < numCentroids; ++j) {
            double distance = calculateDistance(&data[i * numDimensions], &centroids[j * numDimensions]);
            if (distance < minDistance) {
                minDistance = distance;
                clusterID = j;
            }
        }

        assignments[i] = clusterID;
    }

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if ((end.tv_nsec - start.tv_nsec) < 0) {
    // temp.tv_sec = end.tv_sec-start.tv_sec-1;
    // temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    // } else {
    // temp.tv_sec = end.tv_sec - start.tv_sec;
    // temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;

    // time_count += time_used;
    //printf("%f second\n"
    //, time_used);
}

// Function to update centroids based on assigned points
void updateCentroids(double* data, int* assignments, double* centroids) {
    int* clusterSizes = calloc(numCentroids, sizeof(int));
    double* clusterSums = calloc(numCentroids * numDimensions, sizeof(double));

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
    //srand(time(NULL));
    srand(42);

    // Constants


    // Generate random data points and centroids
    double* data = (double*)malloc(numPoints * numDimensions * sizeof(double));
    double* centroids = (double*)malloc(numCentroids * numDimensions * sizeof(double));

    for (int i = 0; i < numPoints * numDimensions; ++i) {
        data[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < numCentroids * numDimensions; ++i) {
        centroids[i] = (double)rand() / RAND_MAX;
    }

    // Output the begining
    // printf("Begining centroids:\n");
    // for (int i = 0; i < numCentroids; ++i) {
    //     printf("Centroid %d: ", i);
    //     for (int j = 0; j < numDimensions; ++j) {
    //         printf("%f ", centroids[i * numDimensions + j]);
    //     }
    //     printf("\n");
    // }

    // Perform K-means clustering iterations
    int* assignments = (int*)malloc(numPoints * sizeof(int));
    double* oldCentroids = (double*)malloc(numCentroids * numDimensions * sizeof(double));
    for (int iter = 0; iter < maxIterations; ++iter) {
        // memcpy(oldCentroids, centroids, numCentroids * numDimensions * sizeof(double));

        // Assign points to centroids
        assignToCentroids(data, centroids, assignments);

        // Update centroids based on assigned points
        updateCentroids(data, assignments, centroids);
        // if (hasConverged(oldCentroids, centroids)) {
        //     printf("Converged at iteration %d\n", iter + 1);
        //     break;
        // }
    }

    // Output the results
    //printf("Centroid[0] %d", centroids[0]);
    // printf("Final centroids:\n");
    // for (int i = 0; i < numCentroids; ++i) {
    //     printf("Centroid %d: ", i);
    //     for (int j = 0; j < numDimensions; ++j) {
    //         printf("%f ", centroids[i * numDimensions + j]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < 2; ++i) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < 2; ++j) {
            printf("%f ", centroids[i * numDimensions + j]);
        }
        printf("\n");
    }

    // printf("%f second\n", time_count);

    free(data);
    free(centroids);
    free(assignments);
    free(oldCentroids);

    return 0;
}