#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

int cmp(const void* a, const void* b){
    float* va = (float*) a;
    float* vb = (float*) b;

    if(*va > *vb) return 1;
    else if(*va < *vb) return -1;
    else return 0;
}

void merge_low(float** local_data, float* temp, int local_n_of_a, int local_n_of_b){
    int a = 0, b = 0, c = 0, count = 0;
    float* merge = (float*)malloc(local_n_of_a * sizeof(float));


    while(a < local_n_of_a && b < local_n_of_b && count < local_n_of_a) {
        if((*local_data)[a] < temp[b]){
            merge[c++] = (*local_data)[a++];
        } else {
            merge[c++] = temp[b++];
        }
        count++;
    }

    while(a < local_n_of_a && count < local_n_of_a) {
        merge[c++] = (*local_data)[a++];
        count++;
    }

    while(b < local_n_of_b && count < local_n_of_a) {
        merge[c++] = temp[b++];
        count++;
    }

    free(*local_data);
    *local_data = merge; 
}

void merge_high(float** local_data, float* temp, int local_n_of_a, int local_n_of_b) {
    int a = local_n_of_a - 1, b = local_n_of_b - 1, c = local_n_of_a - 1, count = 0;
    float* merge = (float*)malloc(local_n_of_a * sizeof(float));

    while (a > -1 && b > -1 && count < local_n_of_a) {
        if ((*local_data)[a] > temp[b]) {
            merge[c--] = (*local_data)[a--];
        } else {
            merge[c--] = temp[b--];
        }
        count++;
    }

    while (a > -1 && count < local_n_of_a) {
        merge[c--] = (*local_data)[a--];
        count++;
    }

    while (b > -1 && count < local_n_of_a) {
        merge[c--] = temp[b--];
        count++;
    }

    free(*local_data);
    *local_data = merge;
}

int main(int argc, char** argv) {
    int rank; //this process's rank
    int p; //how many process
    int i, j;
    float* local_data = NULL;
    float* merge = NULL;
    float* temp = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = atoi(argv[1]); //how many data to sort
    char *input_filename = argv[2]; //input file
    char *output_filename = argv[3]; //output file
    MPI_File input_file, output_file;

    //Scatter data for eachprocess
    int local_n = n / p;
    int remainder = n % p; //extra elements
    
    //scatter the extra elements
    if(rank < remainder) local_data = (float*)malloc( (local_n + 1) * sizeof(float));
    else local_data = (float*)malloc(local_n * sizeof(float));

    //record the local_n for each process
    int* send_count = (int*) malloc(p * sizeof(int));
    int* displs = (int*) malloc(p * sizeof(int));

    for(i=0; i<p; i++){
        send_count[i] = (i<remainder) ? (local_n+1) : local_n; //how many data for this process
        displs[i] = (i==0) ? 0 : (displs[i-1] + send_count[i-1]); //displacement for file reading
    }

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at_all(input_file, sizeof(float) * displs[rank], local_data, send_count[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    MPI_Barrier(MPI_COMM_WORLD);

    //local sort
    qsort(local_data, send_count[rank], sizeof(float), cmp);

    //find partner
    int even_partner;  
    int odd_partner;   
    
    if (rank % 2 != 0) { //odd rank
        even_partner = rank - 1;
        odd_partner = rank + 1;
        if (odd_partner == p) odd_partner = MPI_PROC_NULL;  //idle
    } else { //even rank 
        even_partner = rank + 1;
        if (even_partner == p) even_partner = MPI_PROC_NULL;  //idle
        odd_partner = rank-1;
    }

    //int local_terminate_flag = 1;
    temp = (float*)malloc((local_n+1) * sizeof(float));
    for(i=0; i<p+1; i++){ 
        if(i % 2 == 0){ //even phase
            if(even_partner >= 0){
                if(rank % 2 == 0){ //even rank
                    MPI_Sendrecv(local_data, send_count[rank], MPI_FLOAT, even_partner, 0,
                        temp, send_count[rank + 1], MPI_FLOAT, even_partner, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // for early termination
                    if(local_data[send_count[rank] - 1] < temp[0]) {
                        //local_terminate_flag = 0; //todo here
                        continue;
                    }

                    merge_low(&local_data, temp, send_count[rank], send_count[rank+1]);              
                }
                else{ //odd rank
                    MPI_Sendrecv(local_data, send_count[rank], MPI_FLOAT, even_partner, 0,
                        temp, send_count[rank-1], MPI_FLOAT, even_partner, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // for early termination
                    if(local_data[0] > temp[send_count[rank-1] - 1]) continue;
                    merge_high(&local_data, temp, send_count[rank], send_count[rank-1]);  
                }
            }
        
        }
        else{ //odd phase
            if(odd_partner >= 0){
                if(rank % 2 == 0){ //even rank
                    MPI_Sendrecv(local_data, send_count[rank], MPI_FLOAT, odd_partner, 0,
                        temp, send_count[rank-1], MPI_FLOAT, odd_partner, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // for early termination
                    if(local_data[0] > temp[send_count[rank-1] - 1]) continue;
                    merge_high(&local_data, temp, send_count[rank], send_count[rank-1]);  
                    
                }
                else if(rank%2 == 1){ //odd rank
                    MPI_Sendrecv(local_data, send_count[rank], MPI_FLOAT, odd_partner, 0,
                        temp, send_count[rank+1], MPI_FLOAT, odd_partner, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // for early termination
                    if(local_data[send_count[rank] - 1] < temp[0]) continue;
                    merge_low(&local_data, temp, send_count[rank], send_count[rank+1]);  
                    
                }
            
            }
            
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 

    //output the result
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);   
    MPI_File_write_at_all(output_file, sizeof(float) * displs[rank], local_data, send_count[rank], MPI_FLOAT, MPI_STATUS_IGNORE); 
    MPI_File_close(&output_file);
    
    MPI_Finalize();
    return 0;
}