#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <emmintrin.h> // SSE2

#include <mpi.h>
#include <omp.h>

#include <queue>



union PackTwoDouble {
    alignas(16) double d[2];
    __m128d d2;
};

typedef struct {
    int thread_id;
} ThreadInfo;

typedef struct {
    int row;
    int width_start;
    int width_end;
} Task_Struct;

std::queue<int> row_queue;

std::queue<Task_Struct> work_queue;

int num_threads;
int* image;
int iters; // iteration幾次。int; [1, 2×108]
double left; // real軸的左邊邊界。double; [-10, 10]
double right; // real軸的右邊邊界。double; [-10, 10]
double lower; // image軸的下邊界。double; [-10, 10]
double upper; // image軸的上邊界。double; [-10, 10]
int width; // 圖片x軸有多少points。int; [1, 16000]
int height; // 圖片y軸有多少points。int; [1, 16000]


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void cal_pixel_sse2(Task_Struct T, int* buffer, int cal_case){
    int index = T.width_start;
    int repeats[2] = {0, 0};
    int cursor[2] = {0, 0};
    int flags[2] = {0, 0};

    cursor[0] = index++;
    cursor[1] = index++;

    union PackTwoDouble length_squared, z_reals, z_imags, c_reals, c_imags;

    length_squared.d2 = _mm_set_pd(0, 0);
    z_reals.d2 = _mm_set_pd(0, 0);
    z_imags.d2 = _mm_set_pd(0, 0);

    double origin_imag = T.row * ((upper - lower) / height) + lower;

    c_reals.d[0] = cursor[0] * ((right - left) / width) + left;
    c_reals.d[1] = cursor[1] * ((right - left) / width) + left;

    c_imags.d[0] = origin_imag;
    c_imags.d[1] = origin_imag;



    while(cursor[0] < T.width_end || cursor[1] < T.width_end ){
        
        // if length_squared.d[0] < 4.0 break
        // if length_squared.d[1] < 4.0 break

        while(true){
            __m128d z_reals_squared = _mm_mul_pd(z_reals.d2, z_reals.d2);
            __m128d z_imags_squared = _mm_mul_pd(z_imags.d2, z_imags.d2);

            length_squared.d2 = _mm_add_pd(z_reals_squared, z_imags_squared);

            if(length_squared.d[0] > 4 || length_squared.d[1] > 4) break;

            __m128d z_reals_imags_mul = _mm_mul_pd(z_reals.d2, z_imags.d2);

            z_reals.d2 = _mm_add_pd(_mm_sub_pd(z_reals_squared, z_imags_squared), c_reals.d2);
            z_imags.d2 = _mm_add_pd(_mm_add_pd(z_reals_imags_mul, z_reals_imags_mul), c_imags.d2);


            ++repeats[0];
            ++repeats[1];

            if (repeats[0] >= iters || repeats[1] >= iters) break;
        }   


        if((length_squared.d[0] > 4) || repeats[0] >= iters) {
            // 儲存入本pixel的圖片
            // 因為是1維的所以做mapping
            if(cal_case == 1) image[T.row * width + cursor[0]] = repeats[0];
            else buffer[cursor[0]] = repeats[0];

            repeats[0] = 0;
            cursor[0] = index++;

            z_reals.d[0] = 0;
            z_imags.d[0] = 0;
            length_squared.d[0] = 0;

            c_reals.d[0] = cursor[0] * ((right - left) / width) + left;
            c_imags.d[0] = origin_imag;
            
            if(cursor[0] >= T.width_end){
                c_reals.d[0] = 0;
                c_imags.d[0] = 0;
            }
        }
        
        if((length_squared.d[1] > 4) || repeats[1] >= iters) {
            // 儲存入本pixel的圖片
            if(cal_case == 1) image[T.row * width + cursor[1]] = repeats[1];
            else buffer[cursor[1]] = repeats[1];

            repeats[1] = 0;
            cursor[1] = index++;

            z_reals.d[1] = 0;
            z_imags.d[1] = 0;
            length_squared.d[1] = 0;

            c_reals.d[1] = cursor[1] * ((right - left) / width) + left;
            c_imags.d[1] = origin_imag;
            
            if(cursor[1] >= T.width_end){
                c_reals.d[1] = 0;
                c_imags.d[1] = 0;
            }
            
        } 
    }
}

void render_mandelbrot(int* buffer, int cal_case) {
    while (1) {
        Task_Struct T;
        bool should_break = false;

        #pragma omp critical
        {
            if (work_queue.empty()) {
                should_break = true;  // No more work to be done
            } else {
                T = work_queue.front();
                work_queue.pop();
                
            }


        }

        if (should_break) {
            break;
        }

        cal_pixel_sse2(T, buffer, cal_case);
    }
}



int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10); // iteration幾次。int; [1, 2×108]
    left = strtod(argv[3], 0); // real軸的左邊邊界。double; [-10, 10]
    right = strtod(argv[4], 0); // real軸的右邊邊界。double; [-10, 10]
    lower = strtod(argv[5], 0); // image軸的下邊界。double; [-10, 10]
    upper = strtod(argv[6], 0); // image軸的上邊界。double; [-10, 10]
    width = strtol(argv[7], 0, 10); // 圖片x軸有多少points。int; [1, 16000]
    height = strtol(argv[8], 0, 10); // 圖片y軸有多少points。int; [1, 16000]

    /* MPI initialize */
    int p; // how many process
    int rank; // this process's rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 考慮只有一個process的情況
    if(p == 1){
        printf("only 1 process\n");
        /* allocate memory for image */
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);

        Task_Struct T;

        for (int j = 0; j < height; ++j) {
            int width_start = 0;
            int chunk_size = 10;

            while (width_start < width) {
                int width_end = width_start + chunk_size;
                if (width_end > width) {
                    width_end = width; // Adjust the last chunk
                }

                // Create a task for the chunk
                T.row = j;
                T.width_start = width_start;
                T.width_end = width_end;

                // Push the task (chunk) into the work_queue
                work_queue.push(T);

                width_start = width_end;
            }
        }

        int* buf = (int*) malloc(1 * sizeof(int));
        /* omp parallel */
        #pragma omp parallel num_threads(num_threads)
        {
            render_mandelbrot(buf, 1);
        }
    }
    else{
        if(rank == 0){ // master process
            /* allocate memory for image */
            image = (int*)malloc(width * height * sizeof(int));
            assert(image);

            /* MPI work pool */

            for (int j = height-1; j >= 0; --j) {
                row_queue.push(j);
            }

            // 初始化變數
            int* master_buffer = (int*) malloc(width * sizeof(int)); // 接收算完的那個row顏色分別要是多少，之後再由rank 0來畫
            int count = 0;
            int j;
            int termination_tag = -1; // Termination tag
            int completed_row = -1; // Variable to store the completed row
            int completed_worker;
            MPI_Status status;

            // 先送第一筆工作給workers
            for(int k=1; k<p; k++){
                j = row_queue.front();
                row_queue.pop();
                MPI_Send(&j, 1, MPI_INT, k, 0, MPI_COMM_WORLD);
                count++;
            }

            // 接著開始動態分工作
            do{
                MPI_Recv(master_buffer, width, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
                completed_worker = status.MPI_SOURCE;
                completed_row = status.MPI_TAG; // Store the row that was processed
                count--;

                // draw the image
                //for(int i=0; i<width; ++i){
                //    image[completed_row * width + i] = master_buffer[i];
                //}
                memcpy(&image[completed_row * width], master_buffer, width * sizeof(int));

                // allocate next task
                if(!row_queue.empty()){
                    j = row_queue.front();
                    row_queue.pop();
                    MPI_Send(&j, 1, MPI_INT, completed_worker, 0, MPI_COMM_WORLD);
                    count++;
                }
                else{
                    MPI_Send(&termination_tag, 1, MPI_INT, completed_worker, 0, MPI_COMM_WORLD);
                }
            }while(count > 0); // 在分發出去的工作都收回來之前不能停
        }
        else{ // worker process
            int received_row;
            Task_Struct T;

            // 創建一個buffer
            int* buffer = (int*) malloc(width * sizeof(int)); // 接收算完的那個row顏色分別要是多少，之後再由rank 0來畫
            while (1) {
                // Receive a row from the manager
                MPI_Recv(&received_row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Check if there's no more work to be done
                if (received_row == -1) { // terminating tag
                    break;
                }

                // Process the entire row    
                int width_start = 0;
                int chunk_size = 10;

                while (width_start < width) {
                    int width_end = width_start + chunk_size;
                    if (width_end > width) {
                        width_end = width; // Adjust the last chunk
                    }

                    // Create a task for the chunk
                    T.row = received_row;
                    T.width_start = width_start;
                    T.width_end = width_end;

                    // Push the task (chunk) into the work_queue
                    work_queue.push(T);
                    width_start = width_end;
                }
                
                // omp parallel 
                #pragma omp parallel num_threads(num_threads)
                {
                    render_mandelbrot(buffer, 0);
                }
                MPI_Send(buffer, width, MPI_INT, 0, received_row, MPI_COMM_WORLD);
            }
        }
    }
    
    

    // Initialize the work queue with rows to process
    // 這裡改成把width區間放盡work queue裡
    // 概念上會變成，MPI分一個row進來，這邊切成很多個width為20的區間，分給一個個thread來做
    // 做到這，做好了分割chunk，接下來是MPI的work pool

    if(rank == 0){
        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
        free(image);
    }

    MPI_Finalize();
}
