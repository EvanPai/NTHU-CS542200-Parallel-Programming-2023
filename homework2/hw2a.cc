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

#include <pthread.h>

#include <queue>

std::queue<int> work_queue;
pthread_mutex_t work_queue_mutex = PTHREAD_MUTEX_INITIALIZER;

union PackTwoDouble {
    alignas(16) double d[2];
    __m128d d2;
};

typedef struct {
    int thread_id;
} ThreadInfo;

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

void cal_pixel_sse2(int row){
    int index = 0;
    int repeats[2] = {0, 0};
    int cursor[2] = {0, 0};

    cursor[0] = index++;
    cursor[1] = index++;

    union PackTwoDouble length_squared, z_reals, z_imags, c_reals, c_imags;

    length_squared.d2 = _mm_set_pd(0, 0);
    z_reals.d2 = _mm_set_pd(0, 0);
    z_imags.d2 = _mm_set_pd(0, 0);

    double origin_imag = row * ((upper - lower) / height) + lower;

    c_reals.d[0] = cursor[0] * ((right - left) / width) + left;
    c_reals.d[1] = cursor[1] * ((right - left) / width) + left;

    c_imags.d[0] = origin_imag;
    c_imags.d[1] = origin_imag;



    while(cursor[0] < width || cursor[1] < width ){
        
        // if length_squared.d[0] < 4.0 break
        // if length_squared.d[1] < 4.0 break

        while(true){
            __m128d z_reals_squared = _mm_mul_pd(z_reals.d2, z_reals.d2);
            __m128d z_imags_squared = _mm_mul_pd(z_imags.d2, z_imags.d2);

            length_squared.d2 = _mm_add_pd(z_reals_squared, z_imags_squared);

            if(length_squared.d[0] > 4 || length_squared.d[1] > 4) break;

            __m128d z_reals_images_mul = _mm_mul_pd(z_reals.d2, z_imags.d2);

            z_reals.d2 = _mm_add_pd(_mm_sub_pd(z_reals_squared, z_imags_squared), c_reals.d2);
            z_imags.d2 = _mm_add_pd(_mm_add_pd(z_reals_images_mul, z_reals_images_mul), c_imags.d2);


            ++repeats[0];
            ++repeats[1];

            if (repeats[0] >= iters || repeats[1] >= iters) break;
        }   


        if((length_squared.d[0] > 4) || repeats[0] >= iters) {
            // 儲存入本pixel的圖片
            image[row * width + cursor[0]] = repeats[0];

            repeats[0] = 0;
            cursor[0] = index++;

            z_reals.d[0] = 0;
            z_imags.d[0] = 0;
            length_squared.d[0] = 0;

            c_reals.d[0] = cursor[0] * ((right - left) / width) + left;
            c_imags.d[0] = origin_imag;
            
            if(cursor[0] >= width){
                c_reals.d[0] = 0;
                c_imags.d[0] = 0;
            }
        }
        
        if((length_squared.d[1] > 4) || repeats[1] >= iters) {
            // 儲存入本pixel的圖片
            image[row * width + cursor[1]] = repeats[1];

            repeats[1] = 0;
            cursor[1] = index++;

            z_reals.d[1] = 0;
            z_imags.d[1] = 0;
            length_squared.d[1] = 0;

            c_reals.d[1] = cursor[1] * ((right - left) / width) + left;
            c_imags.d[1] = origin_imag;
            
            if(cursor[1] >= width){
                c_reals.d[1] = 0;
                c_imags.d[1] = 0;
            }
            
        } 
    }
}



void* render_mandelbrot(void* arg) {
    ThreadInfo* thread_info = (ThreadInfo*)arg;

    while (1) {
        int row;
        pthread_mutex_lock(&work_queue_mutex);

        if (work_queue.empty()) {
            pthread_mutex_unlock(&work_queue_mutex);
            break;  // No more work to be done
        }

        row = work_queue.front();
        work_queue.pop();
        pthread_mutex_unlock(&work_queue_mutex);


        cal_pixel_sse2(row);
        
    }

    pthread_exit(NULL);
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

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Initialize the work queue with rows to process
    for (int j = 0; j < height; ++j) {
        work_queue.push(j);
    }

    /* Pthread parallel */
    pthread_t threads[num_threads];
    ThreadInfo thread_info[num_threads];

    for (int i = 0; i < num_threads; ++i) {
        thread_info[i].thread_id = i;
        pthread_create(&threads[i], NULL, render_mandelbrot, &thread_info[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }


    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
