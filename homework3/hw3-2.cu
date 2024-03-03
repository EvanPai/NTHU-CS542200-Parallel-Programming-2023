#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 32

// Blocking factor（一個floyd warshall block要切多大）
#define B 64
#define offset 32
#include <sys/mman.h>
#include <sys/stat.h> 
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

//======================
#define DEV_NO 0
//cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
//const int V = 50010;
int n, m;
int N; // 用來算GPU memory size

//可優化，讓Dist的大小根據input決定
int *Dist = NULL;
//static int Dist[V][V];

// 可以加上inline來增快
void input(char* infile);
void output(char* outFileName);
int ceil(int a, int b);
void block_FW();

__global__ void phase1(int *dst, int Round, int N);
__global__ void phase2_1(int *dst, int Round, int N);
__global__ void phase2_2(int *dst, int Round, int N);
__global__ void phase3(int *dst, int Round, int N);


void input(char* infile) {
	int file = open(infile, O_RDONLY);
    int a = 0;
	int *ft = (int*)mmap(NULL, 2*sizeof(int), PROT_READ, MAP_PRIVATE, file, 0);
    m = ft[1];
    n = ft[0];
    // n是有幾個vertex, m是有幾個edge
    // 設定N，之後kernel計算就不用branch來看boundry
	if (n % B) N = n + (B - n % B);
	else N = n;

    int *pair = (int*)(mmap(NULL, (3 * m + 2) * sizeof(int), PROT_READ, MAP_PRIVATE, file, 0));
	Dist = (int*)malloc(N*N*sizeof(int));

	for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                Dist[i*N + j] = 0;
            } else {
                Dist[i*N + j] = INF;
            }
        }
    }

	#pragma unroll 4
	for (int i = 0; i < m; ++i) {
		Dist[pair[i*3+2]*N+pair[i*3+3]]= pair[i*3+4];
	}

	close(file);
	munmap(pair, (3 * m + 2) * sizeof(int));
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i*N + j] >= INF) Dist[i*N + j] = INF;
        }
        fwrite(&Dist[i*N], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW() {
	int round = ceil(n, B);
	int *dst = NULL;
	unsigned int size = N*N*sizeof(int);

    // 把Dist memory pin住，增加performance
	cudaHostRegister(Dist, size, cudaHostRegisterDefault);

    // 在GPU中開一塊size大小的memory給dst
	cudaMalloc(&dst, size);

    // 把Dist搬進GPU中的dst裡面
	cudaMemcpy(dst, Dist, size, cudaMemcpyHostToDevice);
	
    // 總共要 N / B個blocks（包含有多出來的）
	int blocks = (N + B - 1) / B;
	
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_dim(blocks, blocks);


    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
	for (int r = 0; r < round; ++r) {
		// phase 1
		phase1<<<1, block_dim>>>(dst, r, N);
		
		// phase 2
		phase2_1<<<blocks, block_dim, 0, stream1>>>(dst, r, N);
        phase2_2<<<blocks, block_dim, 0, stream2>>>(dst, r, N);

		// Synchronize with both streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

		// phase 3
		phase3<<<grid_dim, block_dim>>>(dst, r, N);
	}

    // GPU算完搬回CPU
	cudaMemcpy(Dist, dst, size, cudaMemcpyDeviceToHost);

    // 清掉dst
	cudaFree(dst);
}


// ------------------------ GPU --------------------------- //
__device__ int Min(int a, int b) {
	return min(a, b);
}

__global__ void phase1(int *dst, int Round, int N) {
    int y = threadIdx.y; // y軸 = row
	int y_offset = y + offset;
	__shared__ int s[B][B];
	int x = threadIdx.x; // x軸 = column
	int x_offset = x + offset;

	// y => 0~31
    // x => 0~31

    

    // 因為最多只能用1024(32 * 32)個threads，但要算(64 * 64)大小的block(B=64)
    // 且又要盡量用shared memory
    // 所以讓一個thread算4個點的資料。
	int top_left = Round * B * (N + 1) + y * N + x;
    s[y][x] = dst[top_left];

	int top_right = Round * B * (N + 1) + y * N + x + offset;
	s[y][x_offset] = dst[top_right];

	int bottom_left = Round * B * (N + 1) + (y + offset) * N + x;
	s[y_offset][x] = dst[bottom_left];

	int bottom_right = Round * B * (N + 1) + (y + offset) * N + x + offset;
	s[y_offset][x_offset] = dst[bottom_right];
	// load gloabal data to shared memory
	
	__syncthreads();

	for (int k = 0; k < B; ++k) {
		s[y][x] = Min(s[y][k] + s[k][x], s[y][x]);
		s[y][x_offset] = Min(s[y][k] + s[k][x_offset], s[y][x_offset]);
		s[y_offset][x] = Min(s[y_offset][k] + s[k][x], s[y_offset][x]);
		s[y_offset][x_offset] = Min(s[y_offset][k] + s[k][x_offset], s[y_offset][x_offset]);
		__syncthreads();
	}
	dst[top_left] = s[y][x];
	dst[top_right] = s[y][x_offset];
	dst[bottom_left] = s[y_offset][x];
	dst[bottom_right] = s[y_offset][x_offset];
}

__global__ void phase2_1(int *dst, int Round, int N) {
	if (blockIdx.x == Round) return;
	__shared__ int s[B][B];
	int y = threadIdx.y;
	int y_B = y + offset;
	__shared__ int col[B][B];
	int x = threadIdx.x;
	int x_B = x + offset;

	
	

    // 要算跟pivot B有row或col相同的所有B
    // 一樣，每個thread要算4個點
    // 算col的matrix B
    
	int main_top_left = Round * B * (N + 1) + y * N + x;
    s[y][x] = dst[main_top_left];
    int col_top_left = blockIdx.x * B * N + Round * B + y * N + x;
	col[y][x] = dst[col_top_left];

	int main_top_right = Round * B * (N + 1) + y * N + x + offset;
    s[y][x_B] = dst[main_top_right];
	int col_top_right = blockIdx.x * B * N + Round * B + y * N + x + offset;
	col[y][x_B] = dst[col_top_right];


	int main_bottom_left = Round * B * (N + 1) + (y + offset) * N + x;
    s[y_B][x] = dst[main_bottom_left];
	int col_bottom_left = blockIdx.x * B * N + Round * B + (y + offset) * N + x;
	col[y_B][x] = dst[col_bottom_left];


	int main_bottom_right = Round * B * (N + 1) + (y + offset) * N + x + offset;
    s[y_B][x_B] = dst[main_bottom_right];
	int col_bottom_right = blockIdx.x * B * N + Round * B + (y + offset) * N + x + offset;
	col[y_B][x_B] = dst[col_bottom_right];

	__syncthreads();
	
	for (int k = 0; k < B; ++k) {
        col[y][x] = Min(col[y][x], col[y][k] + s[k][x]);
        col[y][x_B] = Min(col[y][x_B], col[y][k] + s[k][x_B]);
        col[y_B][x] = Min(col[y_B][x], col[y_B][k] + s[k][x]);
        col[y_B][x_B] = Min(col[y_B][x_B], col[y_B][k] + s[k][x_B]);
		__syncthreads();
	}
    dst[col_top_left] = col[y][x];
    dst[col_top_right] = col[y][x_B];
    dst[col_bottom_left] = col[y_B][x];
    dst[col_bottom_right] = col[y_B][x_B];
}

__global__ void phase2_2(int *dst, int Round, int N) {
	if (blockIdx.x == Round) return;
	__shared__ int s[B][B];
	int y = threadIdx.y;
	int y_B = y + offset;
	__shared__ int row[B][B];
	int x = threadIdx.x;
	int x_B = x + offset;

    // 要算跟pivot B有row或col相同的所有B
    // 一樣，每個thread要算4個點

    // 算row的matrix B

	int main_top_left = Round * B * (N + 1) + y * N + x;
    s[y][x] = dst[main_top_left];
    int row_top_left = Round * B * N + blockIdx.x * B + y * N + x;
    row[y][x] = dst[row_top_left];

	int main_top_right = Round * B * (N + 1) + y * N + x + offset;
    s[y][x_B] = dst[main_top_right];
    int row_top_right = Round * B * N + blockIdx.x * B + y * N + x + offset;
    row[y][x_B] = dst[row_top_right];

	int main_bottom_left = Round * B * (N + 1) + (y + offset) * N + x;
    s[y_B][x] = dst[main_bottom_left];
    int row_bottom_left = Round * B * N + blockIdx.x * B + (y + offset) * N + x;
    row[y_B][x] = dst[row_bottom_left];

	int main_bottom_right = Round * B * (N + 1) + (y + offset) * N + x + offset;
    s[y_B][x_B] = dst[main_bottom_right];
    int row_bottom_right = Round * B * N + blockIdx.x * B + (y + offset) * N + x + offset;
    row[y_B][x_B] = dst[row_bottom_right];

	__syncthreads();
	
	for (int k = 0; k < B; ++k) {
        row[y][x] = Min(row[y][x], s[y][k] + row[k][x]);
        row[y][x_B] = Min(row[y][x_B], s[y][k] + row[k][x_B]);
        row[y_B][x] = Min(row[y_B][x], s[y_B][k] + row[k][x]);
        row[y_B][x_B] = Min(row[y_B][x_B], s[y_B][k] + row[k][x_B]);
		__syncthreads();
	}
    dst[row_top_left] = row[y][x];
    dst[row_top_right] = row[y][x_B];
    dst[row_bottom_left] = row[y_B][x];
    dst[row_bottom_right] = row[y_B][x_B];
}

__global__ void phase3(int *dst, int Round, int N) {
	if (blockIdx.x == Round || blockIdx.y == Round) return;
	__shared__ int col[B][B];
	int y = threadIdx.y;
	int y_B = y + offset;
	__shared__ int row[B][B];
	int x = threadIdx.x;
	int x_B = x + offset;
	__shared__ int target[B][B];
    
    
    

    int target_top_left = blockIdx.y * B * N + blockIdx.x * B + y * N + x;
    target[y][x] = dst[target_top_left];
    int col_top_left = blockIdx.y * B * N + Round * B + y * N + x;
    col[y][x] = dst[col_top_left];
    int row_top_left = Round * B * N + blockIdx.x * B + y * N + x;
    row[y][x] = dst[row_top_left];

    int target_top_right = blockIdx.y * B * N + blockIdx.x * B + y * N + x + offset;
    target[y][x_B] = dst[target_top_right];
    int col_top_right = blockIdx.y * B * N + Round * B + y * N + x + offset;
    col[y][x_B] = dst[col_top_right];
    int row_top_right = Round * B * N + blockIdx.x * B + y * N + x + offset;
    row[y][x_B] = dst[row_top_right];
	
    int target_bottom_left = blockIdx.y * B * N + blockIdx.x * B + (y + offset) * N + x;
    target[y_B][x] = dst[target_bottom_left];
    int col_bottom_left = blockIdx.y * B * N + Round * B + (y + offset) * N + x;
    col[y_B][x] = dst[col_bottom_left];
    int row_bottom_left = Round * B * N + blockIdx.x * B + (y + offset) * N + x;
    row[y_B][x] = dst[row_bottom_left];

    int target_bottom_right = blockIdx.y * B * N + blockIdx.x * B + (y + offset) * N + x + offset;
    target[y_B][x_B] = dst[target_bottom_right];
	int col_bottom_right = blockIdx.y * B * N + Round * B + (y + offset) * N + x + offset;
    col[y_B][x_B] = dst[col_bottom_right];
    int row_bottom_right = Round * B * N + blockIdx.x * B + (y + offset) * N + x + offset;
	row[y_B][x_B] = dst[row_bottom_right];
	
	__syncthreads();

	#pragma unroll 32
	for (int k = 0; k < B; ++k) {
		target[y][x] = Min(col[y][k] + row[k][x], target[y][x]);
		target[y][x_B] = Min(col[y][k] + row[k][x_B], target[y][x_B]);
		target[y_B][x] = Min(col[y_B][k] + row[k][x], target[y_B][x]);
		target[y_B][x_B] = Min(col[y_B][k] + row[k][x_B], target[y_B][x_B]);
	}
	dst[target_top_left] = target[y][x];
	dst[target_top_right] = target[y][x_B];
	dst[target_bottom_left] = target[y_B][x];
	dst[target_bottom_right] = target[y_B][x_B];
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    
    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    block_FW();
    output(argv[2]);
    return 0;
}