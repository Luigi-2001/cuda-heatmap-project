#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_KERNEL_RADIUS 100
__constant__ float c_kernel[2 * MAX_KERNEL_RADIUS + 1];

// ==================== MACRO PER ERROR CHECKING ====================
#define CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== PROTOTIPI DEI KERNEL (ESSENZIALE!) ====================
__global__ void gaussian_blur_orizzontal_GPU(
    const unsigned int *input,
    float *output,
    int W, int H,
    const float *kernel,
    int r
);

__global__ void gaussian_blur_vertical_GPU(
    const float *input,
    float *output,
    int W, int H,
    const float *kernel,
    int r
);

// ==================== KERNEL GAUSSIANO ====================
void generate_gaussian_kernel(float *kernel, int r, float sigma) {
    float sum = 0.0f;
    int size = 2 * r + 1;
    
    for (int i = -r; i <= r; i++) {
        float x = (float)i;
        kernel[i + r] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i + r];
    }
    
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

// ==================== GENERAZIONE IMMAGINE TEST ====================
void generate_test_image(unsigned int *image, int W, int H, unsigned int seed) {
    srand(seed);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            unsigned int val = ((x / 16 + y / 16) % 2 == 0) ? 255 : 0;
            val += rand() % 20;
            image[y * W + x] = val;
        }
    }
}

// ==================== VERSIONE CPU PER VERIFICA ====================
void gaussian_blur_separable_cpu(
    const unsigned int *input,
    float *output,
    int W, int H,
    const float *kernel,
    int r
) {
    float *temp = (float*)calloc(W * H, sizeof(float));
    
    // Passata orizzontale con CLAMPING (nessuna normalizzazione!)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum_val = 0.0f;
            for (int dx = -r; dx <= r; dx++) {
                // ✅ CLAMPING invece di clipping
                int ix = x + dx;
                ix = (ix < 0) ? 0 : (ix >= W) ? W - 1 : ix;  // Replica bordo sinistro/destro
                sum_val += (float)input[y * W + ix] * kernel[dx + r];
            }
            temp[y * W + x] = sum_val;  // ✅ Nessuna divisione!
        }
    }

    // Passata verticale con CLAMPING
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            float sum_val = 0.0f;
            for (int dy = -r; dy <= r; dy++) {
                int iy = y + dy;
                iy = (iy < 0) ? 0 : (iy >= H) ? H - 1 : iy;  // Replica bordo alto/basso
                sum_val += temp[iy * W + x] * kernel[dy + r];
            }
            output[y * W + x] = sum_val;  // ✅ Nessuna divisione!
        }
    }
    free(temp);
}

// ==================== VERIFICA RISULTATI ====================
int verify_heatmap_float(const float *heat_cpu, const float *heat_gpu, int W, int H, float tolerance) {
    int total_pixels = W * H;
    int different_pixels = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < total_pixels; i++) {
        float diff = fabsf(heat_cpu[i] - heat_gpu[i]); // fabsf per float!
        if (diff > tolerance) {
            different_pixels++;
            if (diff > max_diff) max_diff = diff;
        }
    }

    if (different_pixels == 0) {
        printf("✅ Heatmap: identiche (tolleranza = %e)\n", tolerance);
        return 1;
    } else {
        float percentage = (float)different_pixels / total_pixels * 100.0f;
        printf("❌ Heatmap: DIFFERENZE RILEVATE\n");
        printf("   Pixel diversi: %d / %d (%.3f%%)\n", different_pixels, total_pixels, percentage);
        printf("   Differenza massima: %e\n", max_diff);
        printf("   Tolleranza usata: %e\n", tolerance);
        return 0;
    }
}

// ==================== KERNEL CUDA ====================
__global__ void gaussian_blur_horizontal_GPU(
    const unsigned int *input,
    float *output,
    int W, int H,
    const float *kernel,
    int r
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    float sum_val = 0.0f;

    for (int dx = -r; dx <= r; dx++) {
        int x = ix + dx;
        if (x >= 0 && x < W) {
            x = (x < 0) ? 0 : (x >= W) ? W - 1 : x; // CLAMP
            sum_val += (float)input[iy * W + x] * kernel[dx + r];
        }
    }

    output[iy * W + ix] = sum_val;
}

__global__ void gaussian_blur_vertical_GPU(
    const float *input,
    float *output,
    int W, int H,
    const float *kernel,
    int r
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;

    float sum_val = 0.0f;
    float norm = 0.0f;

    for (int dy = -r; dy <= r; dy++) {
        int y = iy + dy;
        if (y >= 0 && y < H) {
            float w = kernel[dy + r];
            sum_val += input[y * W + ix] * w;
            norm += w;
        }
    }

    output[iy * W + ix] = (norm > 0) ? sum_val / norm : input[iy * W + ix];
}
__global__ void gaussian_blur_horizontal_GPU_opt(
    const unsigned int *input,
    float *output,
    int W, int H,
    int r
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;  // blockDim.y = 1 per configurazione 1D
    
    if (ix >= W || iy >= H) return;

    float sum_val = 0.0f;
    
    #pragma unroll
    for (int dx = -r; dx <= r; dx++) {
        // ✅ CLAMPING invece di clipping
        int x = ix + dx;
        x = x < 0 ? 0 : x;
        x = x >= W ? W - 1 : x;
        sum_val = __fmaf_rn(c_kernel[dx+r],(float)input[iy * W + x],sum_val);
    }

    output[iy * W + ix] = sum_val;  // ✅ Nessuna divisione!
}

__global__ void gaussian_blur_horizontal_GPU_tiled(
    const unsigned int *input,
    float *output,
    int W,
    int H,
    int r
) {
    extern __shared__ float tile[];  // 1D: solo la riga + halo
    
    int tx = threadIdx.x;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y;
    
    /*// === CARICAMENTO COALESCED (1D) ===
    if (tx < blockDim.x + 2 * r) {
        int src_x = gx - r + tx;
        src_x = (src_x < 0) ? 0 : (src_x >= W) ? W - 1 : src_x;
        tile[tx] = (float)input[gy * W + src_x];  // ✅ Coalesced!
    }*/
    // === CARICAMENTO COALESCED CORRETTO ===
    // Dobbiamo caricare (blockDim.x + 2*r) elementi.
    // Usiamo un ciclo for per coprire l'halo se il blocco è piccolo o il raggio è grande.
    int input_start_x = blockIdx.x * blockDim.x - r;
    for (int i = tx; i < blockDim.x + 2 * r; i += blockDim.x) {
        
        // Clamping
        int src_x = input_start_x + i;
        //src_x = (src_x < 0) ? 0 : (src_x >= W) ? W - 1 : src_x;
        // ✅ Più efficiente (istruzioni native GPU)
        src_x = max(0, min(W - 1, src_x));
        
        // Lettura coalesced: thread 'i' legge 'input_start + i'
        // Non c'è più il "2 * tx"
        tile[i] = (float)input[gy * W + src_x];
    }
    __syncthreads();
    
    // === CALCOLO IN SHARED MEMORY ===
    if (gx < W) {
        float sum = 0.0f;
        #pragma unroll
        for (int dx = -r; dx <= r; dx++) {
            sum = __fmaf_rn(tile[tx + r + dx], c_kernel[dx + r], sum);
        }
        output[gy * W + gx] = sum;
    }
}
__global__ void gaussian_blur_vertical_GPU_tiled(
    const float *input,
    float *output,
    int W,
    int H,
    int r
) {
    extern __shared__ float tile_1d[];
    #define TILE(i, j) tile_1d[(i) * (blockDim.x) + (j)]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    int base_x = gx - tx;
    int base_y = gy - ty - r;
    
    // === CARICAMENTO CON CLAMPING (non clipping!) ===
    for (int i = ty; i < blockDim.y + 2 * r; i += blockDim.y) {
        int src_y = base_y + i;
        // ✅ CLAMPING VERTICALE per halo
        //src_y = (src_y < 0) ? 0 : (src_y >= H) ? H - 1 : src_y;
        src_y = max(0, min(H - 1, src_y));
        
        for (int j = tx; j < blockDim.x; j += blockDim.x) {
            int src_x = base_x + j;
            // ✅ CLAMPING ORIZZONTALE per halo
            //src_x = (src_x < 0) ? 0 : (src_x >= W) ? W - 1 : src_x;
            src_x = max(0, min(W - 1, src_x));
            
            TILE(i, j) = input[src_y * W + src_x];  // ✅ Nessun padding a zero!
        }
    }
    __syncthreads();
    
    // === CALCOLO SENZA NORMALIZZAZIONE (kernel già normalizzato) ===
    if (gx < W && gy < H) {
        float sum = 0.0f;
        #pragma unroll
        for (int dy = -r; dy <= r; dy++) {
            sum = __fmaf_rn(TILE(ty + r + dy, tx), c_kernel[dy + r], sum);
        }
        output[gy * W + gx] = sum;  // ✅ Nessuna divisione!
    }
    
    #undef TILE
}

// ==================== KERNEL CUDA CON TEXTURE OBJECT API ====================
__global__ void gaussian_blur_orizzontal_GPU_texture(
    cudaTextureObject_t texObj,  // ✅ Texture Object come parametro
    float *output,
    int W, int H,
    int r
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= W || iy >= H) return;

    float sum_val = 0.0f;
    
    #pragma unroll
    for (int dx = -r; dx <= r; dx++) {
        // ✅ TEXTURE OBJECT: tex2D con texture object (non deprecato)
        float val = (float)tex2D<unsigned int>(texObj, ix + dx, iy);
        sum_val = __fmaf_rn(val, c_kernel[dx + r], sum_val);
    }

    output[iy * W + ix] = sum_val;
}

__global__ void gaussian_blur_vertical_GPU_texture(
    cudaTextureObject_t texObj,  // ✅ Texture Object come parametro
    float *output,
    int W, int H,
    int r
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= W || iy >= H) return;

    float sum_val = 0.0f;
    
    #pragma unroll
    for (int dy = -r; dy <= r; dy++) {
        // ✅ TEXTURE OBJECT: tex2D con texture object (non deprecato)
        float val = tex2D<float>(texObj, ix, iy + dy);
        sum_val = __fmaf_rn(val, c_kernel[dy + r], sum_val);
    }

    output[iy * W + ix] = sum_val;
}


// ==================== FUNZIONE DI TEST SINGOLO ====================
int run_single_test(int W, int H, float sigma, int block_x, int block_y) {
    int r = (int)ceilf(3.0f * sigma);
    if (r < 1) r = 1;
    int kernel_size = 2 * r + 1;

    // Allocazione host
    unsigned int *h_input = (unsigned int*)malloc(W * H * sizeof(unsigned int));
    float *h_kernel = (float*)malloc(kernel_size * sizeof(float));
    float *h_output_cpu = (float*)malloc(W * H * sizeof(float));
    float *h_output_gpu = (float*)malloc(W * H * sizeof(float));
    
    if (!h_input || !h_kernel || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Errore allocazione memoria host\n");
        exit(EXIT_FAILURE);
    }

    // Generazione dati
    generate_test_image(h_input, W, H, 42);
    generate_gaussian_kernel(h_kernel, r, sigma);
    clock_t t1=clock();
    gaussian_blur_separable_cpu(h_input, h_output_cpu, W, H, h_kernel, r);
    clock_t t2=clock();
    printf("Tempo gaussian blur cpu: %f secondi\n", ((double)(t2-t1)/CLOCKS_PER_SEC));
    // Allocazione device
    unsigned int *d_input = NULL;  // CORRETTO: NULL invece di nullptr
    float *d_temp = NULL;
    float *d_output = NULL;
    float *d_kernel = NULL;

    CHECK(cudaMalloc(&d_input, W * H * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_temp, W * H * sizeof(float)));
    CHECK(cudaMalloc(&d_output, W * H * sizeof(float)));
    CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, h_input, W * H * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_kernel, d_kernel, (2 * r + 1) * sizeof(float)));

    // ==================== CREAZIONE TEXTURE OBJECT PER INPUT ====================
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>();
    cudaArray *d_inputArray = NULL;
    CHECK(cudaMallocArray(&d_inputArray, &channelDesc, W, H));
    CHECK(cudaMemcpy2DToArray(d_inputArray, 0, 0, d_input, W * sizeof(unsigned int), 
                             W * sizeof(unsigned int), H, cudaMemcpyDeviceToDevice));
    
    // Descrittore risorsa (punta all'array)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_inputArray;
    
    // Descrittore texture (configurazione)
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;  // x-axis: replica bordi
    texDesc.addressMode[1] = cudaAddressModeClamp;  // y-axis: replica bordi
    texDesc.filterMode = cudaFilterModePoint;       // Nessuna interpolazione
    texDesc.readMode = cudaReadModeElementType;     // Leggi come tipo elemento
    texDesc.normalizedCoords = 0;                   // Coordinate intere (non normalizzate)
    
    // Creazione texture object
    cudaTextureObject_t inputTexObj = 0;
    CHECK(cudaCreateTextureObject(&inputTexObj, &resDesc, &texDesc, NULL));

    // Configurazione
    dim3 blockDim(block_x*block_x, block_y);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, H);
    dim3 blockDimV(block_x, block_x);
    dim3 gridDimV((W + blockDimV.x - 1) / blockDimV.x, 
                 (H + blockDimV.y - 1) / blockDimV.y);
    // Esecuzione GPU CON CONTROLLO ERRORI DOPO IL LANCI0
    gaussian_blur_horizontal_GPU<<<gridDim, blockDim>>>(d_input, d_temp, W, H, d_kernel, r);
    CHECK(cudaGetLastError());  // ESSENZIALE!
    CHECK(cudaDeviceSynchronize()); 

    gaussian_blur_vertical_GPU<<<gridDim, blockDim>>>(d_temp, d_output, W, H, d_kernel, r);
    CHECK(cudaGetLastError());  // ESSENZIALE!
    CHECK(cudaDeviceSynchronize()); 
    
    gaussian_blur_horizontal_GPU_opt<<<gridDim, blockDim>>>(d_input, d_temp, W, H, r);
    CHECK(cudaGetLastError());  // ESSENZIALE!
    CHECK(cudaDeviceSynchronize()); 

    size_t shared_h = (blockDim.x + 2 * r) * sizeof(float);
    gaussian_blur_horizontal_GPU_tiled<<<gridDim, blockDim, shared_h>>>(d_input, d_temp, W, H, r);
    CHECK(cudaGetLastError());  // ESSENZIALE!
    CHECK(cudaDeviceSynchronize()); 

    size_t shared = (blockDimV.y + 2 * r) * blockDimV.x * sizeof(float);
    gaussian_blur_vertical_GPU_tiled<<<gridDimV, blockDimV, shared>>>(d_temp, d_output, W, H, r);
    CHECK(cudaGetLastError());  // ESSENZIALE!
    CHECK(cudaDeviceSynchronize()); 

     gaussian_blur_orizzontal_GPU_texture<<<gridDim, blockDim>>>(inputTexObj,d_temp, W, H, r);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // ==================== CREAZIONE TEXTURE OBJECT PER BUFFER TEMPORANEO ====================
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
    cudaArray *d_tempArray = NULL;
    CHECK(cudaMallocArray(&d_tempArray, &channelDescFloat, W, H));
    CHECK(cudaMemcpy2DToArray(d_tempArray, 0, 0, d_temp, W * sizeof(float), 
                             W * sizeof(float), H, cudaMemcpyDeviceToDevice));
    
    cudaResourceDesc resDescTemp = {};
    resDescTemp.resType = cudaResourceTypeArray;
    resDescTemp.res.array.array = d_tempArray;
    
    cudaTextureObject_t tempTexObj = 0;
    CHECK(cudaCreateTextureObject(&tempTexObj, &resDescTemp, &texDesc, NULL));
    // Passata verticale
    gaussian_blur_vertical_GPU_texture<<<gridDimV, blockDimV>>>(tempTexObj,d_output, W, H, r);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());


    CHECK(cudaMemcpy(h_output_gpu, d_output, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    

    // Verifica
    int passed = verify_heatmap_float(h_output_cpu, h_output_gpu, W, H, 1e-4f);

    // Pulizia
    cudaDestroyTextureObject(inputTexObj);
    cudaDestroyTextureObject(tempTexObj);
    cudaFreeArray(d_inputArray);
    cudaFreeArray(d_tempArray);
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_gpu);

    return passed;
}

// ==================== MAIN CON BATTERIA DI TEST ====================
int main() {
    printf("============================================================\n");
    printf("BATTERIA DI TEST: Gaussian Blur Separabile CUDA\n");
    printf("============================================================\n\n");

    const int sizes[] = {1920, 3840};
    const int sizesH[] = {1080, 2160};
    const float sigmas[] = {6.0f, 12.0f};
    const int block_dims[] = {16, 32};

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_sigmas = sizeof(sigmas) / sizeof(sigmas[0]);
    int num_blocks = sizeof(block_dims) / sizeof(block_dims[0]);

    int total = 0, passed = 0;

    printf("%-12s | %-6s | %-3s | %-10s | %-6s\n", "DIMENSIONE", "SIGMA", "R", "BLOCCHI", "ESITO");
    printf("-------------+--------+-----+------------+--------\n");

    for (int s = 0; s < num_sizes; s++) {
        int W = sizes[s];
        int H = sizesH[s];
        
        for (int sg = 0; sg < num_sigmas; sg++) {
            float sigma = sigmas[sg];
            int r = (int)ceilf(3.0f * sigma);
            
            for (int b = 0; b < num_blocks; b++) {
                int bx = block_dims[b];
                int by = 1;
                
                int test_passed = run_single_test(W, H, sigma, bx, by);
                total++;
                if (test_passed) passed++;
                
                printf("%4dx%-4d    | %-6.1f | %-3d | %-2dx%-2d      | %s\n",
                       W, H, sigma, r, bx, by, test_passed ? "PASS" : "FAIL");
            }
        }
    }

    printf("\n============================================================\n");
    printf("RISULTATO FINALE: %d/%d test superati (%.1f%%)\n", passed, total, (float)passed/total*100.0f);
    printf("============================================================\n");

    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}