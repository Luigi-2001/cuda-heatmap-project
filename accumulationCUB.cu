#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cub/cub.cuh>

// ==================== COSTANTI DI CONFIGURAZIONE ====================
#define MARGIN 10  // Margine per la heat map

// ==================== MACRO PER ERROR CHECKING ====================
#define CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== VERSIONE CPU DI RIFERIMENTO ====================
void accumulate_points_cpu(
    unsigned int *heat_map,
    const float *x_coords,
    const float *y_coords,
    long num_points,
    int W, int H
) {
    // Reset heat map
    memset(heat_map, 0, W * H * sizeof(unsigned int));
    
    // Accumula punti
    for (long i = 0; i < num_points; i++) {
        int ix = MARGIN + (int)(x_coords[i] * (W - 2 * MARGIN - 1));
        int iy = MARGIN + (int)(y_coords[i] * (H - 2 * MARGIN - 1));
        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
            heat_map[iy * W + ix] += 1;
        }
    }
}

// ==================== KERNEL GPU BASELINE (1 thread = 1 punto) ====================
__global__ void accumulate_points_gpu_baseline(
    unsigned int *heat_map,
    const float *x_coords,
    const float *y_coords,
    long num_points,
    int W, int H
) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Calcola coordinate pixel
    int ix = MARGIN + (int)(x_coords[idx] * (W - 2 * MARGIN - 1));
    int iy = MARGIN + (int)(y_coords[idx] * (H - 2 * MARGIN - 1));
    
    // Accumula con atomica (race condition protection)
    if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
        atomicAdd(&heat_map[iy * W + ix], 1u);
    }
}

// ==================== KERNEL GPU OTTIMIZZATO (SORT + RLE) ====================

// 1. Kernel per calcolare l'indice del pixel per ogni punto
__global__ void compute_pixel_indices(
    const float* x, const float* y, 
    int* d_pixel_indices, 
    long num_points, int W, int H) 
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int ix = MARGIN + (int)(x[idx] * (W - 2 * MARGIN - 1));
        int iy = MARGIN + (int)(y[idx] * (H - 2 * MARGIN - 1));
        
        if (ix >= 0 && ix < W && iy >= 0 && iy < H)
            d_pixel_indices[idx] = iy * W + ix;
        else
            d_pixel_indices[idx] = -1; // Punto fuori dai bordi
    }
}

// 2. Kernel finale per la versione RLE: scrive i conteggi
// Non usa atomicAdd perché ogni thread scrive su un pixel diverso!
__global__ void render_rle_heatmap(
    unsigned int* heat_map,
    const int* unique_indices,
    const int* counts,
    int num_runs) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_runs) {
        int pixel_idx = unique_indices[idx];
        // Ignora i punti che erano fuori mappa (-1)
        if (pixel_idx >= 0) {
            heat_map[pixel_idx] = (unsigned int)counts[idx];
        }
    }
}

// ==================== GENERAZIONE DATI DI TEST ====================
void generate_test_points(
    float *x_coords,
    float *y_coords,
    long num_points,
    unsigned int seed
) {
    srand(seed);
    for (long i = 0; i < num_points; i++) {
        // Distribuzione gaussiana centrata (simula tracking)
        x_coords[i] = 0.5f + 0.2f * (rand() / (float)RAND_MAX - 0.5f);
        y_coords[i] = 0.5f + 0.2f * (rand() / (float)RAND_MAX - 0.5f);
        
        // Clamp a [0, 1]
        x_coords[i] = fmaxf(0.0f, fminf(1.0f, x_coords[i]));
        y_coords[i] = fmaxf(0.0f, fminf(1.0f, y_coords[i]));
    }
}

// ==================== VERIFICA RISULTATI ====================
int verify_heatmap(
    const unsigned int *heat_cpu,
    const unsigned int *heat_gpu,
    int W, int H,
    long num_points
) {
    long total_diff = 0;
    int max_diff = 0;
    int errors = 0;
    
    for (int i = 0; i < W * H; i++) {
        int diff = abs((int)heat_cpu[i] - (int)heat_gpu[i]);
        if (diff > 0) {
            errors++;
            total_diff += diff;
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    if (errors == 0) {
        printf("✅ Heatmap: identiche (%ld punti)\n", num_points);
        return 1;
    } else {
        float error_pct = (float)errors / (W * H) * 100.0f;
        printf("❌ Heatmap: DIFFERENZE RILEVATE\n");
        printf("   Pixel diversi: %d / %d (%.3f%%)\n", errors, W * H, error_pct);
        printf("   Differenza massima: %d\n", max_diff);
        printf("   Differenza totale: %ld\n", total_diff);
        return 0;
    }
}


// ==================== FUNZIONE DI TEST SINGOLO ====================
int run_single_test(int W, int H, long num_points, int block_size) {
    printf("\nTest: %dx%d heat map, %ld punti, block=%d\n", W, H, num_points, block_size);
    
    // Allocazione host
    unsigned int *h_heat_cpu = (unsigned int*)calloc(W * H, sizeof(unsigned int));
    unsigned int *h_heat_gpu = (unsigned int*)calloc(W * H, sizeof(unsigned int));
    float *h_x = (float*)malloc(num_points * sizeof(float));
    float *h_y = (float*)malloc(num_points * sizeof(float));
    
    if (!h_heat_cpu || !h_heat_gpu || !h_x || !h_y) {
        fprintf(stderr, "Errore allocazione memoria host\n");
        exit(EXIT_FAILURE);
    }
    
    // Generazione dati
    generate_test_points(h_x, h_y, num_points, 42);
    
    // ===== VERSIONE CPU =====
    clock_t t0 = clock();
    accumulate_points_cpu(h_heat_cpu, h_x, h_y, num_points, W, H);
    clock_t t1 = clock();
    double cpu_time = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time accumulation: %.2f ms\n", cpu_time);
    
    // ===== VERSIONE GPU BASELINE =====
    unsigned int *d_heat = NULL;
    float *d_x = NULL;
    float *d_y = NULL;
    
    CHECK(cudaMalloc(&d_heat, W * H * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_x, num_points * sizeof(float)));
    CHECK(cudaMalloc(&d_y, num_points * sizeof(float)));
    
    CHECK(cudaMemcpy(d_x, h_x, num_points * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, num_points * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockDim1D(block_size*block_size, 1);
    dim3 gridDim1D((num_points + blockDim1D.x - 1) / blockDim1D.x, 1);
    dim3 blockDim2D(block_size, block_size);
    dim3 gridDim2D((W + blockDim2D.x - 1) / blockDim2D.x, (H + blockDim2D.y - 1) / blockDim2D.y);

    
    // Reset heat map
    CHECK(cudaMemset(d_heat, 0, W * H * sizeof(unsigned int)));
    
    // Esecuzione GPU baseline
    accumulate_points_gpu_baseline<<<gridDim1D, blockDim1D>>>(d_heat, d_x, d_y, num_points, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    // Copia risultato
    CHECK(cudaMemcpy(h_heat_gpu, d_heat, W * H * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Verifica
    int passed_baseline = verify_heatmap(h_heat_cpu, h_heat_gpu, W, H, num_points);
    printf("Baseline verification: %s\n", passed_baseline ? "PASS" : "FAIL");
    
    
    // Allocazioni temporanee
    int *d_indices_in, *d_indices_sorted;
    int *d_unique_out, *d_counts_out, *d_num_runs_out;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CHECK(cudaMalloc(&d_indices_in, num_points * sizeof(int)));
    CHECK(cudaMalloc(&d_indices_sorted, num_points * sizeof(int)));
    CHECK(cudaMalloc(&d_unique_out, num_points * sizeof(int)));
    CHECK(cudaMalloc(&d_counts_out, num_points * sizeof(int)));
    CHECK(cudaMalloc(&d_num_runs_out, sizeof(int)));
    // 1. Calcolo Indici
    compute_pixel_indices<<<gridDim1D, blockDim1D>>>(d_x, d_y, d_indices_in, num_points, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 2. Ordinamento (Radix Sort)
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_indices_in, d_indices_sorted, num_points);
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_indices_in, d_indices_sorted, num_points);

    // 3. Compressione (Run-Length Encode)
    // Nota: riutilizziamo d_temp_storage se abbastanza grande, altrimenti riallocare. Per sicurezza ricalcolo:
    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes, d_indices_sorted, d_unique_out, d_counts_out, d_num_runs_out, num_points);
    CHECK(cudaFree(d_temp_storage)); // Libero il vecchio
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_indices_sorted, d_unique_out, d_counts_out, d_num_runs_out, num_points);
    // 4. Scrittura Finale
    int h_num_runs;
    CHECK(cudaMemcpy(&h_num_runs, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Passo 3: Accumulazione
    cudaMemset(d_heat, 0, W * H * sizeof(unsigned int));
    int rle_blocks = (h_num_runs + blockDim1D.x - 1) / blockDim1D.x; // Calcolata su h_num_runs!
    render_rle_heatmap<<<rle_blocks, blockDim1D>>>(d_heat, d_unique_out, d_counts_out, h_num_runs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    // Copia risultato
    CHECK(cudaMemcpy(h_heat_gpu, d_heat, W * H * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
    // Verifica
    int passed_opt = verify_heatmap(h_heat_cpu, h_heat_gpu, W, H, num_points);
    printf("Sorted verification: %s\n", passed_opt ? "PASS" : "FAIL");

    //Versione histogram
    // Reset mappa (HistogramEven aggiunge, non resetta)
    CHECK(cudaMemset(d_heat, 0, W * H * sizeof(unsigned int)));

    // 1. Chiedi a CUB quanta memoria serve per l'istogramma
    size_t hist_temp_bytes = 0;
    cub::DeviceHistogram::HistogramEven(NULL, hist_temp_bytes, 
        d_indices_sorted, d_heat, W * H + 1, 0, W * H, num_points);

    // 2. Libera la memoria temporanea usata in precedenza (Sort/RLE) per evitare leak
    if (d_temp_storage) CHECK(cudaFree(d_temp_storage));

    // 3. Alloca la nuova dimensione necessaria
    CHECK(cudaMalloc(&d_temp_storage, hist_temp_bytes));

    // 4. Esegui l'istogramma (ora d_temp_storage è valido e della taglia giusta)
    cub::DeviceHistogram::HistogramEven(d_temp_storage, hist_temp_bytes, 
        d_indices_sorted, d_heat, W * H + 1, 0, W * H, num_points);
    
    // Copia risultato
    CHECK(cudaMemcpy(h_heat_gpu, d_heat, W * H * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int passed_isto = verify_heatmap(h_heat_cpu, h_heat_gpu, W, H, num_points);
    printf("Histogram verification: %s\n", passed_isto ? "PASS" : "FAIL");

    // Pulizia
    cudaFree(d_indices_in); cudaFree(d_indices_sorted);
    cudaFree(d_unique_out); cudaFree(d_counts_out);
    cudaFree(d_num_runs_out); cudaFree(d_temp_storage);

    // Pulizia
    cudaFree(d_heat);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_heat_cpu);
    free(h_heat_gpu);
    free(h_x);
    free(h_y);
    
    return passed_baseline && passed_opt && passed_isto;
}

// ==================== MAIN CON BATTERIA DI TEST ====================
int main() {
    printf("============================================================\n");
    printf("BATTERIA DI TEST: Accumulazione Punti su Heat Map CUDA\n");
    printf("============================================================\n\n");
    
    const int sizes[] = {1920, 3840};
    const int sizesH[] = {1080, 2160};
    const long num_points[] = {100000, 1000000, 10000000};
    const int block_sizes[] = {16, 32};
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_points_configs = sizeof(num_points) / sizeof(num_points[0]);
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    int total = 0, passed = 0;
    
    printf("%-10s | %-12s | %-8s | %-8s | %-6s\n", 
           "DIMENSIONE", "PUNTI", "BLOCK", "TEMPO", "ESITO");
    printf("-----------+--------------+----------+----------+--------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int W = sizes[s];
        int H = sizesH[s];
        
        for (int p = 0; p < num_points_configs; p++) {
            long points = num_points[p];
            
            for (int b = 0; b < num_blocks; b++) {
                int block = block_sizes[b];
                
                int test_passed = run_single_test(W, H, points, block);
                total++;
                if (test_passed) passed++;
                
                printf("%4dx%-4d | %10ld   | %-8d | %-8s | %s\n",
                       W, H, points, block, "-", test_passed ? "PASS" : "FAIL");
            }
        }
    }
    
    printf("\n============================================================\n");
    printf("RISULTATO FINALE: %d/%d test superati (%.1f%%)\n", passed, total, (float)passed/total*100.0f);
    printf("============================================================\n");
    
    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}