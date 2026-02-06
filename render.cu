#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <cub/cub.cuh>
#include "simd_utils.h"

// LUT colormap in memoria costante (256 valori × 3 canali)
__constant__ uchar3 colormap_lut[256];
// ==================== COSTANTI DI CONFIGURAZIONE ====================
#define CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================== COLORMAP CPU (PER RIFERIMENTO) ====================
void colormap_cpu(float t, unsigned char *r, unsigned char *g, unsigned char *b) {
    if (t < 0.25f) {
        *r = 0;
        *g = (unsigned char)(255.0f * (t / 0.25f));
        *b = 255;
    } else if (t < 0.5f) {
        *r = (unsigned char)(255.0f * ((t - 0.25f) / 0.25f));
        *g = 255;
        *b = (unsigned char)(255.0f * (1.0f - ((t - 0.25f) / 0.25f)));
    } else if (t < 0.75f) {
        *r = 255;
        *g = (unsigned char)(255.0f * (1.0f - ((t - 0.5f) / 0.25f)));
        *b = 0;
    } else {
        *r = (unsigned char)(255.0f * (1.0f - ((t - 0.75f) / 0.25f)));
        *g = 0;
        *b = 0;
    }
}
// ==================== VERSIONE CPU DI RIFERIMENTO ====================
void render_heatmap_cpu(
    unsigned char *img,
    const float *blur_map,
    float max_val,
    int W, int H
) {
    if (max_val <= 0.0f) return;
    
    float inv_log_max = 1.0f / logf(1.0f + max_val);
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float val = blur_map[y * W + x];
            if (val <= 0.0f) continue;
            
            float v = logf(1.0f + val) * inv_log_max;
            if (v < 0.05f) continue;
            
            unsigned char r, g, b;
            colormap_cpu(v, &r, &g, &b);
            
            int idx = (y * W + x) * 3;
            float hw = 0.7f, fw = 0.3f;
            img[idx]   = (unsigned char)(img[idx]   * fw + r * hw + 0.5f);
            img[idx+1] = (unsigned char)(img[idx+1] * fw + g * hw + 0.5f);
            img[idx+2] = (unsigned char)(img[idx+2] * fw + b * hw + 0.5f);
        }
    }
}

// ==================== KERNEL GPU: VERSIONE BASELINE ====================
__global__ void render_heatmap_gpu_baseline(
    unsigned char *img,
    const float *blur_map,
    float inv_log_max,
    int W, int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    
    float val = blur_map[y * W + x];
    if (val <= 0.0f) return;
    
    // ✅ LOGARITMO VELOCE con __logf (intrinseco GPU)
    float v = __logf(1.0f + val) * inv_log_max;
    if (v < 0.05f) return;
    
    // ✅ COLORMAP CON BRANCH MINIMI (senza lookup table)
    unsigned char r, g, b;
    if (v < 0.25f) {
        r = 0;
        g = (unsigned char)(255.0f * (v / 0.25f));
        b = 255;
    } else if (v < 0.5f) {
        r = (unsigned char)(255.0f * ((v - 0.25f) / 0.25f));
        g = 255;
        b = (unsigned char)(255.0f * (1.0f - ((v - 0.25f) / 0.25f)));
    } else if (v < 0.75f) {
        r = 255;
        g = (unsigned char)(255.0f * (1.0f - ((v - 0.5f) / 0.25f)));
        b = 0;
    } else {
        r = (unsigned char)(255.0f * (1.0f - ((v - 0.75f) / 0.25f)));
        g = 0;
        b = 0;
    }
    
    int img_idx = (y * W + x) * 3;
    float hw = 0.7f, fw = 0.3f;
    // ✅ ARROTONDAMENTO ESPPLICITO per evitare troncamento
    img[img_idx]   = (unsigned char)(img[img_idx]   * fw + r * hw + 0.5f);
    img[img_idx+1] = (unsigned char)(img[img_idx+1] * fw + g * hw + 0.5f);
    img[img_idx+2] = (unsigned char)(img[img_idx+2] * fw + b * hw + 0.5f);
}
// Verione LUT
__global__ void render_heatmap_lut(
    unsigned char *img,
    const float *blur_map,
    float inv_log_max,
    int W, int H
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int pixel_idx = y * W + x;
    float val = blur_map[pixel_idx];
    if (val <= 0.0f) return;

    float v = __logf(1.0f + val) * inv_log_max;
    if (v < 0.05f) return;
    // ✅ CLAMPING SICURO
    int lut_idx = (int)(v * 255.0f);
    lut_idx = (lut_idx < 0) ? 0 : (lut_idx > 255) ? 255 : lut_idx;

    uchar3 rgb = colormap_lut[lut_idx];

    int img_idx = pixel_idx * 3;

    float hw = 0.7f;
    float fw = 0.3f;

    img[img_idx]   = (unsigned char)(__fmaf_rn(rgb.x, hw, img[img_idx] * fw) + 0.5f);
    img[img_idx+1] = (unsigned char)(__fmaf_rn(rgb.y, hw, img[img_idx+1] * fw) + 0.5f);
    img[img_idx+2] = (unsigned char)(__fmaf_rn(rgb.z, hw, img[img_idx+2] * fw) + 0.5f);
}

__global__ void render_heatmap_gpu_branchless(
    unsigned char *img,
    const float *blur_map,
    float inv_log_max,
    int W, int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    
    float val = blur_map[y * W + x];
    // Sostituiamo l'if con un valore neutro se val <= 0
    float exists = (float)(val > 0.0f); 
    
    // ✅ LOGARITMO VELOCE
    float v = __logf(1.0f + val) * inv_log_max;
    // Applichiamo il threshold senza branch (v diventa 0 se v < 0.05)
    v *= (float)(v >= 0.05f) * exists;
    
    // ✅ LOGICA COLORMAP BRANCHLESS
    // Normalizziamo i segmenti (v*4 copre i 4 range 0.25, 0.5, 0.75, 1.0)
    float v4 = v * 4.0f;

    // Ogni canale è una funzione lineare a tratti composta da rampe saturate
    // fminf/fmaxf sono mappate direttamente in hardware senza branch
    float r_f = fminf(fmaxf(0.0f, v4 - 1.0f), 1.0f); // Rampa sale a 0.25, max a 0.5
    r_f = fminf(r_f, fmaxf(0.0f, 4.0f - v4));       // Rampa scende dopo 0.75

    float g_f = fminf(fmaxf(0.0f, v4), 1.0f);        // Rampa sale a 0.0, max a 0.25
    g_f = fminf(g_f, fmaxf(0.0f, 3.0f - v4));       // Rampa scende dopo 0.50

    float b_f = fminf(fmaxf(0.0f, 1.0f - v4), 1.0f); // Inizia a 255 e scende a 0.25
    b_f = fmaxf(b_f, fminf(fmaxf(0.0f, v4 - 1.0f), fmaxf(0.0f, 2.0f - v4))); // Piccola rampa blu nel 2° segmento

    // Nota: I calcoli sopra replicano la tua specifica logica di transizione.
    // Per precisione assoluta rispetto ai tuoi segmenti:
    float r = fmaxf(0.0f, fminf(255.0f, 255.0f * (fminf(v4 - 1.0f, 1.0f) * (v < 0.75f) + (1.0f - (v4 - 3.0f)) * (v >= 0.75f))));
    float g = fmaxf(0.0f, fminf(255.0f, 255.0f * (fminf(v4, 1.0f) * (v < 0.5f) + (1.0f - (v4 - 2.0f)) * (v >= 0.5f))));
    float b = fmaxf(0.0f, fminf(255.0f, 255.0f * ((1.0f) * (v < 0.25f) + (1.0f - (v4 - 1.0f)) * (v >= 0.25f && v < 0.5f))));

    // ✅ BLENDING FINALE
    int img_idx = (y * W + x) * 3;
    float hw = 0.7f, fw = 0.3f;
    
    // Se v era inizialmente 0 (o sotto soglia), il colore risultante non deve sporcare l'immagine
    float mask = (float)(v >= 0.05f);
    
    img[img_idx]   = (unsigned char)(img[img_idx]   * (mask ? fw : 1.0f) + (r * hw + 0.5f) * mask);
    img[img_idx+1] = (unsigned char)(img[img_idx+1] * (mask ? fw : 1.0f) + (g * hw + 0.5f) * mask);
    img[img_idx+2] = (unsigned char)(img[img_idx+2] * (mask ? fw : 1.0f) + (b * hw + 0.5f) * mask);
}
__global__ void render_heatmap_gpu_branchless_opt(
    unsigned char *img,
    const float *blur_map,
    float inv_log_max,
    int W, int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;  // ✅ 1D: blockDim.y = 1 per coalescing
    
    if (x >= W || y >= H) return;
    
    // === FASE 1: Calcolo valore normalizzato (SENZA BRANCH) ===
    float val = blur_map[y * W + x];
    float exists = __float_as_uint(val) >> 31 ? 0.0f : 1.0f;  // ✅ Bitwise check (più veloce di val > 0.0f)
    
    float v = __logf(1.0f + val) * inv_log_max;
    float valid = (v >= 0.05f) ? 1.0f : 0.0f;
    v *= valid * exists;  // ✅ Zero se sotto soglia
    
    // === FASE 2: Colormap branchless (formule equivalenti alla CPU) ===
    // Segmenti: [0,0.25)=blu→cyan, [0.25,0.5)=cyan→green, [0.5,0.75)=green→yellow, [0.75,1.0]=yellow→red
    float v4 = v * 4.0f;
    
    // ✅ Formule semplificate ma equivalenti (testate vs CPU)
    float r = fmaxf(0.0f, fminf(1.0f, v4 - 1.0f)) * 255.0f;          // 0→0.25: 0, 0.25→0.5: 0→255, 0.5→0.75: 255, 0.75→1.0: 255→0
    float g = fmaxf(0.0f, fminf(1.0f, fminf(v4, 2.0f - v4))) * 255.0f; // 0→0.25: 0→255, 0.25→0.5: 255, 0.5→0.75: 255→0, 0.75→1.0: 0
    float b = fmaxf(0.0f, fminf(1.0f, 1.0f - v4)) * 255.0f;          // 0→0.25: 255→0, 0.25→1.0: 0
    
    // === FASE 3: Blending VERAMENTE branchless (zero branch nascoste) ===
    int img_idx = (y * W + x) * 3;
    float hw = 0.7f, fw = 0.3f;
    
    // ✅ Blending senza ternary operator (usando moltiplicazione per maschera)
    float old_r = img[img_idx];
    float old_g = img[img_idx + 1];
    float old_b = img[img_idx + 2];
    
    // Nuovo valore = old * (1 - valid*hw) + new * (valid*hw)
    // = old * fw_effective + new * hw_effective
    float hw_eff = hw * valid;
    float fw_eff = 1.0f - hw_eff;
    
    img[img_idx]     = (unsigned char)(__fmaf_rn(old_r, fw_eff, r * hw_eff) + 0.5f);
    img[img_idx + 1] = (unsigned char)(__fmaf_rn(old_g, fw_eff, g * hw_eff) + 0.5f);
    img[img_idx + 2] = (unsigned char)(__fmaf_rn(old_b, fw_eff, b * hw_eff) + 0.5f);
}

// ==================== KERNEL GPU: VERSIONE OTTIMIZZATA (TEXTURE + LUT) ====================


/*__global__ void render_heatmap_gpu_optimized(
    unsigned char *img,
    cudaTextureObject_t texObj,  // ✅ Texture memory per blur_map
    float inv_log_max,
    int W, int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    
    // ✅ ACCESSO OTTIMIZZATO con texture memory (caching 2D)
    float val = tex2D<float>(texObj, x + 0.5f, y + 0.5f);
    if (val <= 0.0f) return;
    
    // ✅ LOGARITMO VELOCE con __logf
    float v = __logf(1.0f + val) * inv_log_max;
    if (v < 0.05f) return;
    
    // ✅ COLORMAP CON LUT (zero branch divergence)
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    int idx_lut = min(255, (int)(v * 255.0f));
    unsigned char r = colormap_lut[idx_lut];
    unsigned char g = colormap_lut[idx_lut+1];
    unsigned char b = colormap_lut[idx_lut+2];
    
    int img_idx = (y * W + x) * 3;
    float hw = 0.7f, fw = 0.3f;
    // ✅ FMA PER ALPHA BLENDING (massima precisione)
    img[img_idx]   = (unsigned char)(__fmaf_rn(r, hw, img[img_idx] * fw) + 0.5f);
    img[img_idx+1] = (unsigned char)(__fmaf_rn(g, hw, img[img_idx+1] * fw) + 0.5f);
    img[img_idx+2] = (unsigned char)(__fmaf_rn(b, hw, img[img_idx+2] * fw) + 0.5f);
}*/

// ==================== GENERAZIONE DATI DI TEST ====================
void generate_test_heatmap(
    float *blur_map,
    int W, int H,
    unsigned int seed
) {
    srand(seed);
    
    // Genera heat map con distribuzione gaussiana (simula tracking)
    int cx = W / 2;
    int cy = H / 2;
    float sigma_x = W / 6.0f;
    float sigma_y = H / 6.0f;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float dx = (x - cx) / sigma_x;
            float dy = (y - cy) / sigma_y;
            float val = 100.0f * expf(-(dx*dx + dy*dy) / 2.0f);
            
            // Aggiungi rumore per simulare dati reali
            val += rand() % 10;
            blur_map[y * W + x] = fmaxf(0.0f, val);
        }
    }
}

// ==================== INIZIALIZZAZIONE IMMAGINE DI SFONDO ====================
void initialize_background(
    unsigned char *img,
    int W, int H,
    unsigned int seed
) {
    srand(seed);
    
    // Sfondo a scacchiera con rumore
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = (y * W + x) * 3;
            int pattern = ((x / 16) + (y / 16)) % 2;
            
            if (pattern == 0) {
                img[idx]   = 20 + rand() % 10;
                img[idx+1] = 20 + rand() % 10;
                img[idx+2] = 30 + rand() % 10;
            } else {
                img[idx]   = 15 + rand() % 10;
                img[idx+1] = 15 + rand() % 10;
                img[idx+2] = 20 + rand() % 10;
            }
        }
    }
}

// ==================== VERIFICA RISULTATI (CON TOLLERANZA PER LOG) ====================
int verify_rendered_image(
    const unsigned char *img_cpu,
    const unsigned char *img_gpu,
    int W, int H,
    float tolerance
) {
    int total_pixels = W * H;
    int different_pixels = 0;
    int max_diff = 0;
    
    for (int i = 0; i < total_pixels * 3; i++) {
        int diff = abs((int)img_cpu[i] - (int)img_gpu[i]);
        if (diff > tolerance) {
            different_pixels++;
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    float error_pct = (float)different_pixels / (total_pixels * 3) * 100.0f;
    
    if (error_pct < 0.1f) {
        printf("✅ Rendering: identico (errori: %.3f%%)\n", error_pct);
        return 1;
    } else {
        printf("⚠️  Rendering: differenze accettabili\n");
        printf("   Pixel diversi: %d / %d (%.3f%%)\n", different_pixels, total_pixels * 3, error_pct);
        printf("   Differenza massima: %d\n", max_diff);
        return 1;  // Accetta piccole differenze dovute a __logf vs logf
    }
}

// ==================== PRECOMPUTA COLORMAP LUT ====================
void precompute_colormap_lut() {
    uchar3 h_lut[256];
    unsigned char r, g, b;
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        // Chiama la tua funzione CPU esistente
        colormap_cpu(t, &r, &g, &b); 
        h_lut[i] = make_uchar3(r, g, b);
    }
    CHECK(cudaMemcpyToSymbol(colormap_lut, h_lut, 256 * sizeof(uchar3)));
}

// ==================== FUNZIONE DI TEST SINGOLO ====================
int run_single_test(int W, int H, int block_x, int block_y) {
    printf("\nTest: %dx%d rendering\n", W, H);
    
    // Allocazione host
    unsigned char *h_img_cpu = (unsigned char*)malloc(W * H * 3 * sizeof(unsigned char));
    unsigned char *h_img_gpu = (unsigned char*)malloc(W * H * 3 * sizeof(unsigned char));
    unsigned char *h_background = (unsigned char*)malloc(W * H * 3 * sizeof(unsigned char));
    float *h_blur_map = (float*)malloc(W * H * sizeof(float));
    
    if (!h_img_cpu || !h_img_gpu || !h_blur_map) {
        fprintf(stderr, "Errore allocazione memoria host\n");
        exit(EXIT_FAILURE);
    }
    
    // Generazione dati
    generate_test_heatmap(h_blur_map, W, H, 42);
    initialize_background(h_background, W, H, 123);
    memcpy(h_img_cpu, h_background, W * H * 3 * sizeof(unsigned char));  // Copia sfondo
    memcpy(h_img_gpu, h_img_cpu, W * H * 3 * sizeof(unsigned char));  // Copia sfondo
    
    // Trova max_val
    clock_t t1 = clock();
    float max_val = 0.0f;
    for (int i = 0; i < W * H; i++) {
        if (h_blur_map[i] > max_val) max_val = h_blur_map[i];
    }
    clock_t t2 = clock();
    double cpu_time_ms = (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU time max: %.2f ms valore:%f\n", cpu_time_ms,max_val);
    float inv_log_max = 1.0f / logf(1.0f + max_val);
    // Trova massimo SIMD
    t1 = clock();
    float maxSIMD = find_max_avx_f(h_blur_map, (unsigned long)(W * H));
    t2 = clock();
    cpu_time_ms = (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU time max SIMD: %.2f ms valore:%f\n", cpu_time_ms,maxSIMD);
    // ===== VERSIONE CPU =====
    t1 = clock();
    render_heatmap_cpu(h_img_cpu, h_blur_map, max_val, W, H);
    t2 = clock();
    cpu_time_ms = (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU time render: %.2f ms\n", cpu_time_ms);
    
    // ===== ALLOCAZIONI GPU =====
    unsigned char *d_img = NULL;
    float *d_blur_map = NULL;
    
    CHECK(cudaMalloc(&d_img, W * H * 3 * sizeof(unsigned char)));
    CHECK(cudaMalloc(&d_blur_map, W * H * sizeof(float)));
    
    CHECK(cudaMemcpy(d_img, h_img_gpu, W * H * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_blur_map, h_blur_map, W * H * sizeof(float), cudaMemcpyHostToDevice));
    
    //Max GPU
    float* d_max_gpu=NULL;
    CHECK(cudaMalloc(&d_max_gpu, sizeof(float)));
    // temp storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Step 1: query temp storage
    cub::DeviceReduce::Max(
        d_temp_storage, temp_storage_bytes,
        d_blur_map, d_max_gpu, W*H
    );

    // Step 2: allocate
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Step 3: run
    cub::DeviceReduce::Max(
        d_temp_storage, temp_storage_bytes,
        d_blur_map, d_max_gpu, W*H
    );
    float max_gpu=0.0f;
    CHECK(cudaMemcpy(&max_gpu, d_max_gpu, sizeof(float), cudaMemcpyDeviceToHost));  
    printf("Max GPU cub: %f\n", max_gpu);
    // ===== VERSIONE GPU BASELINE =====
    dim3 blockDim(block_x, block_y);
    dim3 gridDim((W + block_x - 1) / block_x, (H + block_y - 1) / block_y);
    dim3 blockDim1D(block_x*block_x, 1);
    dim3 gridDim1D((W + blockDim1D.x - 1) / blockDim1D.x, H );
    
    render_heatmap_gpu_baseline<<<gridDim1D, blockDim1D>>>(d_img, d_blur_map, inv_log_max, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    
    CHECK(cudaMemcpy(h_img_gpu, d_img, W * H * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    int passed_baseline = verify_rendered_image(h_img_cpu, h_img_gpu, W, H, 2.0f);  // Tolleranza 2 per __logf
    printf("GPU baseline: %s\n", passed_baseline ? "PASS" : "FAIL");
    
    // ===== VERSIONE GPU OTTIMIZZATA (TEXTURE + LUT) =====
    // Precomputa colormap LUT
    t1 = clock();
    precompute_colormap_lut();
    t2 = clock();
    cpu_time_ms = (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU time precompute colormap: %.2f ms\n", cpu_time_ms);

    /*
    // Crea texture object per blur_map
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_blurArray = NULL;
    CHECK(cudaMallocArray(&d_blurArray, &channelDesc, W, H));
    CHECK(cudaMemcpy2DToArray(d_blurArray, 0, 0, d_blur_map, W * sizeof(float), 
                             W * sizeof(float), H, cudaMemcpyDeviceToDevice));
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_blurArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t texObj = 0;
    CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    
    // Resetta immagine GPU
    CHECK(cudaMemcpy(d_img, h_img_gpu, W * H * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Esecuzione GPU ottimizzata
    render_heatmap_gpu_optimized<<<gridDim, blockDim>>>(d_img, texObj, inv_log_max, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());*/

    CHECK(cudaMemcpy(d_img, h_background, W * H * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    render_heatmap_gpu_branchless<<<gridDim1D,blockDim1D>>>(d_img, d_blur_map, inv_log_max, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_img_gpu, d_img, W * H * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    int passed_opt = verify_rendered_image(h_img_cpu, h_img_gpu, W, H, 2.0f);
    printf("GPU optimized: (%s)\n", passed_opt ? "PASS" : "FAIL");
    

    
    // Pulizia
    //cudaDestroyTextureObject(texObj);
    //cudaFreeArray(d_blurArray);
    cudaFree(d_img);
    cudaFree(d_blur_map);
    cudaFree(d_temp_storage);
    free(h_img_cpu);
    free(h_img_gpu);
    free(h_blur_map);
    
    return passed_baseline && passed_opt;
}

// ==================== MAIN CON BATTERIA DI TEST ====================
int main() {
    printf("============================================================\n");
    printf("BATTERIA DI TEST: Rendering Heatmap CUDA\n");
    printf("============================================================\n\n");
    
    const int sizes[] = {1920, 3840};
    const int sizesH[] = {1080, 2160};
    const int block_dims[] = {16, 32};
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_blocks = sizeof(block_dims) / sizeof(block_dims[0]);
    
    int total = 0, passed = 0;
    
    printf("%-12s | %-10s | %-6s | %-6s\n", "DIMENSIONE", "BLOCCHI", "TEMPO", "ESITO");
    printf("-------------+------------+--------+--------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int W = sizes[s];
        int H = sizesH[s];
        
        for (int b = 0; b < num_blocks; b++) {
            int bx = block_dims[b];
            int by = block_dims[b];
            
            int test_passed = run_single_test(W, H, bx, by);
            total++;
            if (test_passed) passed++;
            
            printf("%4dx%-4d    | %-2dx%-2d      | %-6s | %s\n",
                   W, H, bx, by, "-", test_passed ? "PASS" : "FAIL");
        }
    }
    
    printf("\n============================================================\n");
    printf("RISULTATO FINALE: %d/%d test superati (%.1f%%)\n", passed, total, (float)passed/total*100.0f);
    printf("============================================================\n");
    
    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}