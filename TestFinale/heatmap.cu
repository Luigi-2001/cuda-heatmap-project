#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <float.h>
#include "simd_utils.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cub/cub.cuh>

#define MAX_LINE 1024
#define MARGIN 40
#define FIELD_WIDTH 105.0f
#define FIELD_HEIGHT 68.0f
#define SOGLIA_ACC_OPT 200000

#define MAX_KERNEL_RADIUS 100


// --------------------------------------------------------
// MEMORIA COSTANTE (Cache L1 dedicata per tutti i thread)
// --------------------------------------------------------
// Deve essere dichiarata a livello globale (fuori dalle funzioni)
__constant__ float c_kernel[2 * MAX_KERNEL_RADIUS + 1];


// Error checking macro
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ---------------- Utilità temporale ----------------
float elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// ---------------- Parsing CSV ----------------
int parse_csv_line(const char *line, char *fields[], int max_fields) {
    if (!line || *line == '\0') return 0;
    int count = 0, in_quotes = 0;
    const char *start = line;
    const char *p = line;

    while (*p && count < max_fields) {
        if (*p == '"') in_quotes = !in_quotes;
        else if (*p == ',' && !in_quotes) {
            int len = p - start;
            fields[count] = (char*)malloc(len + 1);
            if (!fields[count]) {
                for (int i = 0; i < count; i++) free(fields[i]);
                return -1;
            }
            strncpy(fields[count], start, len);
            fields[count][len] = '\0';
            if (len > 0 && fields[count][0] == '"' && fields[count][len-1] == '"') {
                memmove(fields[count], fields[count] + 1, len - 2);
                fields[count][len - 2] = '\0';
            }
            count++;
            start = p + 1;
        }
        p++;
    }

    if (count < max_fields) {
        int len = p - start;
        fields[count] = (char*)malloc(len + 1);
        if (!fields[count]) {
            for (int i = 0; i < count; i++) free(fields[i]);
            return -1;
        }
        strncpy(fields[count], start, len);
        fields[count][len] = '\0';
        if (len > 0 && fields[count][0] == '"' && fields[count][len-1] == '"') {
            memmove(fields[count], fields[count] + 1, len - 2);
            fields[count][len - 2] = '\0';
        }
        count++;
    }
    return count;
}

void free_fields(char *fields[], int count) {
    for (int i = 0; i < count; i++) free(fields[i]);
}

// ---------------- Funzioni di supporto per array 1D ----------------
void draw_field_1d(unsigned char *img, int W, int H) {
    float aspect_ratio = FIELD_WIDTH / FIELD_HEIGHT;
    int field_w = W - 2 * MARGIN;
    int field_h = (int)(field_w / aspect_ratio);
    if (field_h > H - 2 * MARGIN) {
        field_h = H - 2 * MARGIN;
        field_w = (int)(field_h * aspect_ratio);
    }
    int x0 = (W - field_w) / 2;
    int y0 = (H - field_h) / 2;
    int x1 = x0 + field_w;
    int y1 = y0 + field_h;
    int cx = (x0 + x1) / 2;
    int cy = (y0 + y1) / 2;

    // Sfondo verde
    for (int i = 0; i < W * H * 3; i += 3) {
        img[i] = 20;     // R
        img[i+1] = 110;  // G
        img[i+2] = 20;   // B
    }

    unsigned char R = 245, G = 245, B = 245;

    // Bordo
    for (int x = x0; x <= x1; x++) {
        int idx_top = (y0 * W + x) * 3;
        int idx_bot = (y1 * W + x) * 3;
        img[idx_top] = img[idx_bot] = R;
        img[idx_top+1] = img[idx_bot+1] = G;
        img[idx_top+2] = img[idx_bot+2] = B;
    }
    for (int y = y0; y <= y1; y++) {
        int idx_left = (y * W + x0) * 3;
        int idx_right = (y * W + x1) * 3;
        img[idx_left] = img[idx_right] = R;
        img[idx_left+1] = img[idx_right+1] = G;
        img[idx_left+2] = img[idx_right+2] = B;
    }

    // Linea centrale
    for (int y = y0; y <= y1; y++) {
        int idx = (y * W + cx) * 3;
        img[idx] = R; img[idx+1] = G; img[idx+2] = B;
    }

    // Cerchio centrale
    int radius = (int)(field_w * 9.15 / FIELD_WIDTH);
    int thickness = 2;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int dx = x - cx, dy = y - cy;
            int d2 = dx*dx + dy*dy, r2 = radius*radius;
            if (abs(d2 - r2) <= radius * thickness) {
                int idx = (y * W + x) * 3;
                img[idx] = R; img[idx+1] = G; img[idx+2] = B;
            }
        }
    }

    // Aree di rigore
    int pa_w = (int)(field_w * 16.5f / FIELD_WIDTH);
    int pa_h = (int)(field_h * 40.3f / FIELD_HEIGHT);
    int l_pa_x = x0, r_pa_x = x1 - pa_w;
    int pa_y_top = cy - pa_h/2, pa_y_bot = cy + pa_h/2;

    for (int y = pa_y_top; y <= pa_y_bot; y++) {
        int idx_l = (y * W + l_pa_x + pa_w) * 3;
        int idx_r = (y * W + r_pa_x) * 3;
        img[idx_l] = img[idx_r] = R;
        img[idx_l+1] = img[idx_r+1] = G;
        img[idx_l+2] = img[idx_r+2] = B;
    }
    for (int x = l_pa_x; x <= l_pa_x + pa_w; x++) {
        int idx_top = (pa_y_top * W + x) * 3;
        int idx_bot = (pa_y_bot * W + x) * 3;
        img[idx_top] = img[idx_bot] = R;
        img[idx_top+1] = img[idx_bot+1] = G;
        img[idx_top+2] = img[idx_bot+2] = B;
    }
    for (int x = r_pa_x; x <= x1; x++) {
        int idx_top = (pa_y_top * W + x) * 3;
        int idx_bot = (pa_y_bot * W + x) * 3;
        img[idx_top] = img[idx_bot] = R;
        img[idx_top+1] = img[idx_bot+1] = G;
        img[idx_top+2] = img[idx_bot+2] = B;
    }

    // Aree di porta
    int ga_w = (int)(field_w * 5.5f / FIELD_WIDTH);
    int ga_h = (int)(field_h * 18.3f / FIELD_HEIGHT);
    int l_ga_x = x0, r_ga_x = x1 - ga_w;
    int ga_y_top = cy - ga_h/2, ga_y_bot = cy + ga_h/2;

    for (int y = ga_y_top; y <= ga_y_bot; y++) {
        int idx_l = (y * W + l_ga_x + ga_w) * 3;
        int idx_r = (y * W + r_ga_x) * 3;
        img[idx_l] = img[idx_r] = R;
        img[idx_l+1] = img[idx_r+1] = G;
        img[idx_l+2] = img[idx_r+2] = B;
    }
    for (int x = l_ga_x; x <= l_ga_x + ga_w; x++) {
        int idx_top = (ga_y_top * W + x) * 3;
        int idx_bot = (ga_y_bot * W + x) * 3;
        img[idx_top] = img[idx_bot] = R;
        img[idx_top+1] = img[idx_bot+1] = G;
        img[idx_top+2] = img[idx_bot+2] = B;
    }
    for (int x = r_ga_x; x <= x1; x++) {
        int idx_top = (ga_y_top * W + x) * 3;
        int idx_bot = (ga_y_bot * W + x) * 3;
        img[idx_top] = img[idx_bot] = R;
        img[idx_top+1] = img[idx_bot+1] = G;
        img[idx_top+2] = img[idx_bot+2] = B;
    }

    // Punti di rigore
    int spot_r = 1;
    int spot_lx = x0 + (int)(field_w * 11.0f / FIELD_WIDTH);
    int spot_rx = x1 - (int)(field_w * 11.0f / FIELD_WIDTH);
    for (int dy = -spot_r; dy <= spot_r; dy++) {
        for (int dx = -spot_r; dx <= spot_r; dx++) {
            if (dx*dx + dy*dy <= spot_r*spot_r) {
                int idx_l = ((cy+dy) * W + spot_lx+dx) * 3;
                int idx_r = ((cy+dy) * W + spot_rx+dx) * 3;
                img[idx_l] = img[idx_r] = R;
                img[idx_l+1] = img[idx_r+1] = G;
                img[idx_l+2] = img[idx_r+2] = B;
            }
        }
    }
}
// Rendering della heatmap su immagine 1D
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

void save_accumulation_thick_inverted(const char* filename, const unsigned int* matrix, int W, int H, int thickness) {
    unsigned char* img = (unsigned char*)malloc(W * H);
    // Inizializziamo tutto a bianco (255)
    memset(img, 255, W * H);
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (matrix[y * W + x] > 0) {
                // Se troviamo un punto, disegniamo un "quadrato" intorno ad esso
                int radius = thickness / 2;
                
                // Calcoliamo l'intensità (più punti nello stesso pixel = più scuro)
                unsigned int intensity = matrix[y * W + x] * 50;
                if (intensity > 255) intensity = 255;
                unsigned char pixel_val = (unsigned char)(255 - intensity);

                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        
                        // Controllo dei bordi
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            // Usiamo fmin per non sovrascrivere punti più scuri già disegnati
                            int idx = ny * W + nx;
                            img[idx] = (img[idx] < pixel_val) ? img[idx] : pixel_val;
                        }
                    }
                }
            }
        }
    }
    stbi_write_png(filename, W, H, 1, img, W);
    free(img);
}

void save_blur_raw_gray_thick(const char* filename, const float* matrix, int W, int H, float power) {
    unsigned char* img = (unsigned char*)malloc(W * H);
    float max_val = 0.0f;
    for (int i = 0; i < W * H; i++) if (matrix[i] > max_val) max_val = matrix[i];

    float log_max = logf(1.0f + max_val);

    for (int i = 0; i < W * H; i++) {
        if (max_val > 0.0f) {
            // 1. Normalizzazione logaritmica standard (0.0 a 1.0)
            float val = logf(1.0f + matrix[i]) / log_max;
            
            // 2. EFFETTO INSPESSIMENTO: Correzione Gamma
            // Se power < 1.0 (es. 0.5), le zone grigie diventano più scure e larghe
            // Se power > 1.0, l'effetto diventa più sottile e concentrato sui picchi
            val = powf(val, power);

            // 3. Inversione (Bianco = fondo, Nero = dati)
            img[i] = (unsigned char)(255.0f - (val * 255.0f));
        } else {
            img[i] = 255;
        }
    }
    stbi_write_png(filename, W, H, 1, img, W);
    free(img);
}
void save_img_as_png(const char* filename, unsigned char* img, int W, int H, int channels = 3) {

    // 2. Salva come PNG (1 o 3 canali)
    stbi_write_png(filename, W, H, channels, img, W * channels);

}
void save_ppm_1d(const char *filename, unsigned char *img, int W, int H) {
    FILE *out = fopen(filename, "wb");
    if (!out) {
        perror("Errore apertura file output");
        return;
    }
    fprintf(out, "P6\n%d %d\n255\n", W, H);
    fwrite(img, sizeof(unsigned char), W * H * 3, out);
    fclose(out);
}


// ---------------- Funzioni di elaborazione ----------------

// ==================== VERSIONE CPU DI RIFERIMENTO ====================
void accumulate_points_cpu(
    unsigned int *heat_map,
    const float *x_coords,
    const float *y_coords,
    long num_points,
    int W, int H
) {
    for (long i = 0; i < num_points; i++) {
        int ix = MARGIN + (int)(x_coords[i] * (W - 2 * MARGIN ));
        int iy = MARGIN + (int)(y_coords[i] * (H - 2 * MARGIN ));
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
    int ix = MARGIN + (int)(x_coords[idx] * (W - 2 * MARGIN));
    int iy = MARGIN + (int)(y_coords[idx] * (H - 2 * MARGIN));
    
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
        int ix = MARGIN + (int)(x[idx] * (W - 2 * MARGIN));
        int iy = MARGIN + (int)(y[idx] * (H - 2 * MARGIN));
        
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

// Crea kernel gaussiano
float* kernel_gauss(float sigma, int *out_r) {
    if (sigma <= 0.0f) sigma = 1e-6f;
    int r = (int)(3.0f * sigma);
    if (r < 1) r = 1;
    if (r > 100) r = 100;

    float *kernel = (float*)malloc((2 * r + 1) * sizeof(float));
    float s2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -r; i <= r; i++) {
        float w = exp(-(i * i) / s2);
        kernel[i + r] = w;
        sum += w;
    }
    for (int i = 0; i < 2 * r + 1; i++) kernel[i] /= sum;
    if (out_r) *out_r = r;
    return kernel;
}

// Blur gaussiano separabile su array 1D
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
    
    int input_start_x = blockIdx.x * blockDim.x - r;
    for (int i = tx; i < blockDim.x + 2 * r; i += blockDim.x) {
        
        // Clamping
        int src_x = input_start_x + i;
        // ✅ Più efficiente (istruzioni native GPU)
        src_x = max(0, min(W - 1, src_x));
        
        // Lettura coalesced: thread 'i' legge 'input_start + i'
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
        src_y = max(0, min(H - 1, src_y));
        
        for (int j = tx; j < blockDim.x; j += blockDim.x) {
            int src_x = base_x + j;
            // ✅ CLAMPING ORIZZONTALE per halo
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

// Confronta due immagini RGB (array 1D) e restituisce 1 se identiche (entro tolleranza), 0 altrimenti
int verify_heatmap_results(
    const unsigned char *img_cpu,
    const unsigned char *img_gpu,
    int W, int H,
    int tolerance
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

    if (different_pixels == 0) {
        printf("✅ Verifica superata: le immagini sono identiche.\n");
        return 1;
    } else {
        float percentage = (float)different_pixels / (total_pixels * 3) * 100.0;
        printf("❌ Verifica FALLITA:\n");
        printf("   Pixel diversi: %d / %d (%.3f%%)\n", different_pixels, total_pixels * 3, percentage);
        printf("   Differenza massima: %d\n", max_diff);
        printf("   Tolleranza usata: %d\n", tolerance);
        return 0;
    }
}
int verify_accumulation(
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
int verify_blur(const float *heat_cpu, const float *heat_gpu, int W, int H, float tolerance) {
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




// ---------------- Lettura dati tracking ----------------
typedef struct {
    float *x_coords;
    float *y_coords;
    long count;
    long capacity;
} PlayerData;

PlayerData* read_tracking_data(
    const char *csvfile,
    char **players,
    int player_count,
    int *col_x,
    int *col_y,
    int W, int H
) {
    FILE *fp = fopen(csvfile, "r");
    if (!fp) {
        perror("Errore apertura CSV");
        exit(1);
    }

    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, fp)) {
        fprintf(stderr, "File CSV vuoto\n");
        fclose(fp);
        exit(1);
    }
    if (strstr(line, "Home") || strstr(line, "Away")) {
        if (!fgets(line, MAX_LINE, fp)) {
            fprintf(stderr, "File CSV troppo corto\n");
            fclose(fp);
            exit(1);
        }
    }

    char *header_fields[MAX_LINE];
    int header_count = parse_csv_line(line, header_fields, MAX_LINE);
    if (header_count <= 0) {
        fprintf(stderr, "Errore parsing header\n");
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < header_count; i++) {
        char *clean = header_fields[i];
        clean[strcspn(clean, "\r\n")] = '\0';
        for (int p = 0; p < player_count; p++) {
            if (col_x[p] == -1 &&
                ((strstr(clean, players[p]) && strstr(clean, "X")) ||
                 strcmp(clean, players[p]) == 0)) {
                col_x[p] = i;
                col_y[p] = i + 1;
                break;
            }
        }
    }

    free_fields(header_fields, header_count);

    // Alloca PlayerData
    PlayerData *pdata = (PlayerData*)calloc(player_count, sizeof(PlayerData));
    for (int p = 0; p < player_count; p++) {
        pdata[p].capacity = 10000;
        pdata[p].x_coords = (float*)malloc(pdata[p].capacity * sizeof(float));
        pdata[p].y_coords = (float*)malloc(pdata[p].capacity * sizeof(float));
    }

    long total_points = 0;
    while (fgets(line, MAX_LINE, fp)) {
        total_points++;
        if (strlen(line) < 5) continue;

        char *data_fields[MAX_LINE];
        int data_count = parse_csv_line(line, data_fields, MAX_LINE);
        if (data_count <= 0) continue;

        for (int p = 0; p < player_count; p++) {
            if (col_x[p] >= data_count || col_y[p] >= data_count) continue;

            char *x_str = data_fields[col_x[p]];
            char *y_str = data_fields[col_y[p]];
            x_str[strcspn(x_str, "\r\n")] = '\0';
            y_str[strcspn(y_str, "\r\n")] = '\0';

            if (strlen(x_str) == 0 || strlen(y_str) == 0) continue;

            float x = atof(x_str);
            float y = atof(y_str);

            if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0) {
                if (pdata[p].count >= pdata[p].capacity) {
                    pdata[p].capacity *= 2;
                    pdata[p].x_coords = (float*)realloc(pdata[p].x_coords, pdata[p].capacity * sizeof(float));
                    pdata[p].y_coords = (float*)realloc(pdata[p].y_coords, pdata[p].capacity * sizeof(float));
                }
                pdata[p].x_coords[pdata[p].count] = x;
                pdata[p].y_coords[pdata[p].count] = y;
                pdata[p].count++;
            }
        }

        free_fields(data_fields, data_count);
    }

    fclose(fp);
    printf("Punti letti: %ld\n", total_points);
    return pdata;
}

// ---------------- MAIN ----------------
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s tracking.csv giocatore1 [giocatore2 ...] "
                        "[--width W] [--height H] [--sigma S]\n", argv[0]);
        return 1;
    }

    char *csvfile = argv[1];

    // ---- Estrai giocatori ----
    int player_count = 0;
    char **players = NULL;
    int i = 2;
    while (i < argc && strncmp(argv[i], "--", 2) != 0) {
        player_count++;
        players = (char**)realloc(players, player_count * sizeof(char*));
        players[player_count - 1] = argv[i];
        i++;
    }

    if (player_count == 0) {
        fprintf(stderr, "Errore: specifica almeno un giocatore.\n");
        return 1;
    }

    // ---- Parametri opzionali ----
    int W = 800, H = 520;
    float sigma = 6.0;
    int block = 16;

    while (i < argc) {
        if (strcmp(argv[i], "--width") == 0) {
            if (++i >= argc) { fprintf(stderr, "--width richiede un valore\n"); return 1; }
            W = atoi(argv[i]);
        } else if (strcmp(argv[i], "--height") == 0) {
            if (++i >= argc) { fprintf(stderr, "--height richiede un valore\n"); return 1; }
            H = atoi(argv[i]);
        } else if (strcmp(argv[i], "--sigma") == 0) {
            if (++i >= argc) { fprintf(stderr, "--sigma richiede un valore\n"); return 1; }
            sigma = atof(argv[i]);
        } else if (strcmp(argv[i], "--block") == 0) {
            if (++i >= argc) { fprintf(stderr, "--block richiede un valore\n"); return 1; }
            block = atoi(argv[i]);
        } else {
            fprintf(stderr, "Opzione sconosciuta: %s\n", argv[i]);
            return 1;
        }
        i++;
    }

    if (W <= 0 || H <= 0 || sigma <= 0) {
        fprintf(stderr, "Errore: larghezza, altezza e sigma devono essere > 0\n");
        return 1;
    }

    printf("Elaborazione di %d giocatori: ", player_count);
    for (int j = 0; j < player_count; j++) printf("%s ", players[j]);
    printf("\nDimensioni: %dx%d, sigma: %.2f\n", W, H, sigma);

    // ---- Trova colonne ----
    int *col_x = (int*)calloc(player_count, sizeof(int));
    int *col_y = (int*)calloc(player_count, sizeof(int));
    for (int p = 0; p < player_count; p++) col_x[p] = col_y[p] = -1;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    
    // ---- Leggi dati tracking ----
    PlayerData *pdata = read_tracking_data(csvfile, players, player_count, col_x, col_y, W, H);

    // ---- Concatena tutti i punti in un unico array ----
    long total_points = 0;
    for (int p = 0; p < player_count; p++) {
        total_points += pdata[p].count;
    }
    printf("Punti totali da elaborare: %ld\n", total_points);

    float *all_x = (float*)malloc(total_points * sizeof(float));
    float *all_y = (float*)malloc(total_points * sizeof(float));
    long idx = 0;
    for (int p = 0; p < player_count; p++) {
        memcpy(all_x + idx, pdata[p].x_coords, pdata[p].count * sizeof(float));
        memcpy(all_y + idx, pdata[p].y_coords, pdata[p].count * sizeof(float));
        idx += pdata[p].count;
    }

    // ---- Calcola kernel gaussiano ----
    int r;
    float *kernel = kernel_gauss(sigma, &r);
    //float *d_kernel;
    int sizeKer = 2*r+1;
    //CHECK(cudaMalloc(&d_kernel, sizeKer * sizeof(float)));
    //CHECK(cudaMemcpy(d_kernel, kernel, sizeKer * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaFree(0));  // Chiamata fittizia per inizilizzare cuda e leggere report più affidabili
    CHECK(cudaMemcpyToSymbol(c_kernel, kernel, sizeKer * sizeof(float)));

    // ---- Alloca memoria GPU ----
    float *d_x, *d_y, *d_temp, *d_blur;
    unsigned int *d_heat;
    unsigned char *d_img;
    
    CHECK(cudaMalloc(&d_x, total_points * sizeof(float)));
    CHECK(cudaMalloc(&d_y, total_points * sizeof(float)));
    CHECK(cudaMalloc(&d_heat, W * H * sizeof(unsigned int)));
    //CHECK(cudaMalloc(&d_heat_opt, W * H * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_temp, W * H * sizeof(float)));
    CHECK(cudaMalloc(&d_blur, W * H * sizeof(float)));
    CHECK(cudaMalloc(&d_img, W * H * 3 * sizeof(unsigned char)));

    
    // Copia coordinate su GPU
    CHECK(cudaMemcpy(d_x, all_x, total_points * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, all_y, total_points * sizeof(float), cudaMemcpyHostToDevice));
    

    // ---- Configurazione kernel ----
    dim3 block_acc(block*block);
    dim3 grid_acc((total_points + block_acc.x - 1) / block_acc.x);
    
    dim3 block_blur_h(block*block);
    dim3 grid_blur_h((W + block_blur_h.x - 1) / block_blur_h.x, 
                   (H + block_blur_h.y - 1) / block_blur_h.y);
    dim3 block_blur_v(block,block);
    dim3 grid_blur_v((W + block_blur_v.x - 1) / block_blur_v.x, 
                   (H + block_blur_v.y - 1) / block_blur_v.y);

    // Accumulo GPU
    CHECK(cudaMemset(d_heat, 0, W * H * sizeof(unsigned int)));
    if(total_points<SOGLIA_ACC_OPT){
        accumulate_points_gpu_baseline<<<grid_acc, block_acc>>>(d_heat, d_x, d_y, total_points, W, H);
        cudaDeviceSynchronize();
    }else{
        //Accumulo GPU ottimizzato 
        // Allocazioni temporanee
        int *d_indices_in, *d_indices_sorted;
        int *d_unique_out, *d_counts_out, *d_num_runs_out;
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        CHECK(cudaMalloc(&d_indices_in, total_points * sizeof(int)));
        CHECK(cudaMalloc(&d_indices_sorted, total_points * sizeof(int)));
        CHECK(cudaMalloc(&d_unique_out, total_points * sizeof(int)));
        CHECK(cudaMalloc(&d_counts_out, total_points * sizeof(int)));
        CHECK(cudaMalloc(&d_num_runs_out, sizeof(int)));
        // 1. Calcolo Indici
        compute_pixel_indices<<<grid_acc, block_acc>>>(d_x, d_y, d_indices_in, total_points, W, H);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // 2. Ordinamento (Radix Sort)
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_indices_in, d_indices_sorted, total_points);
        CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_indices_in, d_indices_sorted, total_points);

        // 3. Compressione (Run-Length Encode)
        // Nota: riutilizziamo d_temp_storage se abbastanza grande, altrimenti riallocare. Per sicurezza ricalcolo:
        temp_storage_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes, d_indices_sorted, d_unique_out, d_counts_out, d_num_runs_out, total_points);
        CHECK(cudaFree(d_temp_storage)); // Libero il vecchio
        CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_indices_sorted, d_unique_out, d_counts_out, d_num_runs_out, total_points);
        // 4. Scrittura Finale
        int h_num_runs;
        CHECK(cudaMemcpy(&h_num_runs, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost));

        // 5. Accumulazione
        cudaMemset(d_heat, 0, W * H * sizeof(unsigned int));
        int rle_blocks = (h_num_runs + block_acc.x - 1) / block_acc.x; // Calcolata su h_num_runs!
        render_rle_heatmap<<<rle_blocks, block_acc>>>(d_heat, d_unique_out, d_counts_out, h_num_runs);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        // Pulizia
        CHECK(cudaFree(d_indices_in));
        CHECK(cudaFree(d_indices_sorted));
        CHECK(cudaFree(d_unique_out));
        CHECK(cudaFree(d_counts_out));
        CHECK(cudaFree(d_num_runs_out));
        CHECK(cudaFree(d_temp_storage));
    }
    

    
    
    // Blur orizzontale GPU tiled
    size_t shared_h = (block_blur_h.x + 2 * r) * sizeof(float);
    gaussian_blur_horizontal_GPU_tiled<<<grid_blur_h, block_blur_h, shared_h>>>(d_heat, d_temp, W, H, r);
    CHECK(cudaGetLastError());  
    CHECK(cudaDeviceSynchronize()); 

    // Blur verticale GPU tiled
    size_t shared = (block_blur_v.y + 2 * r) * block_blur_v.x * sizeof(float);
    gaussian_blur_vertical_GPU_tiled<<<grid_blur_v, block_blur_v, shared>>>(d_temp, d_blur, W, H, r);
    CHECK(cudaGetLastError());  
    CHECK(cudaDeviceSynchronize());


    //Max GPU
    float* d_max_gpu=NULL;
    CHECK(cudaMalloc(&d_max_gpu, sizeof(float)));
    // temp storage
    void *d_temp_storage_max = NULL;
    size_t temp_storage_bytes = 0;

    // Step 1: query temp storage
    cub::DeviceReduce::Max(
        d_temp_storage_max, temp_storage_bytes,
        d_blur, d_max_gpu, W*H
    );

    // Step 2: allocate
    CHECK(cudaMalloc(&d_temp_storage_max, temp_storage_bytes));

    // Step 3: run
    cub::DeviceReduce::Max(
        d_temp_storage_max, temp_storage_bytes,
        d_blur, d_max_gpu, W*H
    );
    float max_gpu=0.0f;
    CHECK(cudaMemcpy(&max_gpu, d_max_gpu, sizeof(float), cudaMemcpyDeviceToHost));  
    printf("Max GPU cub: %f\n", max_gpu);
    CHECK(cudaFree(d_temp_storage_max));
    CHECK(cudaFree(d_max_gpu));
    // ---- Rendering GPU ----
    unsigned char *h_img_gpu = (unsigned char*)malloc(W * H * 3);
    unsigned char *h_img = (unsigned char*)malloc(W * H * 3);
    
    // Disegna campo su CPU
    draw_field_1d(h_img, W, H);
    CHECK(cudaMemcpy(d_img, h_img, W * H * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    // Rendering
    float inv_log_max = 1.0f / logf(1.0f + max_gpu);
    render_heatmap_gpu_branchless<<<grid_blur_h, block_blur_h>>>(d_img, d_blur, inv_log_max, W, H);
    cudaDeviceSynchronize();

    // Copia risultato su CPU
    CHECK(cudaMemcpy(h_img_gpu, d_img, W * H * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
   

    // ---- Risultati CPU per confronto ----
    // ---- Misurazione temporale CPU (dettagliata) ----
    struct timespec t0, t1, t2, t3, t4, t5, t6, t7, t8;
    struct timespec tA0, tB1;
    

    // Accumulo punti
    clock_gettime(CLOCK_MONOTONIC, &tA0);
    unsigned int *heat_cpu = (unsigned int*)calloc(W * H, sizeof(unsigned int));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    accumulate_points_cpu(heat_cpu, all_x, all_y, total_points, W, H);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    // Blur
    float *blur_cpu = (float*)calloc(W * H, sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &tB1);
    gaussian_blur_separable_cpu(heat_cpu, blur_cpu, W, H, kernel,r);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    // Trova massimo
    float max_val_cpu = 0.0;
    for (int i = 0; i < W * H; i++) {
        if (blur_cpu[i] > max_val_cpu) max_val_cpu = blur_cpu[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t3);
    
    // Trova massimo SIMD
    float maxSIMD = find_max_avx_f(blur_cpu, (unsigned long)(W * H));
    clock_gettime(CLOCK_MONOTONIC, &t4);
    

    // Alloca immagine
    unsigned char *img_cpu = (unsigned char*)malloc(W * H * 3);
    clock_gettime(CLOCK_MONOTONIC, &t5);

    // Disegno campo
    draw_field_1d(img_cpu, W, H);
    clock_gettime(CLOCK_MONOTONIC, &t6);

    // Rendering
    render_heatmap_cpu(img_cpu, blur_cpu, max_val_cpu, W, H);
    clock_gettime(CLOCK_MONOTONIC, &t7);

    // ---- Salvataggio risultati ----
    save_accumulation_thick_inverted("accumulation.png", heat_cpu, W, H, 3);
    save_blur_raw_gray_thick("blur.png", blur_cpu, W, H, 0.5f);
    save_img_as_png("heatmap_team_gpu.png", h_img_gpu, W, H);
    save_img_as_png("heatmap_team_cpu.png", img_cpu, W, H);
    //save_ppm_1d("heatmap_team_gpu.ppm", h_img_gpu, W, H);
    //save_ppm_1d("heatmap_team_cpu.ppm", img_cpu, W, H);
    clock_gettime(CLOCK_MONOTONIC, &t8);


    // ---- Stampa tempi CPU ----
    printf("  Tempi per team (CPU):\n");
    printf("    Allocazione heat: %.3f ms\n", elapsed_ms(tA0, t0));
    printf("    Accumulo:        %.3f ms\n", elapsed_ms(t0, t1));
    printf("    Allocazione blur: %.3f ms\n", elapsed_ms(t1, tB1));
    printf("    Blur:            %.3f ms\n", elapsed_ms(tB1, t2));
    printf("    Massimo scalare:    %f     %.3f ms\n", max_val_cpu,elapsed_ms(t2, t3));
    printf("    Massimo SIMD:      %f   %.3f ms\n", maxSIMD,elapsed_ms(t3, t4));
    printf("    Allocazione img: %.3f ms\n", elapsed_ms(t4, t5));
    printf("    Disegno campo:   %.3f ms\n", elapsed_ms(t5, t6));
    printf("    Rendering:       %.3f ms\n", elapsed_ms(t6, t7));
    printf("    Salvataggio:     %.3f ms\n", elapsed_ms(t7, t8));
    printf("    TOTALE:          %.3f ms\n", elapsed_ms(t0, t8));

    // ---- Confronti ----
    printf("\n=== CONFRONTO ACCURATEZZA RISUTATI ===\n");
    

    // Confronto rendering finale
    unsigned int *h_heat_gpu = (unsigned int*)malloc(W * H * sizeof(unsigned int));
    CHECK(cudaMemcpy(h_heat_gpu, d_heat, W * H * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    verify_accumulation(heat_cpu, h_heat_gpu,W, H, total_points);

    float *h_blur_gpu = (float*)malloc(W * H * sizeof(float));
    CHECK(cudaMemcpy(h_blur_gpu, d_blur, W * H * sizeof(float), cudaMemcpyDeviceToHost));
    verify_blur(blur_cpu,h_blur_gpu,W,H,1e-4f);

    int cpu_ok = verify_heatmap_results(img_cpu, h_img_gpu, W, H, 1);


    // ---- Pulizia ----
    free(all_x);
    free(all_y);
    free(kernel);
    free(heat_cpu);
    free(blur_cpu);
    free(img_cpu);
    free(h_img);
    free(h_img_gpu);
    free(h_blur_gpu);
    free(h_heat_gpu);



    for (int p = 0; p < player_count; p++) {
        free(pdata[p].x_coords);
        free(pdata[p].y_coords);
    }
    free(pdata);
    free(col_x);
    free(col_y);

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_heat));
    CHECK(cudaFree(d_temp));
    CHECK(cudaFree(d_blur));
    CHECK(cudaFree(d_img));

    
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    printf("\nTempo totale di esecuzione: %.3f ms\n", elapsed_ms(t_start, t_end));
    
    return 0;
}