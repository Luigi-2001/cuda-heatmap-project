// simd_utils.c
#include "simd_utils.h"
#include <immintrin.h>

double find_max_avx_d(const double *arr, unsigned long n) {
    if (n == 0) return -1e300;
    
    __m256d max_vec = _mm256_set1_pd(arr[0]);
    unsigned long i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d data = _mm256_loadu_pd(&arr[i]);
        max_vec = _mm256_max_pd(max_vec, data);
    }
    
    double temp[4];
    _mm256_storeu_pd(temp, max_vec);
    double max_val = temp[0];
    for (int j = 1; j < 4; j++) {
        if (temp[j] > max_val) max_val = temp[j];
    }
    
    for (; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    return max_val;
}

float find_max_avx_f(const float *arr, unsigned long n) {
    if (n == 0) return -1e30f;  // Valore molto negativo per float
    
    // Inizializza il vettore con il primo elemento
    __m256 max_vec = _mm256_set1_ps(arr[0]);
    unsigned long i = 0;
    
    // Processa blocchi di 8 float (256 bit / 32 bit = 8 elementi)
    for (; i + 8 <= n; i += 8) {
        __m256 data = _mm256_loadu_ps(&arr[i]);      // Carica 8 float non allineati
        max_vec = _mm256_max_ps(max_vec, data);      // Massimo componente per componente
    }
    
    // Estrai i valori dal vettore AVX
    float temp[8];
    _mm256_storeu_ps(temp, max_vec);
    
    // Trova il massimo tra i 8 valori
    float max_val = temp[0];
    for (int j = 1; j < 8; j++) {
        if (temp[j] > max_val) max_val = temp[j];
    }
    
    // Gestisci gli elementi rimanenti (meno di 8)
    for (; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    return max_val;
}
unsigned int find_max_avx_u(const unsigned int *arr, unsigned long n) {
    if (n == 0) return 0;  // Minimo per unsigned int
    
    // Inizializza con il valore minimo (0) invece del primo elemento
    // → evita problemi se arr[0] non è allineato o n < 8
    __m256i max_vec = _mm256_setzero_si256();
    unsigned long i = 0;
    
    // Processa blocchi di 8 unsigned int (256 bit / 32 bit = 8 elementi)
    for (; i + 8 <= n; i += 8) {
        __m256i data = _mm256_loadu_si256((__m256i const*)&arr[i]);  // Carica 8 uint32 non allineati
        max_vec = _mm256_max_epu32(max_vec, data);                   // Massimo unsigned int componente per componente
    }
    
    // Estrai i valori dal vettore AVX2
    unsigned int temp[8];
    _mm256_storeu_si256((__m256i*)temp, max_vec);
    
    // Trova il massimo tra gli 8 valori
    unsigned int max_val = temp[0];
    for (int j = 1; j < 8; j++) {
        if (temp[j] > max_val) max_val = temp[j];
    }
    
    // Gestisci gli elementi rimanenti (meno di 8)
    for (; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    return max_val;
}