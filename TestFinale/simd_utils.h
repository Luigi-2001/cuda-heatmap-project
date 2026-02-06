// simd_utils.h
#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

double find_max_avx_d(const double *arr, unsigned long n);

float find_max_avx_f(const float *arr, unsigned long n);
unsigned int find_max_avx_u(const unsigned int *arr, unsigned long n);

#ifdef __cplusplus
}
#endif

#endif