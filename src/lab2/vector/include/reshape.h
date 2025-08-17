#ifndef _AVX_UTILS_H
#define _AVX_UTILS_H

#include <immintrin.h>
#include <cstdint>

void reshape(uint8_t* matrix, int rows, int colsb, uint8_t* matrix_o);

void reshape_block(uint8_t* matrix, int rows, int colsb, int cur_row, int start_col, uint8_t* matrix_o);

#endif