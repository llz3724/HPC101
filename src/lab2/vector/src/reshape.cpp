#include "reshape.h"
#include <immintrin.h>
#include <iostream>
#include <assert.h>

void reshape(uint8_t* matrix, int rows, int colsb, uint8_t* matrix_o) {
    assert(rows % 16 == 0 && colsb % 64 == 0);
    uint8_t* matrix_p;
    for (int i = 0; i < rows; i += 16) {
        matrix_p = matrix + i * colsb;
        for (int j = 0; j < colsb; j += 64) {
            reshape_block(matrix_p, rows, colsb, i, j, matrix_o);
        }
    }
}

inline void reshape_block(uint8_t* matrix, int rows, int colsb, int cur_row, int start_col, uint8_t* matrix_o) {
    int trans_rows = start_col / 4;
    int new_col = 4 * rows;
    int new_start_col = cur_row * 4;    
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    
    int mask;
    alignas(64) static const int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static const int64_t idx2[8] = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static const int32_t idx3[16] = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    const __m512i vidx1 = _mm512_load_epi64(idx1);
    const __m512i vidx2 = _mm512_load_epi64(idx2);
    const __m512i vidx3 = _mm512_load_epi32(idx3);
    
    
    t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 0 * colsb)), _mm256_loadu_epi64(matrix + start_col + 8 * colsb), 1);
    t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 0 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 8 * colsb + 32), 1);

    t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 1 * colsb)), _mm256_loadu_epi64(matrix + start_col + 9 * colsb), 1);
    t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 1 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 9 * colsb + 32), 1);

    t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 2 * colsb)), _mm256_loadu_epi64(matrix + start_col + 10 * colsb), 1);
    ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 2 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 10 * colsb + 32), 1);

    t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 3 * colsb)), _mm256_loadu_epi64(matrix + start_col + 11 * colsb), 1);
    tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 3 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 11 * colsb + 32), 1);

    t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 4 * colsb)), _mm256_loadu_epi64(matrix + start_col + 12 * colsb), 1);
    tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 4 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 12 * colsb + 32), 1);

    t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 5 * colsb)), _mm256_loadu_epi64(matrix + start_col + 13 * colsb), 1);
    td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 5 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 13 * colsb + 32), 1);

    t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 6 * colsb)), _mm256_loadu_epi64(matrix + start_col + 14 * colsb), 1);
    te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 6 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 14 * colsb + 32), 1);

    t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 7 * colsb)), _mm256_loadu_epi64(matrix + start_col + 15 * colsb), 1);
    tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_epi64(matrix + start_col + 7 * colsb + 32)), _mm256_loadu_epi64(matrix + start_col + 15 * colsb + 32), 1);

    
    mask= 0xcc;
    r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
    r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
    r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
    r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
    r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
    r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
    ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
    rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);
    
    mask= 0x33;
    r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
    r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
    r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
    r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
    rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
    rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
    re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
    rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);
    
    mask = 0xaa;
    t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
    t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
    t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
    t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
    t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
    t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
    tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
    td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);
    
    mask = 0x55;
    t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
    t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
    t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
    t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
    ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
    tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
    te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
    tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);
    
    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  0) * new_col + new_start_col, r0);
    r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  2) * new_col + new_start_col, r2);
    r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  4) * new_col + new_start_col, r4);
    r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  6) * new_col + new_start_col, r6);
    r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  8) * new_col + new_start_col, r8);
    ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
    _mm512_storeu_epi64(matrix_o + (trans_rows + 10) * new_col + new_start_col, ra);
    rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
    _mm512_storeu_epi64(matrix_o + (trans_rows + 12) * new_col + new_start_col, rc);
    re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    
    _mm512_storeu_epi64(matrix_o + (trans_rows + 14) * new_col + new_start_col, re);
    
    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  1) * new_col + new_start_col, r1);
    r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  3) * new_col + new_start_col, r3);
    r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  5) * new_col + new_start_col, r5);
    r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
    _mm512_storeu_epi64(matrix_o + (trans_rows +  7) * new_col + new_start_col, r7);
    r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
    _mm512_storeu_epi64(matrix_o + (trans_rows +  9) * new_col + new_start_col, r9);
    rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
    _mm512_storeu_epi64(matrix_o + (trans_rows + 11) * new_col + new_start_col, rb);
    rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
    _mm512_storeu_epi64(matrix_o + (trans_rows + 13) * new_col + new_start_col, rd);
    rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);
    _mm512_storeu_epi64(matrix_o + (trans_rows + 15) * new_col + new_start_col, rf);
}



// void reshape_block(uint8_t* matrix, int rows, int colsb, int cur_row, int start_col, uint8_t* matrix_o) {
//     int trans_rows = start_col / 4;
//     int new_col = 4 * rows;
//     int new_start_col = cur_row * 4;
//     __mmask64 mask = 0xFFFFFFFFFFFFFFFFULL;
//     __m512i r0 = _mm512_loadu_epi8(matrix + start_col + 0 * colsb);
//     __m512i r1 = cur_row + 1 < rows ? _mm512_loadu_epi8(matrix + start_col + 1 * colsb) : _mm512_setzero_si512();
//     __m512i r2 = cur_row + 2 < rows ? _mm512_loadu_epi8(matrix + start_col + 2 * colsb) : _mm512_setzero_si512();
//     __m512i r3 = cur_row + 3 < rows ? _mm512_loadu_epi8(matrix + start_col + 3 * colsb) : _mm512_setzero_si512();
//     __m512i r4 = cur_row + 4 < rows ? _mm512_loadu_epi8(matrix + start_col + 4 * colsb) : _mm512_setzero_si512();
//     __m512i r5 = cur_row + 5 < rows ? _mm512_loadu_epi8(matrix + start_col + 5 * colsb) : _mm512_setzero_si512();
//     __m512i r6 = cur_row + 6 < rows ? _mm512_loadu_epi8(matrix + start_col + 6 * colsb) : _mm512_setzero_si512();
//     __m512i r7 = cur_row + 7 < rows ? _mm512_loadu_epi8(matrix + start_col + 7 * colsb) : _mm512_setzero_si512();
//     __m512i r8 = cur_row + 8 < rows ? _mm512_loadu_epi8(matrix + start_col + 8 * colsb) : _mm512_setzero_si512();
//     __m512i r9 = cur_row + 9 < rows ? _mm512_loadu_epi8(matrix + start_col + 9 * colsb) : _mm512_setzero_si512();
//     __m512i ra = cur_row + 10 < rows ? _mm512_loadu_epi8(matrix + start_col + 10 * colsb) : _mm512_setzero_si512();
//     __m512i rb = cur_row + 11 < rows ? _mm512_loadu_epi8(matrix + start_col + 11 * colsb) : _mm512_setzero_si512();
//     __m512i rc = cur_row + 12 < rows ? _mm512_loadu_epi8(matrix + start_col + 12 * colsb) : _mm512_setzero_si512();
//     __m512i rd = cur_row + 13 < rows ? _mm512_loadu_epi8(matrix + start_col + 13 * colsb) : _mm512_setzero_si512();
//     __m512i re = cur_row + 14 < rows ? _mm512_loadu_epi8(matrix + start_col + 14 * colsb) : _mm512_setzero_si512();
//     __m512i rf = cur_row + 15 < rows ? _mm512_loadu_epi8(matrix + start_col + 15 * colsb) : _mm512_setzero_si512();

//     __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

//     t0 = _mm512_unpacklo_epi32(r0, r1);
//     t1 = _mm512_unpackhi_epi32(r0, r1);
//     t2 = _mm512_unpacklo_epi32(r2, r3);
//     t3 = _mm512_unpackhi_epi32(r2, r3);
//     t4 = _mm512_unpacklo_epi32(r4, r5);
//     t5 = _mm512_unpackhi_epi32(r4, r5);
//     t6 = _mm512_unpacklo_epi32(r6, r7);
//     t7 = _mm512_unpackhi_epi32(r6, r7);
//     t8 = _mm512_unpacklo_epi32(r8, r9);
//     t9 = _mm512_unpackhi_epi32(r8, r9);
//     ta = _mm512_unpacklo_epi32(ra, rb);
//     tb = _mm512_unpackhi_epi32(ra, rb);
//     tc = _mm512_unpacklo_epi32(rc, rd);
//     td = _mm512_unpackhi_epi32(rc, rd);
//     te = _mm512_unpacklo_epi32(re, rf);
//     tf = _mm512_unpackhi_epi32(re, rf);

//     r0 = _mm512_unpacklo_epi64(t0, t2);
//     r1 = _mm512_unpackhi_epi64(t0, t2);
//     r2 = _mm512_unpacklo_epi64(t1, t3);
//     r3 = _mm512_unpackhi_epi64(t1, t3);
//     r4 = _mm512_unpacklo_epi64(t4, t6);
//     r5 = _mm512_unpackhi_epi64(t4, t6);
//     r6 = _mm512_unpacklo_epi64(t5, t7);
//     r7 = _mm512_unpackhi_epi64(t5, t7);
//     r8 = _mm512_unpacklo_epi64(t8, ta);
//     r9 = _mm512_unpackhi_epi64(t8, ta);
//     ra = _mm512_unpacklo_epi64(t9, tb);
//     rb = _mm512_unpackhi_epi64(t9, tb);
//     rc = _mm512_unpacklo_epi64(tc, te);
//     rd = _mm512_unpackhi_epi64(tc, te);
//     re = _mm512_unpacklo_epi64(td, tf);
//     rf = _mm512_unpackhi_epi64(td, tf);

//     t0 = _mm512_shuffle_i32x4(r0, r4, 0x88);
//     t1 = _mm512_shuffle_i32x4(r1, r5, 0x88);
//     t2 = _mm512_shuffle_i32x4(r2, r6, 0x88);
//     t3 = _mm512_shuffle_i32x4(r3, r7, 0x88);
//     t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd);
//     t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd);
//     t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd);
//     t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd);
//     t8 = _mm512_shuffle_i32x4(r8, rc, 0x88);
//     t9 = _mm512_shuffle_i32x4(r9, rd, 0x88);
//     ta = _mm512_shuffle_i32x4(ra, re, 0x88);
//     tb = _mm512_shuffle_i32x4(rb, rf, 0x88);
//     tc = _mm512_shuffle_i32x4(r8, rc, 0xdd);
//     td = _mm512_shuffle_i32x4(r9, rd, 0xdd);
//     te = _mm512_shuffle_i32x4(ra, re, 0xdd);
//     tf = _mm512_shuffle_i32x4(rb, rf, 0xdd);

//     r0 = _mm512_shuffle_i32x4(t0, t8, 0x88);
//     r1 = _mm512_shuffle_i32x4(t1, t9, 0x88);
//     r2 = _mm512_shuffle_i32x4(t2, ta, 0x88);
//     r3 = _mm512_shuffle_i32x4(t3, tb, 0x88);
//     r4 = _mm512_shuffle_i32x4(t4, tc, 0x88);
//     r5 = _mm512_shuffle_i32x4(t5, td, 0x88);
//     r6 = _mm512_shuffle_i32x4(t6, te, 0x88);
//     r7 = _mm512_shuffle_i32x4(t7, tf, 0x88);
//     r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd);
//     r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd);
//     ra = _mm512_shuffle_i32x4(t2, ta, 0xdd);
//     rb = _mm512_shuffle_i32x4(t3, tb, 0xdd);
//     rc = _mm512_shuffle_i32x4(t4, tc, 0xdd);
//     rd = _mm512_shuffle_i32x4(t5, td, 0xdd);
//     re = _mm512_shuffle_i32x4(t6, te, 0xdd);
//     rf = _mm512_shuffle_i32x4(t7, tf, 0xdd);

//     _mm512_storeu_epi64(matrix_o + (trans_rows) * new_col + new_start_col, r0);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 1) * new_col + new_start_col, r1);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 2) * new_col + new_start_col, r2);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 3) * new_col + new_start_col, r3);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 4) * new_col + new_start_col, r4);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 5) * new_col + new_start_col, r5);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 6) * new_col + new_start_col, r6);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 7) * new_col + new_start_col, r7);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 8) * new_col + new_start_col, r8);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 9) * new_col + new_start_col, r9);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 10) * new_col + new_start_col, ra);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 11) * new_col + new_start_col, rb);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 12) * new_col + new_start_col, rc);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 13) * new_col + new_start_col, rd);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 14) * new_col + new_start_col, re);
//     _mm512_storeu_epi64(matrix_o + (trans_rows + 15) * new_col + new_start_col, rf);
// }
