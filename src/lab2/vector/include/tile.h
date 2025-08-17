#ifndef _TILE_H_
#define _TILE_H_

#include <immintrin.h>
#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdexcept>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

typedef struct __tile_config
{
    // 64 bytes
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16]; 
    uint8_t rows[16]; 
} __tilecfg;

// Just write a new tile config
static void init_tile_config(int M)
{
    __tilecfg config = {0};
    {
        config.palette_id = 1;
        config.start_row = 0;
    
        config.colsb[0] = 64;
        config.rows[0] = M < 16 ? M : 16;

        config.colsb[1] = 64;
        config.rows[1] = M < 16 ? M : 16;
    
        config.colsb[2] = 64;
        config.rows[2] = 16;

    }
    _tile_loadconfig(&config);
}

static void set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
    {
        throw std::runtime_error("\n Fail to do XFEATURE_XTILEDATA \n\n");
    }
}

#endif