#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib_util.h"

dataset * alloc_cifar_10_dataset(){
    dataset * ds = alloc_dataset(
        CIFAR_10_NUM_ROWS,
        CIFAR_10_IMG_DEPTH,
        CIFAR_10_IMG_HEIGHT,
        CIFAR_10_IMG_WIDTH
    );
    return ds;
}

void read_cifar_10_bacth(dataset * ds, int id){
    char path[PATH_MAX_LEN];
    FILE * f;
    byte b[CIFAR_10_ROW_BYTE];
    int i, n;
    ASSERT(id<CIFAR_10_TOT_SETS);
    ASSERT(ds!=NULL);
    ASSERT(ds->p!=NULL);
    ASSERT(ds->l!=NULL);
    sprintf(path, CIFAR_10_DATA_PATH, id);
    f = fopen(path, "rb");
    for(i=0;i<CIFAR_10_NUM_ROWS;++i){
        n = fread(b, sizeof(byte), sizeof(b), f);
        ASSERT(n==CIFAR_10_ROW_BYTE);
        ds->l[i] = b[0]; // first byte is image label
        memmove(
            ds->p+i*(CIFAR_10_ROW_BYTE-1),
            b+1,
            sizeof(byte)*(CIFAR_10_ROW_BYTE-1)
        );
    }
}
