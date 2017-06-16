// lib_read_dataset.h
#ifndef LIB_READ_DATASET
#define LIB_READ_DATASET

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib_util.h"

dataset * alloc_cifar_10_train_dataset(){
    dataset * ds = alloc_dataset(
        CIFAR_10_BATCH_ROWS*CIFAR_10_TRAIN_BATCHES,
        CIFAR_10_IMG_DEPTH,
        CIFAR_10_IMG_HEIGHT,
        CIFAR_10_IMG_WIDTH
    );
    return ds;
}

dataset * alloc_cifar_10_test_dataset(){
    dataset * ds = alloc_dataset(
        CIFAR_10_BATCH_ROWS*CIFAR_10_TEST_BATCHES,
        CIFAR_10_IMG_DEPTH,
        CIFAR_10_IMG_HEIGHT,
        CIFAR_10_IMG_WIDTH
    );
    return ds;
}

dataset * alloc_cifar_100_train_dataset(){
    dataset * ds = alloc_dataset(
        CIFAR_100_TRAIN_ROWS,
        CIFAR_100_IMG_DEPTH,
        CIFAR_100_IMG_HEIGHT,
        CIFAR_100_IMG_WIDTH
    );
    return ds;
}

dataset * alloc_cifar_100_test_dataset(){
    dataset * ds = alloc_dataset(
        CIFAR_100_TEST_ROWS,
        CIFAR_100_IMG_DEPTH,
        CIFAR_100_IMG_HEIGHT,
        CIFAR_100_IMG_WIDTH
    );
    return ds;
}

dataset * get_cifar_10_train(){
    dataset * ds;
    char path[PATH_MAX_LEN];
    FILE * f;
    byte b[CIFAR_10_ROW_BYTE];
    int i, k, m, n;
    ds = alloc_cifar_10_train_dataset();
    ASSERT(ds!=NULL);
    ASSERT(ds->p!=NULL);
    ASSERT(ds->l!=NULL);
    for(k=0;k<CIFAR_10_TRAIN_BATCHES;++k){
        sprintf(path, CIFAR_10_TRAIN_PATH, k+1);
        f = fopen(path, "rb");
        i=k*CIFAR_10_BATCH_ROWS;
        m = i+CIFAR_10_BATCH_ROWS;
        for(;i<m;++i){
            n = fread(b, sizeof(byte), sizeof(b), f);
            ASSERT(n==CIFAR_10_ROW_BYTE);
            ds->l[i].data[0] = b[0]; // first byte is image label
            memmove(
                ds->p+i*(CIFAR_10_ROW_BYTE-1),
                b+1,
                sizeof(byte)*(CIFAR_10_ROW_BYTE-1)
            );
        }
        fclose(f);
    }
    return ds;
}

dataset * get_cifar_10_test(){
    dataset * ds;
    char path[PATH_MAX_LEN];
    FILE * f;
    byte b[CIFAR_10_ROW_BYTE];
    int i, k, m, n;
    ds = alloc_cifar_10_test_dataset();
    ASSERT(ds!=NULL);
    ASSERT(ds->p!=NULL);
    ASSERT(ds->l!=NULL);
    sprintf(path, CIFAR_10_TEST_PATH);
    f = fopen(path, "rb");
    i=0;
    m = i+CIFAR_10_BATCH_ROWS;
    for(;i<m;++i){
        n = fread(b, sizeof(byte), sizeof(b), f);
        ASSERT(n==CIFAR_10_ROW_BYTE);
        ds->l[i].data[0] = b[0]; // first byte is image label
        memmove(
            ds->p+i*(CIFAR_10_ROW_BYTE-1),
            b+1,
            sizeof(byte)*(CIFAR_10_ROW_BYTE-1)
        );
    }
    fclose(f);
    return ds;
}

// CIFAR-100 dataset
// label_class = 0 for big class(20)
// label_class = 1 for small class(100)
dataset * get_cifar_100_train(){
    dataset * ds;
    char path[PATH_MAX_LEN];
    FILE * f;
    byte b[CIFAR_100_ROW_BYTE];
    int i, m, n;
    ds = alloc_cifar_100_train_dataset();
    sprintf(path, CIFAR_100_TRAIN_PATH);
    f = fopen(path, "rb");
    i = 0;
    m = i + CIFAR_100_TRAIN_ROWS;
    for(;i<m;++i){
        n = fread(b, sizeof(byte), sizeof(b), f);
        ASSERT(n==CIFAR_100_ROW_BYTE);
        ds->l[i].data[0] = b[0]; // first byte is coarse label
        ds->l[i].data[1] = b[1]; // second byte is fine label
        memmove(
            ds->p+i*(CIFAR_100_ROW_BYTE-2),
            b+2,
            sizeof(byte)*(CIFAR_100_ROW_BYTE-2)
        );
    }
    fclose(f);
    return ds;
}

dataset * get_cifar_100_test(){
    dataset * ds;
    char path[PATH_MAX_LEN];
    FILE * f;
    byte b[CIFAR_100_ROW_BYTE];
    int i, m, n;
    ds = alloc_cifar_100_test_dataset();
    sprintf(path, CIFAR_100_TEST_PATH);
    f = fopen(path, "rb");
    i = 0;
    m = i + CIFAR_100_TEST_ROWS;
    for(;i<m;++i){
        n = fread(b, sizeof(byte), sizeof(b), f);
        ASSERT(n==CIFAR_100_ROW_BYTE);
        ds->l[i].data[0] = b[0]; // first byte is coarse label
        ds->l[i].data[1] = b[1]; // second byte is fine label
        memmove(
            ds->p+i*(CIFAR_100_ROW_BYTE-2),
            b+2,
            sizeof(byte)*(CIFAR_100_ROW_BYTE-2)
        );
    }
    fclose(f);
    return ds;
}

#endif
