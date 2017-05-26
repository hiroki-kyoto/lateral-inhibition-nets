#ifndef LIB_UTIL_H
#define LIB_UTIL_H

#define PATH_MAX_LEN        1024
#define CIFAR_10_DATA_PATH  "../../lin/cifar-10-binary/data_batch_%d.bin"
#define CIFAR_10_TEST_PATH  "../../lin/cifar-10-binary/test_batch.bin"
#define CIFAR_10_IMG_WIDTH   32
#define CIFAR_10_IMG_HEIGHT  32
#define CIFAR_10_IMG_DEPTH   3
#define CIFAR_10_ROW_BYTE   3073
#define CIFAR_10_NUM_ROWS   10000
#define CIFAR_10_TOT_SETS   5

#define ASSERT(x) {\
    if(!(x)){\
        fprintf(stdout, "ERROR:@%s#%d\n", __FILE__, __LINE__);\
        exit(1);\
    }\
}\

typedef unsigned char byte;

typedef enum E_STATUS{
    SUCCESS,
    FAILURE,
}status;

typedef struct T_DATASET{
    int n;          // number of images
    int d;          // depth of image
    int w;          // width of image
    int h;          // height of image
    byte * p;       // pixels in row-wise
    byte * l;       // labels of images
}dataset;

typedef struct T_IMAGE{
    int d;          // channel number
    int w;          // width
    int h;          // height
    byte * p;       // pixels of the image
    byte l;         // label of the image
}image;

typedef struct T_MAP{
    int w;
    int h;
    float * p;
}map;

dataset * alloc_dataset(int n, int d, int h, int w){
    dataset * ds = (dataset*)malloc(sizeof(dataset));
    ASSERT(ds!=NULL);
    ds->n = n;
    ds->d = d;
    ds->h = h;
    ds->w = w;
    ds->p = (byte*)malloc(sizeof(byte)*ds->n*ds->d*ds->h*ds->w);
    ds->l = (byte*)malloc(sizeof(byte)*ds->n);
    ASSERT(ds->p!=NULL);
    ASSERT(ds->l!=NULL);
    ASSERT(ds!=NULL);
    return ds;
}

void free_dataset(dataset * ds){
    free(ds->p);
    free(ds->l);
    free(ds);
}

void get_image(image * im, dataset * ds, int id){
    ASSERT(id<ds->n);
    ASSERT(ds!=NULL);
    ASSERT(im!=NULL);
    im->p = ds->p + id * ds->d * ds->h * ds->w;
    im->l = ds->l[id];
    im->d = ds->d;
    im->h = ds->h;
    im->w = ds->w;
}


#endif
