#ifndef LIB_UTIL_H
#define LIB_UTIL_H

#define PATH_MAX_LEN        1024
#define LABEL_GROUPS        2
// cifar-10
#define CIFAR_10_TRAIN_PATH     "../../lin/cifar-10-binary/data_batch_%d.bin"
#define CIFAR_10_TEST_PATH      "../../lin/cifar-10-binary/test_batch.bin"
#define CIFAR_10_IMG_WIDTH      32
#define CIFAR_10_IMG_HEIGHT     32
#define CIFAR_10_IMG_DEPTH      3
#define CIFAR_10_ROW_BYTE       3073
#define CIFAR_10_BATCH_ROWS     10000
#define CIFAR_10_TRAIN_BATCHES  5
#define CIFAR_10_TEST_BATCHES   1
// cifar-100
#define CIFAR_100_TRAIN_PATH    "../../lin/cifar-100-binary/train.bin"
#define CIFAR_100_TEST_PATH     "../../lin/cifar-100-binary/test.bin"
#define CIFAR_100_IMG_WIDTH     32
#define CIFAR_100_IMG_HEIGHT    32
#define CIFAR_100_IMG_DEPTH     3
#define CIFAR_100_ROW_BYTE      3074
#define CIFAR_100_TRAIN_ROWS    50000
#define CIFAR_100_TEST_ROWS     10000
// debug tool
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

typedef struct T_LABEL{
    byte data[LABEL_GROUPS];
}label;

typedef struct T_DATASET{
    int n;          // number of images
    int d;          // depth of image
    int h;          // height of image
    int w;          // width of image
    byte * p;       // pixels in row-wise
    label * l;      // labels of images
}dataset;

typedef struct T_IMAGE{
    int d;          // channel number
    int h;          // height
    int w;          // width
    byte * p;       // pixels of the image
    label l;        // label of the image
}image;

typedef struct T_LAYER{
    int d;
    int h;
    int w;
    float * p;
}layer;

typedef struct T_MAP{
    int w;
    int h;
    float * p;
}map;

typedef struct T_FILTER_GROUP{
    int n;
    int h;
    int w;
    float * p;  // weight vector and its bias
}group;

typedef struct T_FILTER{
    int h;
    int w;
    float * p; // weight vector and bias
}filter;

// memory allocated alone
dataset * alloc_dataset(int n, int d, int h, int w){
    dataset * ds = (dataset*)malloc(sizeof(dataset));
    ASSERT(ds!=NULL);
    ds->n = n;
    ds->d = d;
    ds->h = h;
    ds->w = w;
    ds->p = (byte*)malloc(sizeof(byte)*ds->n*ds->d*ds->h*ds->w);
    ds->l = (label*)malloc(sizeof(label)*ds->n);
    ASSERT(ds->p!=NULL);
    ASSERT(ds->l!=NULL);
    return ds;
}
// free the memory allocated alone by dataset creator
void free_dataset(dataset * ds){
    free(ds->p);
    free(ds->l);
    free(ds);
}

image * alloc_image(){
    image * im = (image*)malloc(sizeof(image));
    ASSERT(im!=NULL);
    return im;
}

void free_image(image * im){
    free(im);
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

// layers
// memory allocated alone
layer * alloc_layer(int d, int h, int w){
    layer * lr = (layer*)malloc(sizeof(layer));
    ASSERT(lr!=NULL);
    lr->d = d;
    lr->h = h;
    lr->w = w;
    lr->p = (float*)malloc(sizeof(float)*lr->d*lr->h*lr->w);
    ASSERT(lr->p!=NULL);
    return lr;
}
// free the memory allocated alone by dataset creator
void free_layer(layer * lr){
    free(lr->p);
    free(lr);
}

map * alloc_map_alone(int h, int w){
    map * m = (map*)malloc(sizeof(map));
    ASSERT(m!=NULL);
    m->w = w;
    m->h = h;
    m->p = (float*)malloc(sizeof(float)*w*h);
    ASSERT(m->p!=NULL);
    return m;
}

map * alloc_map(){
    map * m = (map*)malloc(sizeof(map));
    ASSERT(m!=NULL);
    return m;
}

void free_map_alone(map * m){
    free(m->p);
    free(m);
}

void free_map(map * m){
    free(m);
}

// getter for map
void get_map(map * m, layer * l, int id){
    ASSERT(m!=NULL);
    ASSERT(l!=NULL);
    ASSERT(id<l->d);
    m->w = l->w;
    m->h = l->h;
    m->p = l->p + (id * l->w * l->h);
}

// groups and filters
group * alloc_group(int n, int h, int w){
    group * g = (group*)malloc(sizeof(group));
    g->n = n;
    g->h = h;
    g->w = w;
    g->p = (float*)malloc(sizeof(float)*n*(w*h+1));
    return g;
}

void group_rand(group * g, float min, float max){
    int i, n;
    n = (g->w*g->h*+1)*g->n;
    for(i=0;i<n;++i){
        g->p[i] = min + (max-min)*(rand()%RAND_MAX);
    }
}

void free_group(group * g){
    free(g->p);
    free(g);
}

filter * alloc_filter(){
    return (filter*)malloc(sizeof(filter));
}

void free_filter(filter * f){
    free(f);
}

void get_filter(filter * f, group * g, int id){
    ASSERT(f!=NULL);
    ASSERT(g!=NULL);
    ASSERT(id<g->n);
    f->w = g->w;
    f->h = g->h;
    f->p = g->p + (id*(g->w*g->h+1));
}

#endif
