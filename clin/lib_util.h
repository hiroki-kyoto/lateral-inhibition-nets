#ifndef LIB_UTIL_H
#define LIB_UTIL_H

#define PATH_MAX_LEN        1024
#define LABEL_GROUPS        2
#define LABEL_CATEGORY      256
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
}

#define DOT(x){\
    fprintf(stdout, "MSG@%s#%d:%s\n", __FILE__, __LINE__, x);\
}

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

typedef struct T_LAYER_GROUP{
    int n;
    layer ** l;
}layer_group;

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

// stacked filters
typedef struct T_STACKED_FILTER{
    int d; // depth of stack of filters
    filter ** f; // filters in stack
}stacked_filter;

typedef struct T_STACKED_FILTER_GROUP{
    int n;
    stacked_filter ** sf;
}stacked_filter_group;

typedef struct T_CHANNEL_MERGER_GROUP{
    int n;  // number of mergers
    int d;  // number of channels to merge
    float * p;
}merger_group;

typedef struct T_CHANNEL_MEGER{
    int d;  // number channels to merge(depth)
    float * p;
}merger;

typedef struct T_LATERAL_INHIBITION{
    int r; // radius for lateral inhibition
    filter * f; // filter specifed for lateral inhibition
}lainer;

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

// layer group methods
layer_group * alloc_layer_group(int n, int d, int h, int w){
    int i;
    layer_group * lg = (layer_group*)malloc(sizeof(layer_group));
    lg->n = n;
    lg->l = (layer**)malloc(sizeof(layer*)*n);
    for(i=0; i<n; ++i){
        lg->l[i] = alloc_layer(d, h, w);
    }
}

void free_layer_group(layer_group * lg){
    int i;
    for(i=0; i<lg->n; ++i){
        free_layer(lg->l[i]);
    }
    free(lg->l);
    free(lg);
}

layer * get_layer(layer_group * lg, int id){
    ASSERT(id<lg->n);
    return lg->l[id];
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

group * alloc_group_with_filter(int n, filter * f){
    group * g;
    int i;
    g = (group*)malloc(sizeof(group));
    g->n = n;
    g->h = f->h;
    g->w = f->w;
    g->p = (float*)malloc(sizeof(float)*n*(g->w*g->h+1));
    for(i=0; i<n; ++i){
        memmove(
            (char*)(g->p+(i*(g->w*g->h+1))),
            (char*)(f->p),
            sizeof(float)*(g->w*g->h+1)
        );
    }
    return g;
}

void group_rand(group * g, float min, float max){
    int i, n;
    n = (g->w*g->h+1)*g->n;
    for(i=0;i<n;++i){
        g->p[i] = min + (max-min)*rand()/RAND_MAX;
    }
}

void free_group(group * g){
    free(g->p);
    free(g);
}

filter * alloc_filter(){
    return (filter*)malloc(sizeof(filter));
}

filter * alloc_filter_alone(int h, int w){
    filter * f = (filter*)malloc(sizeof(filter));
    f->h = h;
    f->w = w;
    f->p = (float*)malloc(sizeof(float)*(h*w+1));
    return f;
}

void free_filter(filter * f){
    free(f);
}

void free_filter_alone(filter * f){
    free(f->p);
    free(f);
}

void get_filter(filter * f, group * g, int id){
    ASSERT(id<g->n);
    f->w = g->w;
    f->h = g->h;
    f->p = g->p + (id*(g->w*g->h+1));
}

// methods provided for allocating releasing channel mergers
merger * alloc_merger(){
    return (merger*)malloc(sizeof(merger));
}

merger_group * alloc_merger_group(int n, int d){
    merger_group * mg =
    (merger_group*)malloc(sizeof(merger_group));
    mg->n = n;
    mg->d = d;
    mg->p = (float*)malloc(sizeof(float)*n*(d+1));
}

void merger_group_rand(merger_group * mg, float min, float max){
    int i, n;
    n = mg->n * (mg->d+1);
    for(i=0; i<n; ++i){
        mg->p[i] = min + (max-min) * rand()/RAND_MAX;
    }
}

void get_merger(merger * m, merger_group * mg, int id){
    ASSERT(id<mg->n);
    m->d = mg->d;
    m->p = mg->p + (id*(mg->d+1));
}

void free_merger(merger * m){
  free(m);
}

void free_merger_group(merger_group *mg){
    free(mg->p);
    free(mg);
}

void make_lain_filter(filter * f, int r){
    int i, k;
    float s;
    ASSERT(f!=NULL);
    ASSERT(f->p!=NULL);
    ASSERT(r>0);
    ASSERT(f->w==2*r+1);
    ASSERT(f->h==2*r+1);
    s = 0.0;
    f->p[r*f->w+r] = 1.0; // center weight
    f->p[f->h*f->w] = 0.0; // bias set to be 0
    for(i=-r; i<=r; ++i){
        for(k=-r; k<=r; ++k){
            if(i!=0 || k!=0){
                s += 1.0/(i*i+k*k);
            }
        }
    }
    for(i=-r; i<=r; ++i){
        for(k=-r; k<=r; ++k){
            if(i!=0 || k!=0){
                f->p[(i+r)*f->w+(k+r)] = -1.0/(i*i+k*k)/s;
            }
        }
    }
}

// lateral inhibition layer
lainer * alloc_lainer(int r){
    lainer * la;
    ASSERT(r>0);
    la = (lainer*)malloc(sizeof(lainer));
    la->r = r;
    la->f = alloc_filter_alone(2*r+1, 2*r+1);
    make_lain_filter(la->f, r);
    return la;
}

void free_lainer(lainer * la){
    free_filter(la->f);
    free(la);
}

// extend layer to specifed size with zeros
void extend_layer_with_zeros(
    layer *nl,
    layer * l,
    int r
){
    int i, j, k;
    ASSERT(r>0);
    ASSERT(nl->w==l->w+2*r);
    ASSERT(nl->h==l->h+2*r);
    ASSERT(nl->d==l->d);
    // fill zeros into surrounded region
    for(i=0; i<nl->h; ++i){
        for(j=0; j<nl->w; ++j){
            // this part is easy to be paralleled in GPUs
            if(i<r || i>l->h-1+r || j<r || j>l->w-1+r){
                for(k=0; k<nl->d; ++k){
                    nl->p[k*nl->w*nl->h + i*nl->w+j] = 0;
                }
            } else {
                for(k=0; k<nl->d; ++k){
                    nl->p[k*nl->w*nl->h + i*nl->w+j] =
                    l->p[k*l->w*l->h + (i-r)*l->w + j-r];
                }
            }
        }
    }
}

layer * alloc_next_layer_with_conv(layer * l, group * g){
    return alloc_layer(l->d*g->n, l->h-g->h+1, l->w-g->w+1);
}

layer * alloc_same_size_layer(layer * l){
    return alloc_layer(l->d, l->h, l->w);
}

layer * alloc_extended_layer_with_lainer(
    layer * l,
    lainer * la
){
    layer * el;
    el = alloc_layer(l->d, l->h+2*la->r, l->w+2*la->r);
    return el;
}

layer * alloc_input_layer(dataset * ds){
    return alloc_layer(ds->d, ds->h, ds->w);
}

layer * alloc_next_layer_with_merger_group(
    layer * l,
    merger_group * mg
){
    return alloc_layer(mg->n, l->h, l->w);
}

layer * alloc_next_layer_with_stacked_filter_group(
    layer * l,
    stacked_filter_group * sfg
){
    // have to allocate a few intermidiate layers
    //return alloc_layer(sfg->n, );
}

void load_input(layer * il, image * im){
    int i, j, k;
    ASSERT(il->d==im->d);
    ASSERT(il->h==im->h);
    ASSERT(il->w==im->w);
    for(i=0; i<im->d; ++i){
        for(j=0; j<im->h; ++j){
            for(k=0; k<im->w; ++k){
                il->p[i*il->h*il->w+j*il->w+k] =
                im->p[i*il->h*il->w+j*il->w+k];
            }
        }
    }
}

stacked_filter * alloc_stacked_filter(int d){
    stacked_filter * sf;
    int i;
    sf = (stacked_filter*)malloc(sizeof(stacked_filter));
    sf->d = d;
    sf->f = (filter**)malloc(sizeof(filter*)*d);
    for(i=0; i<sf->d; ++i){
        sf->f[i] = NULL;
    }
    return sf;
}

void sf_set_filter(stacked_filter * sf, int id, int h, int w){
    sf->f[id] = alloc_filter_alone(h, w);
}

// memory allocation only apply for once!!!
// d : depth of stack of filters
// w : width array
// h : height array
stacked_filter * alloc_sf_once(int d, int * h, int * w){
    int i;
    stacked_filter * sf = alloc_stacked_filter(d);
    for(i=0; i<sf->d; ++i){
        sf_set_filter(sf, i, h[i], w[i]);
    }
    return sf;
}

void free_stacked_filter(stacked_filter * sf){
    int i;
    for(i=0; i<sf->d; ++i){
        free_filter(sf->f[i]);
    }
    free(sf->f);
    free(sf);
}

void filter_rand(filter * f, float min, float max){
    int i, n;
    n = f->w*f->h+1;
    for(i=0;i<n;++i){
        f->p[i] = min + (max-min)*rand()/RAND_MAX;
    }
}

void sf_rand(stacked_filter * sf, float min, float max){
    int k;
    for(k=0; k<sf->d; ++k){
        filter_rand(sf->f[k], min, max);
    }
}

filter * sf_get_filter(stacked_filter * sf, int id){
    ASSERT(id<sf->d);
    return sf->f[id];
}

stacked_filter_group * alloc_stacked_filter_group(
    int n,
    int d,
    int * h,
    int * w
){
    stacked_filter_group * sfg;
    int i;
    sfg = (stacked_filter_group*)malloc(
        sizeof(stacked_filter_group)
    );
    sfg->n = n;
    sfg->sf = (stacked_filter**)malloc(sizeof(stacked_filter*)*n);
    for(i=0; i<sfg->n; ++i){
        sfg->sf[i] = alloc_sf_once(d, h, w);
    }
    return sfg;
}

void free_stacked_filter_group(stacked_filter_group * sfg){
    int i;
    for(i=0; i<sfg->n; ++i){
        free_stacked_filter(sfg->sf[i]);
    }
    free(sfg->sf);
    free(sfg);
}

void sfg_rand(stacked_filter_group * sfg, float min, float max){
    int i;
    for(i=0; i<sfg->n; ++i){
        sf_rand(sfg->sf[i], min, max);
    }
}




// python interation
// filter weight retrive
// get weight element at (x,y)
float f_w_r(filter * f, int x, int y){
    ASSERT(x<f->w);
    ASSERT(y<f->h);
    return f->p[y*f->w + x];
}
// filter weight set
void f_w_s(filter * f, float w, int x, int y){
    ASSERT(x<f->w);
    ASSERT(y<f->h);
    f->p[y*f->w + x] = w;
}
// filter bias retrive
float f_b_r(filter * f){ // get bias for filter
    return f->p[f->h*f->w];
}
// filter bias set
void f_b_s(filter *f, float b){
    f->p[f->h*f->w] = b;
}
// map element retrive
// get element at (x,y)
float m_e_r(map * m, int x, int y){
    ASSERT(x<m->w);
    ASSERT(y<m->h);
    return m->p[y*m->w + x];
}
float m_e_s(map * m, float e, int x, int y){
    ASSERT(x<m->w);
    ASSERT(y<m->h);
    m->p[y*m->w + x] = e;
}

void print_filter(filter * f){
    int i, k;
    fprintf(stdout, "[h,w]=[%d,%d]\n", f->h, f->w );
    for(i=0; i<f->h; ++i){
        for(k=0; k<f->w; ++k){
            fprintf(stdout, "%9.3f\t", f_w_r(f, k, i));
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "%9.3f\t\n", f_b_r(f));
}

void print_group(group * g){
    int i;
    filter * f;
    f = alloc_filter();
    fprintf(stdout,
        "[n,h,w]=[%d,%d,%d]\n",
        g->n, g->h, g->w);
    for(i=0; i<g->n; ++i){
        get_filter(f, g, i);
        print_filter(f);
    }
    free_filter(f);
}

void print_map(map * m){
    int i, k;
    fprintf(stdout, "[h,w]=[%d,%d]\n", m->h, m->w );
    for(i=0; i<m->h; ++i){
        for(k=0; k<m->w; ++k){
            fprintf(stdout, "%9.3f\t", m_e_r(m, k, i));
        }
        fprintf(stdout, "\n");
    }
}

void print_layer(layer * l){
    int i;
    map * m;
    fprintf(stdout, "[d,h,w]=[%d,%d,%d]\n", l->d, l->h, l->w);
    m = alloc_map();
    for(i=0; i<l->d; ++i){
        get_map(m, l, i);
        print_map(m);
    }
    free_map(m);
}

void print_lainer(lainer * la){
    fprintf(stdout, "[r]=%d\n", la->r);
    print_filter(la->f);
    fprintf(stdout, "\n");
}

void print_sf(stacked_filter * sf){
    int i;
    fprintf(stdout, "[d]=[%d]\n", sf->d);
    for(i=0; i<sf->d; ++i){
        print_filter(sf->f[i]);
    }
    fprintf(stdout, "\n");
}

void print_sfg(stacked_filter_group * sfg){
    int i;
    fprintf(stdout, "[n,d]=[%d,%d]\n", sfg->n, sfg->d);
    for(i=0; i<sfg->n; ++i){
        print_sf(sfg->sf[i]);
    }
    fprintf(stdout, "\n");
}

void print_merger(merger * m){
    int i;
    fprintf(stdout, "[d]=[%d]\n", m->d);
    for(i=0; i<m->d+1; ++i){
        fprintf(stdout, "%9.3f\t", m->p[i]);
    }
    fprintf(stdout, "\n");
}

void print_merger_group(merger_group * mg){
    int i;
    merger * m = alloc_merger();
    fprintf(stdout, "[n,d]=[%d,%d]\n", mg->n, mg->d);
    for(i=0; i<mg->n; ++i){
        get_merger(m, mg, i);
        print_merger(m);
    }
    free_merger(m);
    fprintf(stdout, "\n");
}

// neural network intergration
typedef enum E_NEURAL_LAYER_TYPE{
    NLT_CONV_NORMAL,
    NLT_CONV_COMBINED,
    NLT_LAIN,
    NLT_MERGE,
    NLT_MAX_POOL,
    NLT_MEAN_POOL,
    NLT_INPUT,
    NLT_FULL_CONN,
    NLT_SOFTMAX
}neural_layer_type;

// notice that input and output dimensions
// are defined by the training dataset, and
// they are supposed to be initialized at
// the very first begining of net construction.
typedef struct T_PARAM{
    int conv_n; // filter number
    int conv_h; // filter height
    int conv_w; // filter weight
    int lain_r; // lateral inhibition radius
    int merg_n; // number of mergers
    int pool_w; // pooling window width
    int pool_h; // pooling window height
    int full_n; // fully connected neurons
    int acti_f; // activation function(ReLU,Sigmoid,...)
}param;

typedef struct T_NEURAL_LAYER{
    neural_layer_type t;
    layer_group * l;
    param * p;
}neural_layer;

typedef struct T_LAYER_DIM{
    int d;
    int h;
    int w;
}dim;

typedef struct T_NET{
    int d; // depth
    neural_layer * l; // neural layers;
    dim i;
    int o;
    int lgi; // label group id
}net;

// set training dataset and set the input and
// output dimensions
// required to set the label group id for dataset
void net_set_train_ds(net * n, dataset * d, int lgi){
    int i, n;
    int hist[LABEL_CATEGORY];
    image * im;
    ASSERT(lgi<sizeof(label));
    n->lgi = lgi;
    memset(hist, 0, sizeof(hist));
    net->i.d = d->d;
    net->i.h = d->h;
    net->i.w = d->w;
    // stats for all label categories
    for(i=0; i<d->n; ++i){
        get_image(im, d, i);
        ASSERT(im->l.data[lgi]<LABEL_CATEGORY);
        hist[im->l.data[lgi]] ++;
    }
    for(i=0; i<LABEL_CATEGORY; ++i){
        if(!hist[i]){
            net->o = i;
            break;
        }
    }
    ASSERT(net->o>=2);
    free_image(im);
}

param * alloc_param(){
    return (param*)malloc(sizeof(param));
}

param * copy_param(param * p){
    param * _p = (param*)malloc(sizeof(param));
    memmove((char*)_p, (char*)p, sizeof(param));
    return _p;
}

void free_param(param * p){
    free(p);
}

void alloc_input_neural_layer(
    neural_layer * nl,
    neural_layer * l,
    net * n
){
    l->t = NLT_INPUT;
    l->l = alloc_layer_group(n->i.d, 1, n->i.h, n->i.w);
    l->p = NULL;
}

void free_neural_layer(neural_layer * l){
    free_param(l->p);
    free_layer_group(l->l);
}

net * alloc_net(){
    return (net*)malloc(sizeof(net));
}

void free_net(net * n){
}

void net_set_depth(net * n, int d){
    n->d = d;
    n->l = (neural_layer*)malloc(sizeof(neural_layer)*d);
}

void net_set_layer(net * n, int id, neural_layer_type t, param * p){
    ASSERT(t!=NLT_INPUT);
    n->l[id].p = copy_param(p);
    if(t==NLT_CONV_NORMAL){
    }
}

void net_construct_layers(net * n){
}


#endif
