// lib_math.h
#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <math.h>

typedef enum E_ACT_FUNC{
    ACT_RLU,
    ACT_SIGMOID
}ACT_FUNC;

float RLU(float x){
    return x>0?x:0;
}

float SIGMOID(float x){
    return 1.0/(1.0+exp(-x));
}

// activation function application
void act(layer * nl, layer * l, ACT_FUNC a){
    int i, k, m;
    ASSERT(nl->d==l->d);
    ASSERT(nl->h==l->h);
    ASSERT(nl->w==l->w);
    ASSERT(a==ACT_RLU || a==ACT_SIGMOID);
    // in GPUs, this will remap into 3-dim task
    for(i=0; i<l->d; ++i){
        for(k=0; k<l->h; ++k){
            for(m=0; m<l->w; ++m){
                if(a==ACT_RLU){
                    nl->p[i*l->h*l->w+k*l->w+m] =
                    RLU(l->p[i*l->h*l->w+k*l->w+m]);
                } else if(a==ACT_SIGMOID) {
                    nl->p[i*l->h*l->w+k*l->w+m] =
                    SIGMOID(l->p[i*l->h*l->w+k*l->w+m]);
                }
            }
        }
    }
}

float conv_unit_valid(map * m, filter * f, int y, int x){
    int i, k;
    float s;
    ASSERT(m!=NULL);
    ASSERT(f!=NULL);
    ASSERT(x>=0);
    ASSERT(x<m->w-f->w+1);
    ASSERT(y>=0);
    ASSERT(y<m->h-f->h+1);
    s = f->p[f->w*f->h]; // bias
    for(i=0; i<f->h; ++i){
        for(k=0; k<f->w; ++k){
            s += m->p[(i+y)*m->w+(k+x)]*f->p[i*f->w+k];
        }
    }
    return s;
}

// filter and map convolution
void conv_fm_valid(map * nm, map * m, filter * f){
    int i, k, u, v;
    ASSERT(m!=NULL);
    ASSERT(f!=NULL);
    u = m->h - f->h + 1;
    v = m->w - f->w + 1;
    ASSERT(nm!=NULL);
    ASSERT(nm->h==u);
    ASSERT(nm->w==v);
    for(i=0; i<u; ++i){
        for(k=0; k<v; ++k){
            nm->p[i*nm->w+k] = conv_unit_valid(m, f, i, k);
        }
    }
}

// layer and group convolution
// this will cut down dimension from a to a-b+1(with filter of dim b)
// [*] after conv_valid(), we should apply act() to the layer!!!
void conv_valid(layer * nl, layer * l, group * g){
    int i, k;
    map * m;
    map * nm;
    filter * f;
    ASSERT(nl!=NULL);
    ASSERT(l!=NULL);
    ASSERT(g!=NULL);
    ASSERT(nl->d==l->d*g->n);
    ASSERT(nl->w==l->w-g->w+1);
    ASSERT(nl->h==l->h-g->h+1);
    m = alloc_map();
    f = alloc_filter();
    nm = alloc_map();
    for(i=0; i<g->n; ++i){
        for(k=0; k<l->d; ++k){
            get_filter(f, g, i);
            get_map(m, l, k);
            get_map(nm, nl, i*l->d+k);
            conv_fm_valid(nm, m, f);
        }
    }
    free_map(m);
    free_map(nm);
    free_filter(f);
}

// nl: new layer
// l: previous layer
// r: radius for inhibition neighborhood
// algorithm: extend maps with (2r,2r)
// of zeros, apply convolution operation
// the same way as conv_valid().
// [*] lain() keeps the dimension as it is.
// [*] after lain, we should apply act() to the layer!!!
// [*] nl: foward layer to update
// [*] el: extended layer from layer [l] with r, also to update
// [*] la: lateral inhibitor
// [*] [el] do not need to be initialized
void lain(layer * nl, layer * el, layer * l, lainer * la){
    group * g;
    ASSERT(nl->d==l->d);
    ASSERT(nl->w==l->w);
    ASSERT(nl->h==l->h);
    ASSERT(el->d==l->d);
    ASSERT(el->w==l->w+2*la->r);
    ASSERT(el->h==l->h+2*la->r);
    g = alloc_group_with_filter(1, la->f);
    extend_layer_with_zeros(el, l, la->r);
    conv_valid(nl, el, g);
    free_group(g);
}

// to-do :
// channel merge,
// stacked filters(3x3=3x1+1x3)(Google Inception V4)
// max-pooling(mean pooling ?)
// stochastic gradient descending
// direct classifier: from map to output:LI+SUM
// try or not : ResNet...

void merge(layer * nl, layer * l, merger_group * mg){
    
}


#endif
