// lib_math.h
#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <math.h>

float RLU(float x){
    return x>0?x:0;
}

float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
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



#endif
