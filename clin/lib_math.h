// lib_math.h
#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <stdlib.h>
#include <math.h>

typedef enum E_ACT_FUNC{
    ACT_RELU,
    ACT_SIGMOID
}ACT_FUNC;

typedef enum E_POOL_FUNC{
    POOL_MAX,
    POOL_MEAN
}POOL_FUNC;

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
    ASSERT(a==ACT_RELU || a==ACT_SIGMOID);
    // in GPUs, this will remap into 3-dim task
    for(i=0; i<l->d; ++i){
        for(k=0; k<l->h; ++k){
            for(m=0; m<l->w; ++m){
                if(a==ACT_RELU){
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
// channel merge
// stacked filters(3x3=3x1+1x3)(Google Inception V4)
// max-pooling(mean pooling ?)
// stochastic gradient descending
// direct classifier: from map to output:LI+SUM
// try or not : ResNet...

void merger_unit(
    layer * nl,
    layer * l,
    merger_group * mg,
    int z, // up to depth of nl
    int y, // up to height of nl
    int x // up to width of nl
){
    float sum;
    int t;
    sum = mg->p[z*(mg->d+1)+l->d]; //bias
    for(t=0; t<l->d; ++t){
        sum += mg->p[z*(mg->d+1)+t]\
        *l->p[t*l->h*l->w+y*l->w+x];
    }
    nl->p[z*nl->h*nl->w+y*nl->w+x] = sum;
}

void merge(layer * nl, layer * l, merger_group * mg){
    int i, k, s;
    ASSERT(nl->d==mg->n);
    ASSERT(l->d==mg->d);
    ASSERT(nl->w==l->w);
    ASSERT(nl->h==l->h);
    for(i=0; i<nl->h; ++i){ // parallel-x
        for(k=0; k<nl->w; ++k){ // parallel-y
            for(s=0; s<nl->d; ++s){ // parallel-z
                merger_unit(nl, l, mg, s, i, k);
            }
        }
    }
}

float vmax(float * v, int n){
    int i;
    float m;
    m = v[0];
    for(i=1; i<n; ++i){
        m = m<v[i]?v[i]:m;
    }
    return m;
}

float vmean(float * v, int n){
    int i;
    float m;
    m = v[0];
    for(i=1; i<n; ++i){
        m += v[i];
    }
    return m/n;
}

// we're not going to use POOLing APIs
void pool(
    layer * nl,
    layer * l,
    POOL_FUNC p,
    int h,
    int w
){
    int i, k, m, s, t, x, y;
    float * v = (float*)malloc(sizeof(float)*w*h);
    ASSERT(w>0&&h>0);
    ASSERT(nl->w==(l->w-1)/w+1);
    ASSERT(nl->h==(l->h-1)/h+1);
    ASSERT(nl->d==l->d);
    for(i=0; i<nl->h; ++i){
        for(k=0; k<nl->w; ++k){
            for(m=0; m<nl->d; ++m){
                for(s=0; s<h; ++s){
                    for(t=0; t<w; ++t){
                        if(i*h+s<l->h){
                            y = i*h + s;
                        } else {
                            y = l->h - 1;
                        }
                        if(k*w+t<l->w){
                            x = k*w + s;
                        } else {
                            x = l->w - 1;
                        }
                        v[s*w+t] = l->p[m*l->h*l->w+y*l->w+x];
                    }
                }
                if(p==POOL_MAX){
                    nl->p[m*(nl->w*nl->h)+i*nl->w+k] =
                    vmax(v, w*h);
                }
                else if(p==POOL_MEAN){
                    nl->p[m*(nl->w*nl->h)+i*nl->w+k] =
                    vmean(v, w*h);
                }
            }
        }
    }
    free(v);
}


/**** constructing nets with a user-defined script file *****/
void load_net_model(net * n, const char * file){
	char c, str[256], t[64], f[64];
	int i, k, d, h, w;
	FILE * fp = fopen(file, "rt");
	ACT_FUNC a; // activation function option
	param * p = alloc_param();
	// this recommends the net model be initialized with
	// some training dataset required reasonably
	ASSERT(fp);
	ASSERT(n);
	ASSERT(n->o>0 && n->i.d>0 && n->i.h>0 && n->i.w>0);
	// get layer number
	k = 0;
	while(!feof(fp)){
		if(fgetc(fp)=='\n'){
			++k; // layer increments
		}
	}
	net_set_depth(n, k+1); // add an input layer
	i = 0;
	k = 1;
	fseek(fp, 0L, SEEK_SET);
	while(!feof(fp)){
		ASSERT(i<sizeof(str)-1);
		str[i] = fgetc(fp);
		str[i+1] = 0;
		if(str[i]=='\n'){
			// extract configuration parameters
			memset(t, 0, sizeof(t));
			sscanf(str, "%s%s%d%d%d", t, f, &d, &h, &w);
			fprintf(stdout, "layer#%d\n", k+1);
			fprintf(stdout, "type:\t%s\n", t);
			fprintf(stdout, "act:\t%s\n", f);
			fprintf(stdout, "depth:\t%d\n", d);
			fprintf(stdout, "height:\t%d\n", h);
			fprintf(stdout, "width:\t%d\n", w);
			// check validality for the activation function
			if(i_str_cmp(f, "sigmoid")){
				a = ACT_SIGMOID;
			} else if(i_str_cmp(f, "relu")){
				a = ACT_RELU;
			} else {
				ASSERT(0); // invalid option for activation
			}
			// construct layers
			if(i_str_cmp(t, "merge")){
				p->merg_n = d;
				p->acti_f = a;
				if(k==n->d-1){ // output layer
					net_set_output_layer(
						n, 
						NLT_MERGE, 
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_MERGE, 
						p
					);
				}
			} else if(i_str_cmp(t, "conv")){
				p->conv_n = d;
				p->conv_h = h;
				p->conv_w = w;
				p->acti_f = a;
				if(k==n->d-1){
					net_set_output_layer(
						n, 
						NLT_CONV_NORMAL,
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_CONV_NORMAL, 
						p
					);
				}
			} else if(i_str_cmp(t, "pool_max")){
				p->pool_h = d;
				p->pool_w = h;
				p->acti_f = a;
				if(k==n->d-1){
					net_set_output_layer(
						n,
						NLT_MAX_POOL,
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_MAX_POOL, 
						p
					);
				}
			} else if(i_str_cmp(t, "lain")){
				p->lain_r = d;
				p->acti_f = a;
				if(k==n->d-1){
					net_set_output_layer(
						n,
						NLT_LAIN,
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_LAIN, 
						p
					);
				}
			} else if(i_str_cmp(t, "conv_comb")){
				p->conv_n = d;
				p->conv_h = h;
				p->conv_w = w;
				p->acti_f = a;
				if(k==n->d-1){
					net_set_output_layer(
						n,
						NLT_CONV_COMBINED,
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_CONV_COMBINED, 
						p
					);
				}
			} else if(t=="pool_mean"){
				p->pool_h = d;
				p->pool_w = h;
				p->acti_f = a;
				if(k==n->d-1){
					net_set_output_layer(
						n,
						NLT_MEAN_POOL,
						p
					);
				} else {
					net_set_layer(
						n, 
						k, 
						NLT_MEAN_POOL, 
						p
					);
				}
			} else if(t=="softmax"){
				// not applicabel yet
				DOT("SOFTMAX not applicable yet!\n");
				ASSERT(0);
			} else {
				DOT("bad configuration of net model!\n");
				ASSERT(0);
			}
			++k;
			i = 0;
		} else {
			++i;
		}
	}
	fclose(fp);
}



// softmax layer computation
// all neurons are considered as filters
void softmax(layer_group * l, group * g){
    // to be implemented
}

void comput_forward(net * n, trainer * t, dataset * d){
	// start from the input layer
	ASSERT(t->n<1024); // epoch num should be reasonable
	
}



#endif

