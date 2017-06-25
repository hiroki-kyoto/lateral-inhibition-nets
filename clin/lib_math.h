// lib_math.h
#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
void merge_unit(
	layer_group * nl, // net layer group
	layer_group * l, // layer provided to be operated
	merger_group * mg, // the merger operator
	int z, // target channel position of pixel in next layer
	int y, // target vertical position of pixel in layer
	int x // target horizontal position of pixel in layer
){
	float sum;
	int _i, _k;
	// dimension match?
	ASSERT(l->n*l->l[0]->d==mg->d);
	sum = M_G_B(mg, z); //bias
	for(_i=0; _i<l->n; ++_i){
		for(_k=0; _k<l->l[0]->d; ++_k){
			sum += M_G_P(mg, z, _i*l->l[0]->d+_k) * L_G_P(l, _i, _k, y, x);
		}
	}
	L_S_P(nl, z, 0, y, x, sum);
}

void merge(
	layer_group * nl, 
	layer_group * l, 
	merger_group * mg
){
	int _i, _j, _k;
	ASSERT(nl->n==mg->n);
	ASSERT(nl->l[0]->d==1);
	ASSERT(l->n*l->l[0]->d==mg->d);
	ASSERT(nl->l[0]->w==l->l[0]->w);
	ASSERT(nl->l[0]->h==l->l[0]->h);
	for(_i=0; _i<nl->n; ++_i){ // parallel-x
		for(_j=0; _j<nl->l[0]->h; ++_j){ // parallel-y
			for(_k=0; _k<nl->l[0]->w; ++_k){ // parallel-z
				merge_unit(nl, l, mg, _i, _j, _k);
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
				net_set_layer(n, k, NLT_MERGE, p);
			} else if(i_str_cmp(t, "conv")){
				p->conv_n = d;
				p->conv_h = h;
				p->conv_w = w;
				p->acti_f = a;
				net_set_layer(n, k, NLT_CONV_NORMAL, p);
			} else if(i_str_cmp(t, "pool_max")){
				p->pool_h = d;
				p->pool_w = h;
				p->acti_f = a;
				net_set_layer(n, k, NLT_MAX_POOL, p);	
			} else if(i_str_cmp(t, "lain")){
				p->lain_r = d;
				p->acti_f = a;
				net_set_layer(n, k, NLT_LAIN, p);
			} else if(i_str_cmp(t, "conv_comb")){
				p->conv_n = d;
				p->conv_h = h;
				p->conv_w = w;
				p->acti_f = a;
				net_set_layer(n, k, NLT_CONV_COMBINED, p);
			} else if(t=="pool_mean"){
				p->pool_h = d;
				p->pool_w = h;
				p->acti_f = a;
				net_set_layer(n, k, NLT_MEAN_POOL, p);
			} else if(t=="full_conn"){
				p->full_n = d;
				net_set_layer(n, k, NLT_FULL_CONN, p);
			} else if(t=="softmax"){
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


// compute current layer
// n : net
// i : index of current layer to compute over
void compute_layer(
	net * n, 
	trainer * t,  
	int i
){ // trainer is only used in batch normalization mode
	ASSERT(i>0 && i<n->d);
	if(n->l[i].t==NLT_MERGE){
		DOT("merge");
		merge(n->l[i].l, n->l[i-1].l, n->l[i].m);
	} else if(n->l[i].t==NLT_CONV_NORMAL){
		//conv(n->l[i].l, n->l[i-1].l, n->l[i].g);
	}
}

// n : the net
// t : the trainer
// d : the dataset
// return the computing state of next step
int compute_forward(net * n, trainer * t, dataset * d){
	int _i, _s;
	char _str[256];
	// load sample into the input layer
	_s = net_load_single_input(n, d, t);
	// compute all layer from input to output
	for(_i=1; _i<n->d; ++_i){
		compute_layer(n, t, _i);
	}
	if(!_s){
		sprintf(_str, "An instance of training with %d epoches of %d samples finished!\n", t->n, d->n);
		DOT(_str);
		return STT_STOP;
	} else {
		return STT_CONTINUE;
	}
}

// trainer configuration once for all
// [*] initialization order is :
// [*] dataset
// [*] net
// [*] trainer
trainer * create_trainer(
	dataset * ds,
	net * nt,
	SEQ_GEN_MODE sgm,
	TRAIN_METHOD m,
	float e, // learning rate
	float d, // learning descending rate
	float n, // number of total epoches to train
	float bs // batch size
){	
	trainer * t = alloc_trainer();
	trainer_set_method(t, m);
	trainer_set_learning_rate(t, e);
	trainer_set_descending_rate(t, d);
	trainer_set_max_epoch_num(t, n);
	trainer_set_batch_size(t, bs);
	trainer_init(t, ds);
	trainer_set_seq(t, ds, nt, sgm);
	return t;
}


// calculate the partial gradient for merge layer
void 
merge_grad(
	net * n,			// net
	trainer * t,	// trainer
	int i					// current neural layer index
	){
	int _i, _k, _s, _t, _v;
	
	int _f;			// activation function
	int _n[2];	// number of layers
	int _d[2];	// number of channels
	int _h[2];	// height of layers
	int _w[2];	// width of layers
	
	float _sum;	// summation
	float _dy;	// dy[i]
	float _y;		// y[i]
	float _x;		// x[i] = y[i-1]
	float _p;		// pixel in merger or filter
	float _b;		// bias in merger or filter

	layer_group * _l[2];	// layer groups
	layer_group * _e[2];	// error layer groups
	merger_group * _m[2];	// merger groups
	param * _pr;					// parameters

	_l[0] = n->l[i-1].l;
	_l[1] = n->l[i].l;
	_e[0] = n->l[i-1].e;
	_e[1] = n->l[i].e;
	_m[0] = n->l[i-1].m;
	_m[1] = n->l[i].m;
	_pr		= n->l[i].p;

	_f		= _pr->acti_f;
	_n[0] = _l[0]->n;
	_n[1] = _l[1]->n;
	_d[0] = _l[0]->l[0]->d;
	_d[1] = _l[1]->l[0]->d;
	_h[0] = _l[0]->l[0]->h;
	_h[1] = _l[1]->l[0]->h;
	_w[0] = _l[0]->l[0]->w;
	_w[1] = _l[1]->l[0]->w;
	
	// update previous error layer first
	for(_i=0; _i<_n[0]; ++_i){
		for(_k=0; _k<_d[0]; ++_k){
			for(_s=0; _s<_h[0]; ++_s){
				for(_t=0; _t<_w[0]; ++_t){
					_sum = 0;
					for(_v=0; _v<_n[1]; ++_v){
						_y = L_G_P(_l[1], _v, 0, _s, _t);
						_dy = L_G_P(_e[1], _v, 0, _s, _t);
						_p = M_G_P(_m[1], _v, _i*_d[0]+_k);
						if(_f==ACT_RELU){
							_sum += _dy*(_y>0)*_p;
						} else if(_f==ACT_SIGMOID){
							_sum += _dy*_y*(1-_y)*_p; 
						}
					}
					L_S_P(_e[0], _i, _k, _s, _t, _sum);
				}
			}
		}
	}

	// update current layer parameters
	for(_v=0; _v<_n[1]; ++_v){
		for(_i=0; _i<_n[0]; ++_i){
			for(_k=0; _k<_d[0]; ++_k){
				_sum = 0;
				for(_s=0; _s<_h[1]; ++_s){
					for(_t=0; _t<_w[1]; ++_t){
						_y = L_G_P(_l[1], _v, 0, _s, _t);
						_dy = L_G_P(_e[1], _v, 0, _s, _t);
						_x = L_G_P(_l[0], _i, _k, _s, _t);
						if(_f==ACT_RELU){
							_sum += _dy*(_y>0)*_x;
						} else if(_f==ACT_SIGMOID){
							_sum += _dy*_y*(1-_y)*_x;
						}
					}
				}
				// update trainable parameters
				M_S_P(_m, _v, _i*_d[0]+_k, _sum*t->ce);
			}
		}
		_sum = 0;
		for(_s=0; _s<_h[1]; ++_s){
			for(_t=0; _t<_w[1]; ++_t){
				_y = L_G_P(_l[1], _v, 0, _s, _t);
				_dy = L_G_P(_e[1], _v, 0, _s, _t);
				_sum += _dy*(_y>0);
			}
		}
		M_S_B(_m, _v, _sum*t->ce);
	}
}


// compute gradient and store them on layers,
// then update the trainable parameters
void 
desc_layer(
		net * n,
		trainer * t,
		int i
		){
	ASSERT(i>0 && i<n->d);
	if(n->l[i].t==NLT_MERGE){
		merge_grad(n, t, i);
	} else if(n->l[i].t==NLT_CONV_NORMAL){
		// conv_normal_grad
	}
}

// update learning rate in trainer
// t : trainer
// e : current training error
void trainer_update_learning_rate(trainer * t, float e){
	if(t->m==TM_SGD){
		t->ce = (1 - t->d) * t->ce;
	} else if(t->m==TM_ADA_DELTA){
		DOT("NOT SUPPORTED YET!\n");
		ASSERT(0);
	} else {
		DOT("CONFIGURE ERROR!\n");
		ASSERT(0);
	}
}


// return the current output error
float compute_back(
	net * n,
	trainer * t,
	dataset * d
){
	// compute the overrall error
	float _e; // total error
	float _t; // temp
	int _i; // output index

	// compute the partial gradient
	for(_i=0; _i<n->o; ++_i){
		_t = L_G_P(n->l[n->d-1].l, _i, 0, 0, 0) - D_G_L(d, n->lgi, t->li, _i);
		L_S_P(n->l[n->d-1].e, _i, 0, 0, 0, _t);
		_e += _t * _t;
	}
	_e = 0.5 * _e / n->o;
	
	// update learnning rate
	trainer_update_learning_rate(t, _e);

	// compute the error partial gradients descending the layers
	for(_i=n->d-1; _i>0; --_i){
		desc_layer(n, t, _i);
	}

	return _e;
}

// initialization of net
typedef enum E_NET_INIT_METHOD{
	NIM_RANDOM_ZERO,
	NIM_RANDOM_ONE
}NET_INIT_METHOD;

// neural layer parameter initialization
void neural_layer_init(
	neural_layer * l,
	float f_min,
	float f_max
){
	int _i, _j, _k;
	if(l->t==NLT_MERGE){
		// merger param init
		for(_i=0; _i<l->m->n; ++_i){
			for(_j=0; _j<l->m->d; ++_j){
				M_S_P(l->m, _i, _j, (rand()%1000)/1000.0*(f_max-f_min)+f_min);
			}
			M_S_B(l->m, _i, (rand()%1000)/1000.0*(f_max-f_min)+f_min);
		}
	} else if(l->t==NLT_CONV_NORMAL||l->t==NLT_CONV_COMBINED||l->t==NLT_FULL_CONN){
		for(_i=0; _i<l->g->n; ++_i){
			for(_j=0; _j<l->g->h; ++_j){
				for(_k=0; _k<l->g->w; ++_k){
					F_S_P(l->g, _i, _j, _k, (rand()%1000)/1000.0*(f_max-f_min)+f_min);
				}
			}
			F_S_B(l->g, _i, (rand()%1000)/1000.0*(f_max-f_min)+f_min);
		}
	}
}

// parameter initialization
void net_init(net * n, NET_INIT_METHOD m){
	int _i;
	if(m==NIM_RANDOM_ZERO){
		// each is randomized between -1 and 1 
		for(_i=1; _i<n->d; ++_i){
			neural_layer_init(n->l+_i, -1, 1);
		}
	} else if(m==NIM_RANDOM_ZERO){
		// each is randomized between 0 and 1 
		for(_i=1; _i<n->d; ++_i){
			neural_layer_init(n->l+_i, 0, 1);
		}
	}
}

// training
void train(
	net * n, 
	trainer * t,
	dataset * d
){
	char str[256];
	STATUS stt;
	float err;
	while(stt!=STT_STOP){
		sprintf(str, "EPOCH:%d\tBATCH:%d.\n", t->ei, t->bi);
		DOT(str); 
		stt = compute_forward(n, t, d);
		err = compute_back(n, t, d);
		sprintf(str, "CURRENT ERROR:%f.\n", err);
		DOT(str);
	}
}




#endif

