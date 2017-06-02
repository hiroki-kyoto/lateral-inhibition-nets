#include <stdio.h>
#include "lib_read_dataset.h"
#include "lib_math.h"

int main()
{
	// read datasets
	dataset * cifar_10_train = get_cifar_10_train();
	fprintf(stdout, "CIFAR-10 TRAINING DATA READ OK.\n");
	dataset * cifar_10_test = get_cifar_10_test();
	fprintf(stdout, "CIFAR-10 TEST DATA READ OK.\n");
	dataset * cifar_100_train = get_cifar_100_train();
	fprintf(stdout, "CIFAR-100 TRAINING DATA READ OK.\n");
	dataset * cifar_100_test = get_cifar_100_test();
	fprintf(stdout, "CIFAR-100 TEST DATA READ OK.\n");

	image * im = alloc_image();
	get_image(im, cifar_10_train, 49999);
	fprintf(stdout, "%u\n", im->l.data[0]);
	get_image(im, cifar_10_train, 0);
	fprintf(stdout, "%u\n", im->l.data[0]);
	get_image(im, cifar_100_train, 49999);
	fprintf(stdout, "%u\n", im->l.data[0]);
	fprintf(stdout, "%u\n", im->l.data[1]);

	// test convolution
	layer * il; // input
	group * g1; // first filter group
	layer * tl; // tmp layer
	layer * cl; // convolution layer
	lainer * la; // lateral inhibitor
	layer * el; // extended layer for lateral inhibition
	layer * tll; // temp lateral inhibition layer
	layer * ll; // output layer

	//map * m; // the map for test
	//filter * f; // the filter for test
	il = alloc_input_layer(cifar_10_train);
	//il = alloc_layer(1, 3, 3);
	g1 = alloc_group(2, 2, 2);
	tl = alloc_next_layer_with_conv(il, g1);
	cl = alloc_same_size_layer(tl);
	la = alloc_lainer(1);
	DOT("lateral inhibitor:");
	print_lainer(la);
	el = alloc_extended_layer_with_lainer(cl, la);
	tll = alloc_same_size_layer(cl); // temp lateral inhibition layer
	ll = alloc_same_size_layer(cl); // lateral inhibited layer

	group_rand(g1, -1.0, 1.0);
	/*f = alloc_filter();
	get_filter(f, g1, 0);
	f_w_s(f, 1, 0, 0);
	f_w_s(f, -1, 1, 0);
	f_w_s(f, -1, 0, 1);
	f_w_s(f, 1, 1, 1);
	f_b_s(f, 1);
	get_filter(f, g1, 1);
	f_w_s(f, -1, 0, 0);
	f_w_s(f, 1, 1, 0);
	f_w_s(f, 1, 0, 1);
	f_w_s(f, -1, 1, 1);
	f_b_s(f, -1);
	free_filter(f);
	DOT("filter group#1:");
	*/
	print_group(g1);

	/*m = alloc_map();
	get_map(m, il, 0);
	m_e_s(m, 1, 0, 0);
	m_e_s(m, 5, 1, 0);
	m_e_s(m, 3, 2, 0);
	m_e_s(m, 2, 0, 1);
	m_e_s(m, 3, 1, 1);
	m_e_s(m, 2, 2, 1);
	m_e_s(m, 1, 0, 2);
	m_e_s(m, 6, 1, 2);
	m_e_s(m, 5, 2, 2);
	free_map(m);*/
	get_image(im, cifar_10_train, 0);
	load_input(il, im);
	DOT("input layer:");
	print_layer(il);
	conv_valid(tl, il, g1);
	DOT("temp layer#1:");
	print_layer(tl);
	act(cl, tl, ACT_RLU);
	DOT("conv layer#1:");
	print_layer(cl);
	// after convolution, apply lateral inhibition immediately
	lain(tll, el, cl, la);
	DOT("extended layer#1:");
	print_layer(el);
	DOT("temp lateral inhibited layer#1");
	print_layer(tll);
	act(ll, tll, ACT_RLU);
	DOT("lateral inhibited layer#1:");
	print_layer(ll);

	// free layers
	free_image(im);
	free_layer(ll);
	free_layer(tll);
	free_layer(el);
	free_lainer(la);
	free_layer(cl);
	free_layer(tl);
	free_group(g1);
	free_layer(il);
	// free datasets
	free_dataset(cifar_10_train);
	free_dataset(cifar_10_test);
	free_dataset(cifar_100_train);
	free_dataset(cifar_100_test);

	return 0;
}
