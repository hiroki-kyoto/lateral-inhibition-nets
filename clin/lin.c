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
	free_image(im);

	// test convolution
	layer * il; // input
	layer * tl; // tmp layer
	layer * ol; // output
	group * g1; // first filter group
	g1 = alloc_group(8, 2, 2);
	group_rand(g1, 0.0, 1.0);
	il = alloc_layer(cifar_10_train->d, cifar_10_train->h, cifar_10_train->w);
	
	tl = alloc_next_layer_with_conv(il, g1);
	ol = alloc_same_size_layer(tl);
	conv_valid(tl, il, g1);
	act(ol, tl, ACT_RLU);

	// free layers
	free_layer(il);
	free_layer(tl);
	free_layer(ol);
	free_group(g1);
	// free datasets
	free_dataset(cifar_10_train);
	free_dataset(cifar_10_test);
	free_dataset(cifar_100_train);
	free_dataset(cifar_100_test);

	return 0;
}
