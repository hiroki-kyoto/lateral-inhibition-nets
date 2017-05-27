#include <stdio.h>
#include "lib_read_dataset.h"

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

	// free datasets
	free_dataset(cifar_10_train);
	free_dataset(cifar_10_test);
	free_dataset(cifar_100_train);
	free_dataset(cifar_100_test);
	
	return 0;
}
