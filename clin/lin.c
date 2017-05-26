#include <stdio.h>
#include "lib_read_dataset.h"

int main()
{
	dataset * ds = alloc_cifar_10_dataset();
	read_cifar_10_bacth(ds, 1);
	fprintf(stdout, "OK.\n");
	return 0;
}
