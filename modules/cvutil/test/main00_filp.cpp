

#include "cvext/cvext.h"

#include <stdio.h>
#include <stdlib.h>

int main()
{
	FILE * fp = fopen("markingData_body_frontal.txt", "r");
	FILE * fout = fopen("markingData_body_frontal_flipped.txt", "w");
	char fn[1024]; int no; int x, y;

	int nfiles = 12;
	for (int ff = 0; ff < nfiles; ff++){
		fscanf(fp, "%s %d\n", fn, &no);
		fprintf(fout, "%s %d\n", fn, no);
		for (int i = 0; i < 23; i++){
			fscanf(fp, "%d %d\n", &x, &y);
			fprintf(fout, "%d %d\n", x, 240-y);
		}fscanf(fp, "\n");fprintf(fout, "\n");
	}

	fclose(fp);
	fclose(fout);
	return 0;
}

