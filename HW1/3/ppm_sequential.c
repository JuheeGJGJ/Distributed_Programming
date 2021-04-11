#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void GET_TIME (double* now) {
	struct timeval t;
	gettimeofday(&t,NULL);
	*now = t.tv_sec + t.tv_usec/ 1000000.0;
}

typedef struct {
	unsigned char r, g, b;
} pixel;

typedef struct {
	int width;
	int height;
	int max;
	pixel **data;
} PPM_Image;

int readPPM (FILE* fp, PPM_Image* img) {
	char buff[256];
	int i, j;

	if (!fgets (buff, sizeof(buff), fp)) { // read image format
		printf ("file error!\n");
		return -1;
	}

	if (buff[0] != 'P' || buff[1] != '6') { // check image format
		printf ("image format error! (P6)\n");
		return -1;
	}

	while (1) {
		fgets (buff, sizeof (buff), fp);
		if (buff[0] == '#') // comment
			continue;
		else {
			sscanf (buff, "%d %d", &img -> width, &img->height);
			break;
		}
	}

	fscanf (fp, "%d\n", &img -> max);

	if (img->max != 255) {
		printf ("image format error! (255)\n");
		return -1;
	}

	//memory allocation
	img -> data = (pixel**) calloc (img -> height, sizeof (pixel*));
	for (i = 0 ; i < img->height ; i++) {
		img->data[i] = (pixel*)calloc(img->width, sizeof(pixel));
	}

	// ppm -> memory
	for (i = 0 ; i < img->height ; i++) {
		for (j = 0 ; j < img->width ; j++) {
			fread (&img->data[i][j].r, sizeof(unsigned char), 1, fp);
			fread (&img->data[i][j].g, sizeof(unsigned char), 1, fp);
			fread (&img->data[i][j].b, sizeof(unsigned char), 1, fp);
		}
	}

	fclose(fp);
	return 1;
}

int writePPM (char* filename, PPM_Image* img) {
	FILE* fp;
	int i, j;

	fp = fopen (filename, "wb");
	if (fp == NULL) {
		printf ("file write fail\n");
		return -1;
	}

	fprintf (fp, "%c%c\n", 'P', '6');
	fprintf (fp, "%d %d\n", img->width, img->height);
	fprintf (fp, "%d\n", 255);

	for (i = 0; i < img->height ; i++) {
		for (j = 0; j < img->width ; j ++) {
			fwrite (&img->data[i][j],1,3,fp);
		}
	}

	fclose (fp);
	return 1;
}

// horizontally
void mode1_sequential (PPM_Image* orig_img, PPM_Image* result_img) {
	int i, j, width;

	width = orig_img ->width;

	for (i = 0 ; i < result_img->height ; i++) {
		for (j = 0 ; j < result_img->width; j++) {
			result_img->data[i][j] = orig_img->data[i][width - 1 - j];
		}
	}
}

void mode2_sequential (PPM_Image* orig_img, PPM_Image* result_img) {
	int i, j, temp;


	for (i = 0 ; i < result_img->height ; i++) {
		for (j = 0 ; j < result_img->width; j++) {
			temp = orig_img->data[i][j].r;
			temp += orig_img->data[i][j].g;
			temp += orig_img->data[i][j].b;
			temp /= 3;
			result_img->data[i][j].r = temp;
			result_img->data[i][j].g = temp;
			result_img->data[i][j].b = temp;
		}
	}

}
void mode3_sequential (PPM_Image* orig_img, PPM_Image* result_img) {
	int i, j, k, l, r, g, b;
	int count;

	for (i = 0 ; i < result_img->height ; i++) {
		for (j = 0 ; j < result_img->width; j++) {
			count = 0;
			r=0;g=0;b=0;
			for (k = -1 ; k < 2 ; k++) {
				if (i + k < 0 || i + k >= result_img->height) continue;
				for (l = -1 ; l < 2 ; l++) {
					if (j + l < 0 || j + l >= result_img->width) continue;
					r += orig_img->data[i+k][j+l].r;
					g += orig_img->data[i+k][j+l].g;
					b += orig_img->data[i+k][j+l].b;
					count++;
				}
			}

			result_img->data[i][j].r = r / count;
			result_img->data[i][j].g = g / count;
			result_img->data[i][j].b = b / count;
		}
	}

}

int main (int argc, char *argv[]) {
	PPM_Image *img, *result_img;
	img = (PPM_Image*) malloc (sizeof(PPM_Image));
	result_img = (PPM_Image*) malloc (sizeof(PPM_Image));
	
	int mode, i;
	char read_image_name[100], write_image_name[100];
	double start, finish;
	FILE* fp;

	printf ("input image name : ");
	scanf ("%s", read_image_name);
	if (read_image_name == NULL) {
		printf ("file name error!\n");
		return -1;
	}

	fp = fopen (read_image_name, "rb");
	if (fp == NULL) {
		printf ("cannot read file!\n");
		return -1;
	}
	
	if (readPPM(fp,img) == -1) {
		fclose(fp);
		return -1;
	}
	
	result_img->width = img->width;
	result_img->height = img ->height;
	//memory allocation
	result_img -> data = (pixel**) calloc (result_img -> height, sizeof (pixel*));
	for (i = 0 ; i < result_img->height ; i++) {
		result_img->data[i] = (pixel*)calloc(result_img->width, sizeof(pixel));
	}

	for ( ; ; ) {
		printf ("----Mode----\n");
		while (1) {
			printf ("1. flip the image horizontally\n");
			printf ("2. reduce the image to grayscale\n");
			printf ("3. smooth the image\n");
			printf ("0. exit\n");
			printf ("select mode : ");
			scanf ("%d", &mode);
			
			if (mode > 0 && mode < 4) break;
			else if (mode == 0) break;
			else {
				printf ("wrong mode! select again\n");
			}
		}

		GET_TIME(&start);
		
		if (mode == 0) break;
		else if (mode == 1) {
			strcpy (write_image_name, "1_seq_");
			strcat (write_image_name, read_image_name);
			mode1_sequential (img, result_img);		
			writePPM (write_image_name, result_img);

			GET_TIME (&finish);
			printf ("time : %e seconds\n", finish - start);
		}
		else if (mode == 2) {
			strcpy (write_image_name, "2_seq_");
			strcat (write_image_name, read_image_name);
			mode2_sequential (img, result_img);		
			writePPM (write_image_name, result_img);

			GET_TIME (&finish);
			printf ("time : %e seconds\n", finish - start);
		}
		else if (mode == 3){
			strcpy (write_image_name, "3_seq_");
			strcat (write_image_name, read_image_name);
			mode3_sequential (img, result_img);		
			writePPM (write_image_name, result_img);

			GET_TIME (&finish);
			printf ("time : %e seconds\n", finish - start);
		}

	}

}
