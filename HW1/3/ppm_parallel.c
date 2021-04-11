#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

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
void mode1_parallel (PPM_Image* orig_img, PPM_Image* result_img, int rank, int numtasks, int spr) {
	int i, j, width;

	width = orig_img ->width;

	for (i = spr * rank ; i < spr * (rank + 1) ; i++) {
		for (j = 0 ; j < result_img->width; j++) {
			result_img->data[i][j] = orig_img->data[i][width - j - 1];
		}
	}

	if (rank == 0) {
		for (i = spr * numtasks ; i < result_img->height ; i++) {
			for (j = 0 ; j < result_img->width ; j++) {
				result_img->data[i][j] = orig_img->data[i][width - j - 1];
			}
		}
	}
}

void mode2_parallel (PPM_Image* orig_img, PPM_Image* result_img, int rank, int numtasks, int spr) {
	int i, j, temp;

	for (i = spr * rank ; i < spr * (rank + 1) ; i++) {
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

	if (rank == 0) {
		for (i = spr * numtasks ; i < result_img->height ; i++) {
			for (j = 0 ; j < result_img->width ; j++) {
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
}

void mode3_parallel (PPM_Image* orig_img, PPM_Image* result_img, int rank, int numtasks, int spr) {
	int i, j, k, l;
	int r, g, b, count;

	for (i = spr * rank ; i < spr * (rank + 1) ; i++) {
		for (j = 0 ; j < result_img->width; j++) {
			count = 0;
			r = 0; g = 0; b = 0;
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

	if (rank == 0) {
		for (i = spr * numtasks ; i < result_img->height ; i++) {
			for (j = 0 ; j < result_img->width ; j++) {
				count = 0;
				r = 0; g = 0; b = 0;
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
}

int main (int argc, char *argv[]) {
	int numtasks, rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	FILE* fp;
	char read_image_name[100], write_image_name[100];
	int mode = 0, i, j, spr; //spr : size per rank
	double start, end;
	PPM_Image *img, *result_img;
	img = (PPM_Image*) malloc (sizeof(PPM_Image));
	result_img = (PPM_Image*) malloc (sizeof(PPM_Image));
	
	MPI_Status stat;
	
	if (rank == 0) { //get ppm file&allocate memory for result in rank 0
		printf ("input image name : \n");
		scanf ("%s", read_image_name);
		if (read_image_name == NULL) {
			printf ("file name error!\n");
			mode = -1;
		}
	
		if (mode != -1) {
			fp = fopen (read_image_name, "rb");
			if (fp == NULL) {
				printf ("cannot read file!\n");
				mode = -1;
			}
		}
	
		if (mode != -1) {
			if (readPPM(fp,img) == -1) {
				fclose(fp);
				mode = -1;
			}
		}
	}

	MPI_Bcast (&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (mode == -1) {//case of error
		MPI_Finalize();
		return 0;
	}

	MPI_Bcast (&img->height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&img->width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	img->max = 255;


	if (mode != -1) {
		result_img->width = img->width;
		result_img->height = img ->height;
		//memory allocation
		result_img -> data = (pixel**) calloc (result_img -> height, sizeof (pixel*));
		for (i = 0 ; i < result_img->height ; i++) {
			result_img->data[i] = (pixel*)calloc(result_img->width, sizeof(pixel));
		}
	}

	if (rank != 0) {
		//memory allocation
		img -> data = (pixel**) calloc (img -> height, sizeof (pixel*));
		for (i = 0 ; i < img->height ; i++) {
			img->data[i] = (pixel*)calloc(img->width, sizeof(pixel));
		}
	}

	//derive datatypes
	MPI_Datatype MPI_pixel;
	MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR,&MPI_pixel);
	MPI_Type_commit (&MPI_pixel);

	MPI_Datatype MPI_pixel_total;
	MPI_Type_contiguous(img->width * img->height, MPI_pixel,&MPI_pixel_total);
	MPI_Type_commit (&MPI_pixel_total);

	MPI_Bcast (&img->data[0][0], 1, MPI_pixel_total, 0, MPI_COMM_WORLD);
	spr = img->height / numtasks;
	//derived datatypes end

	for ( ; ; ) {
		if (rank == 0) {	
			printf ("----Mode----\n");
			while (1) {
			printf ("1. flip the image horizontally\n");
			printf ("2. reduce the image to grayscale\n");
				printf ("3. smooth the image\n");
				printf ("0. exit\n");
				printf ("select mode : \n");
				scanf ("%d", &mode);
				
				if (mode > 0 && mode < 4) break;
				else if (mode == 0) break;
				else {
					printf ("wrong mode! select again\n");
				}
			}
		}

		start = MPI_Wtime();

		MPI_Bcast (&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (mode == 0) break;
		else if (mode == 1) {
			mode1_parallel (img, result_img, rank, numtasks, spr);
			
			for (i = 0 ; i < numtasks ; i++) {
				if (spr != 0)
					MPI_Bcast (&result_img->data[i * spr][0], spr * result_img->width, MPI_pixel, i, MPI_COMM_WORLD);
			}

			if (rank == 0) { //write at mode 0
				strcpy (write_image_name, "1_par_");
				strcat (write_image_name, read_image_name);
				writePPM (write_image_name, result_img);
				
				end = MPI_Wtime();
				printf ("time is %e\n", end-start);
			}
		}
		else if (mode == 2) {
			mode2_parallel (img, result_img, rank, numtasks, spr);		

			for (i = 0 ; i < numtasks ; i++) {
				if (spr != 0)
					MPI_Bcast (&result_img->data[i * spr][0], spr * result_img->width, MPI_pixel, i, MPI_COMM_WORLD);
			}

			if (rank == 0) {
				strcpy (write_image_name, "2_par_");
				strcat (write_image_name, read_image_name);
				writePPM (write_image_name, result_img);
				
				end = MPI_Wtime();
				printf ("time is %e\n", end-start);
			}
		}
		else if (mode == 3){
			mode3_parallel (img, result_img, rank, numtasks, spr);		

			for (i = 0 ; i < numtasks ; i++) {
				if (spr != 0)
					MPI_Bcast (&result_img->data[i * spr][0], spr * result_img->width, MPI_pixel, i, MPI_COMM_WORLD);
			}
	
			if (rank == 0) {
				strcpy (write_image_name, "3_par_");
				strcat (write_image_name, read_image_name);
				writePPM (write_image_name, result_img);

				end = MPI_Wtime();
				printf ("time is %e\n", end-start);
			}
		}
	}

	MPI_Finalize();
}
