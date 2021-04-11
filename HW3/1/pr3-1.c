#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NUM_WORDS 25143
#define WORD_LENGTH 20

int is_palindrome (char*);
int list_palindrome (char*, char*[]); 

int main (int argc, char* argv[]){ 
	int i;
	char *words[NUM_WORDS];

	/* set thread */
	int thread_count = strtol (argv[1], NULL, 10);
	omp_set_num_threads (thread_count);

	/* open files */
	FILE *input_fp, *output_fp;
	input_fp = fopen (argv[2], "r");
	if (input_fp == NULL) {
		printf ("input file not found!\n");
		exit(1);
	}
	output_fp = fopen (argv[3], "w");

	/* start timer */
	double start_time, end_time;
	start_time = omp_get_wtime();

	/* get words from the text file to an array */
# pragma omp parallel for
	for (i = 0 ; i < NUM_WORDS ; i++) {
		words[i] = malloc (WORD_LENGTH);
		fscanf (input_fp, "%s", words[i]);
	}

	/* find palindrome words */
# pragma omp parallel for
	for (i = 0 ; i < NUM_WORDS ; i++) {
		// palindrome word
		if (is_palindrome (words[i]) == 1) {
			fprintf (output_fp, "%s\n", words[i]);
		}
		// find if reversed word is in list
		else if (list_palindrome (words[i], words) == 1) {
			fprintf (output_fp, "%s\n", words[i]);
		}
	}

	/* free array */
# pragma omp parallel for
	for (i = 0 ; i < NUM_WORDS ; i++) {
		free (words[i]);
	}

	/* stop timer */
	end_time = omp_get_wtime();
	printf ("The time : %f seconds\n", end_time - start_time);

	fclose (input_fp);
	fclose (output_fp);

	return 0;
}

/* checks if a word is a palindrome */
int is_palindrome (char* word) {
	int i, j;
	j = strlen (word) - 1;
	
	for (i = 0 ; i <= j / 2 ; i++) {
		if (word[i] != word [j - i]) return 0;
	}

	return 1;
}

/* find if reversed word is in list */
int list_palindrome (char* word, char* words[]) {
	int i, j;
	j = strlen(word) - 1;
	char* temp = malloc (sizeof (word));

	for (i = 0 ; i <= j ; i++) { // reverse word
		temp[i] = word[j - i];
	}

	for (i = 0 ; i < NUM_WORDS ; i++) {
		if (strcmp (temp, words[i]) == 0) {
			free (temp);
			return 1;
		}
	}

	free (temp);
	return 0;
}
