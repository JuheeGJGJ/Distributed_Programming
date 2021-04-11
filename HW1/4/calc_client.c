#include <stdio.h>
#include "calc.h"

int precedence (char op) {
	if (op == '+' || op == '-') return 1;
	else if (op == '*' || op == '/') return 2;
	else if (op == '^') return 3;
	else return -1;
}

int main (int argc, char *argv[]) {
	CLIENT *clnt;

	struct two_int calc_v;
	char calc_op;

	char string[200];
	int i, num;
	int* result;

	int value[10];
	char op[5];
	int value_pos, op_pos, error;

	if (argc != 2) {
		fprintf (stderr, "Usage\n");
		exit(1);
	}

	clnt = clnt_create (argv[1], CALC_PROG, CALC_VERS, "udp");

	if (clnt == (CLIENT*) NULL) {
		clnt_pcreateerror(argv[1]);
		exit(1);
	}

	for ( ; ; ) {
		scanf ("%s", string);
		if (strcmp (string, "exit") == 0) break;
		else if (strcmp (string, "test") == 0) {
			scanf ("%s", string); // get expression
			value_pos = -1; op_pos = -1; error = 0;
		}
		else continue;

		for (i = 0 ; i < strlen(string) ; i++) {
			if (string[i] <= '9' && string [i] >= '0') { // is a number
				num = 0;
				
				while (i < strlen(string) && string[i] <= '9' && string[i] >= '0') {
					num *= 10;
					num += string[i] - '0';
					i++;
				}
	
				value[++value_pos] = num;
				i--; 
			}
			else { //operators
				if (i == 0) {
					printf ("operation error!\n"); error = 1; break;
				}

				if (string[i] == '*' && string [i + 1] == '*') {
					string[++i] = '^'; //power
				}

				if (precedence(string[i]) == -1) {
					printf ("operation error!\n"); error = 1; break;
				}
	

				while (op_pos != -1 && precedence (op[op_pos]) >= precedence (string[i])) {
					if (value_pos < 1) {
						printf ("operation error!\n"); error = 1; break;
					}
					calc_v.a2 = value[value_pos--];
					calc_v.a1 = value[value_pos--];
					calc_op = op[op_pos--];

					switch (calc_op) {
						case '+': result = addition_1 (&calc_v, clnt); break;
						case '-': result = subtraction_1 (&calc_v, clnt); break;
						case '*': result = multiplication_1 (&calc_v, clnt); break;
						case '/': if (calc_v.a2 == 0) {
								  	printf ("you cannot divide by 0!\n");
									error = 1; break;
								  }
								  result = division_1 (&calc_v, clnt); break;
						case '^': result = power_1 (&calc_v, clnt); break;
					}
					value[++value_pos] = *result;
				}

				if (error == 1) break;
				op[++op_pos] = string[i]; //insert current op
			}
		}

		if (error == 1) continue; 

		while (op_pos != -1) {
			if (value_pos < 1) {
				printf ("operation error!\n"); error = 1; break;
			}
			calc_v.a2 = value[value_pos--];
			calc_v.a1 = value[value_pos--];
			calc_op = op[op_pos--];

			switch (calc_op) {
				case '+': result = addition_1 (&calc_v, clnt); break;
				case '-': result = subtraction_1 (&calc_v, clnt); break;
				case '*': result = multiplication_1 (&calc_v, clnt); break;
				case '/': if (calc_v.a2 == 0) {
							  printf ("you cannot divide by 0!\n");
							  error = 1; break;
						  }
						  result = division_1 (&calc_v, clnt); break;
				case '^': result = power_1 (&calc_v, clnt); break;
			}
			if (error == 1) break;
			value[++value_pos] = *result;
		}

		if (error == 1) continue; 

		printf ("The answer is %d\n", value[value_pos]);

	}
}

