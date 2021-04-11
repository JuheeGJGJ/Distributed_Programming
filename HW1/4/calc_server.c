#include "calc.h"

int *
addition_1_svc(two_int *argp, struct svc_req *rqstp)
{
	static int  result;
	result = (*argp).a1 + (*argp).a2;
	return &result;
}

int *
subtraction_1_svc(two_int *argp, struct svc_req *rqstp)
{
	static int  result;
	result = (*argp).a1 - (*argp).a2;
	return &result;
}

int *
multiplication_1_svc(two_int *argp, struct svc_req *rqstp)
{
	static int  result;
	result = (*argp).a1 * (*argp).a2;
	return &result;
}

int *
division_1_svc(two_int *argp, struct svc_req *rqstp)
{
	static int  result;
	result = (*argp).a1 / (*argp).a2;
	return &result;
}

int *
power_1_svc(two_int *argp, struct svc_req *rqstp)
{
	int i;
	static int  result = 1;

	for (i = 0 ; i < (*argp).a2 ; i++) {
		result *= (*argp).a1;
	}
	return &result;
}
