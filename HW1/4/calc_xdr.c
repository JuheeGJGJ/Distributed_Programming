/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#include "calc.h"

bool_t
xdr_two_int (XDR *xdrs, two_int *objp)
{
	register int32_t *buf;

	 if (!xdr_int (xdrs, &objp->a1))
		 return FALSE;
	 if (!xdr_int (xdrs, &objp->a2))
		 return FALSE;
	return TRUE;
}
