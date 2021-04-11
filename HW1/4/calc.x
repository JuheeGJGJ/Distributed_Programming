struct two_int {
	int a1;
	int a2;
};
program CALC_PROG {
	version CALC_VERS {
		int addition (two_int) = 1;	
		int subtraction (two_int) = 2;	
		int multiplication (two_int) = 3;	
		int division (two_int) = 4;	
		int power (two_int) = 5;	
	} = 1;
} = 0x31111111;
