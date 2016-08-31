#include <cuda.h> 
#include <stdio.h>
#define POLYNOM_ORDER 4 

const float u = 1.0;
const float delta_t = 0.02; 
const float delta_x = 0.54;

//Calculate C, for now it's constant 

__device__ float calcC()

{

return u*(delta_t/delta_x); 

}

__device__ float calcCminus() 

{

return calcC();

}

__device__ float calcCplus()

{

return calcC(); 

}

__device__ float calcA1(const float* j) 

{

return ( 8*(*(j+1)) - (*(j+2)) - 8*(*(j-1)) + (*(j-2)))/12;

}

__device__ float calcA2(const float* j) 
{

return ( 16*(*(j+1)) - (*(j+2)) - 30*(*j) + 16*(*(j-1)) - (*(j-2)))/24; 

}

__device__ float calcA3(const float* j)

{

return ( -2*(*(j+1)) + (*(j+2)) + 2*(*(j-1)) - (*(j-2)))/12; 

}

__device__ float calcA4(const float* j) 

{

return ( -4*(*(j+1)) + (*(j+2)) + 6*(*j) - 4*(*(j-1)) + (*(j-2)))/24; 

}

__device__ float calcA(const float* j, const int k) 
{
	switch(k) 
		{
		case 0: 
			return *j;
		case 1:
			return calcA1(j);
		case 2:
			return calcA2(j);
		case 3:
			return calcA3(j);
		case 4:
			return calcA4(j);
		default:
			return *j; 
		}
}	
