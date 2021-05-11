#ifndef VOLUMERENDER_H
#define VOLUMERENDER_H

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned char VolumeType;	//use this 'VolumeType' if loaded data is 1 byte (8 bits). It works now.
//typedef unsigned short VolumeType;	//2 bytes (16 bits). It does not work for now.
typedef unsigned int uint;

/*NOTE:make sure .raw's minmum value is 0.*/
//__device__ float rawDataMaxValue = 52992; //901.00;		//t2_tirm_cor_dark-fluid-fs - 5 max: [0, 901];  //ctneck_tumorcut.raw: [0-2519]

//variables in volumeRender.cpp


//variables in kernel.cu


#endif