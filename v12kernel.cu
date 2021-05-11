#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>
/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

//Simple 3D volume renderer.

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_


#include <math.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "v9volumeRender.h"


#define MINLEN		0.0001



typedef struct
{
	float4 m[3];
} float3x4;


__constant__ float3x4 c_invViewMatrix;  //inverse view matrix.


struct Ray
{
	float3 o;   //origin.
	float3 d;   //direction.
};


// intersect ray with a box.
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}


extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}



//add on 2020/3/30.
__device__ void clamp(float4 *sampleCol)
{
	
	if (sampleCol->x < 0.0f)
		sampleCol->x = 0.0f;
	else if (sampleCol->x > 1.0f)
		sampleCol->x = 1.0f;

	if (sampleCol->y < 0.0f)
		sampleCol->y = 0.0f;
	else if (sampleCol->y > 1.0f)
		sampleCol->y = 1.0f;

	if (sampleCol->z < 0.0f)
		sampleCol->z = 0.0f;
	else if (sampleCol->z > 1.0f)
		sampleCol->z = 1.0f;

	if (sampleCol->w < 0.0f)
		sampleCol->w = 0.0f;
	else if (sampleCol->w > 1.0f)
		sampleCol->w = 1.0f;
}
//add on 2020/3/30.



//launch winWidth * winHeight threads.
//volume ray casting algorithm.
__global__ void	d_render(int winWidth, int winHeight, 
	cudaExtent volumeSize, int maxVolumeDim, float3 voxelSize,
	float3 lightPos, //bool contextOpen,
	float4 bgColor, float4 contextColor, float4 featColor,
	float *dev_mergeData, float *dev_normGrad, float *dev_normNormalX, float *dev_normNormalY, float *dev_normNormalZ,
	uint *d_output)
{
	//获得thread(x, y) id.
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//range: [0, winWidth - 1].
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//range: [0, winHeight - 1].
	

	if ((x < winWidth) && (y < winHeight))
	{
		const int maxSteps = 500;	//max number of samples along a ray.
		const float tstep = 0.01f;	//sampling distance between 2 samples along a ray.
		const float opacityThreshold = 0.95f;
		//both boxMin and boxMax ensure the final image displays in correct dimension along 3 directions.
		const float3 boxMin = make_float3(-1.0f * volumeSize.width * voxelSize.x / maxVolumeDim,
										-1.0f * volumeSize.height * voxelSize.y / maxVolumeDim,
										-1.0f * volumeSize.depth * voxelSize.z / maxVolumeDim);
		const float3 boxMax = make_float3(1.0f * volumeSize.width * voxelSize.x / maxVolumeDim,
										1.0f * volumeSize.height * voxelSize.y / maxVolumeDim,
										1.0f * volumeSize.depth * voxelSize.z / maxVolumeDim);
		
		
		//对于d_ouput上每个pixel(x, y)来说：
		//将d_output上的pixel(x, y)映射为空间中范围为[-1.0, 1.0]的pixel(u, v).
		float u = (x / (float)(winWidth - 1)) * 2.0f - 1.0f;	//u range: [-1.0, 1.0].
		float v = (y / (float)(winHeight - 1)) * 2.0f - 1.0f;	//v range: [-1.0, 1.0].


		//calculate eye ray in world space.
		Ray eyeRay;
		//计算eyeRay origin: eyeRay.o = (0, 0, 4).
		eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		//计算eyeRay direction.
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(c_invViewMatrix, eyeRay.d);


		//find intersection with box.
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
		if (hit)	//如果eyeRay和box相交.
		{
			if (tnear < 0.0f) tnear = 0.0f;     //clamp to near plane.

			//march along ray from front to back, accumulating color.
			float4 color_xy = bgColor;
			float t = tnear;
			float3 pos = eyeRay.o + eyeRay.d * tnear;	//pos range: [-0.5, -1, -1] - [0.5, 1, 1].
			float3 step = eyeRay.d * tstep;


			for (int i = 0; i < maxSteps; i++)
			{
				//for each sample point:
				//1. compute sample point's position samplePos(x, y, z).
				int3 samplePos;
				samplePos.x = roundf((pos.x * maxVolumeDim / (volumeSize.width * voxelSize.x) * 0.5f + 0.5f) * (volumeSize.width - 1)); //range: [0 - 127].
				samplePos.y = roundf((pos.y * maxVolumeDim / (volumeSize.height * voxelSize.y) * 0.5f + 0.5f) * (volumeSize.height - 1));	//range: [0, 255].
				samplePos.z = roundf((pos.z * maxVolumeDim / (volumeSize.depth * voxelSize.z) * 0.5f + 0.5f) * (volumeSize.depth - 1));	//range: [0, 255].
				

				//2. at samplePos(x, y, z), 
				//(2.1)get its mergeData: [0-1, 2].
				float sample_mergeData = dev_mergeData[samplePos.x + samplePos.y * volumeSize.width +
					samplePos.z * volumeSize.width * volumeSize.height];

				//(2.2)get its normGrad.
				float sample_normGrad = dev_normGrad[samplePos.x + samplePos.y * volumeSize.width +
					samplePos.z * volumeSize.width * volumeSize.height];


				//(2.3)get its normNormalX/Y/Z.
				float3 sample_normNormal;
				sample_normNormal.x = dev_normNormalX[samplePos.x + samplePos.y * volumeSize.width +
					samplePos.z * volumeSize.width * volumeSize.height];
				sample_normNormal.y = dev_normNormalY[samplePos.x + samplePos.y * volumeSize.width +
					samplePos.z * volumeSize.width * volumeSize.height];
				sample_normNormal.z = dev_normNormalZ[samplePos.x + samplePos.y * volumeSize.width +
					samplePos.z * volumeSize.width * volumeSize.height];
				

				//3. at samplePos(x, y, z), compute its sample color.
				float4 sampleCol = { 0.0f, 0.0f, 0.0f, 0.0f };
				//(3.1)if this sample = extracted feature, then set this sample's color = feature color.
				if (sample_mergeData == 2)
				{
					sampleCol = featColor;
					sampleCol.w = 100;
				}
				//(3.2)set this sample's color = contextColor.
				else  
				{	
					sampleCol = contextColor;	//bgColor;
					sampleCol.w = sample_mergeData * sample_normGrad * 0.1;	//0;
				}		
				


				//4. at samplePos(x, y, z), compute its sample color after lighting.
				//compute light direction for pos(x, y, z).
				float3 lightDir;
				lightDir.x = lightPos.x - pos.x;
				lightDir.y = lightPos.y - pos.y;
				lightDir.z = lightPos.z - pos.z;
				//normalize lightDir.
				float len_lightDir = sqrt(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z);
				if (len_lightDir < MINLEN)
				{
					lightDir.x = 0;
					lightDir.y = 0;
					lightDir.z = 0;
				}
				else
				{
					lightDir.x /= len_lightDir;
					lightDir.y /= len_lightDir;
					lightDir.z /= len_lightDir;
				}
				//compute diffuse lighting.
				float diffuseFactor = 1.5;
				float diffuse = sample_normNormal.x * lightDir.x + sample_normNormal.y * lightDir.y + sample_normNormal.z * lightDir.z;
				//add diffuse lighting to sampleCol.
				sampleCol.x += diffuse * diffuseFactor;
				sampleCol.y += diffuse * diffuseFactor;
				sampleCol.z += diffuse * diffuseFactor;


				//(4.3)clamp sampleCol to be [0, 1].
				clamp(&sampleCol);
				


				//5. at samplePos(x, y, z), accumulate sampleCol to be color_xy.
				//pre-multiply alpha.
				sampleCol.x *= sampleCol.w;
				sampleCol.y *= sampleCol.w;
				sampleCol.z *= sampleCol.w;

				//"over" operator for front-to-back blending.
				//color_xy = color_xy + sampleCol * (1.0f - color_xy.w);
				color_xy = sampleCol + color_xy * (1.0f - sampleCol.w);		//refer to https://stackoverflow.com/questions/39555050/how-to-do-the-blending-in-volume-rendering-using-glsl
				
				
				//exit early if opaque.
				if (color_xy.w > opacityThreshold)
					break;


				t += tstep;

				if (t > tfar) break;

				pos += step;
			}


			//write each [x, y]'s output color to corresponding pixel location in pbo.
			d_output[x + y * winWidth] = rgbaFloatToInt(color_xy);

		}	//end if intersection.
		else  //if not intersection with data.
		{
			d_output[x + y * winWidth] = rgbaFloatToInt(bgColor);
		}
	
	}//end if.
}



extern "C" void render_kernel(int winWidth, int winHeight, dim3 gridSize, dim3 blockSize, 
	cudaExtent volumeSize, int maxVolumeDim, float3 voxelSize,
	float3 lightPos, //bool contextOpen,
	float4 bgColor, float4 contextColor, float4 featColor,
	float *dev_mergeData, float *dev_normGrad, float *dev_normNormalX, float *dev_normNormalY, float *dev_normNormalZ,
	uint *d_output)
{
		//render image (pointed by d_output).
		d_render << <gridSize, blockSize >> > (winWidth, winHeight, 
			volumeSize, maxVolumeDim, voxelSize,
			lightPos, //contextOpen,
			bgColor, contextColor, featColor,
			dev_mergeData, dev_normGrad, dev_normNormalX, dev_normNormalY, dev_normNormalZ,
			d_output);
}

#endif


