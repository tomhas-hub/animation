
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

//add on 2019/4/19
#define MINLEN		0.0001
//add on 2019/4/19


//these pointers point to gpu�ϵ�.raw data copied from cpu.  
//cudaArray *d_aniMatrix = 0;
cudaArray *d_normInten = 0;
cudaArray *d_normGrad = 0;
cudaArray *d_probVolume = 0;
//cudaArray *d_colormap = 0;

//add on 2019/4/17.
cudaArray *d_normNormalX = 0;
cudaArray *d_normNormalY = 0;
cudaArray *d_normNormalZ = 0;
//add on 2019/4/17.


//NOTE:
//(1)changed 'cudaReadModeNormalizedFloat' to 'cudaReadModeElementType', 
//so that when use tex3D() to extract texture value, it returns VolumeType data whose range is the same as .raw;
//(2)'tex3DRawData.filterMode = cudaFilterModeLinear' can only use for returned value being float-point, not for VolumeType;
//(3)'cudaReadModeNormalizedFloat' makes .raw values (in 3D texture) are normalized to [0.0, 1.0]. Refer to http://blog.csdn.net/yanghangjun/article/details/5587269.
texture<float, 3, cudaReadModeElementType>		tex3D_normInten;	//range: [0, 1].
texture<float, 3, cudaReadModeElementType>		tex3D_normGrad;	//range: [0, 1].
texture<float, 3, cudaReadModeElementType>		tex3D_probVolume;	//range: [0, n].
texture<float4, 1, cudaReadModeElementType>		tex1D_colormap;
texture<int, 2, cudaReadModeElementType>		tex2D_aniMatrix;

//add on 2019/4/17.
texture<float, 3, cudaReadModeElementType>		tex3D_normNormalX;		//range: [0, 1].
texture<float, 3, cudaReadModeElementType>		tex3D_normNormalY;		//range: [0, 1].
texture<float, 3, cudaReadModeElementType>		tex3D_normNormalZ;		//range: [0, 1].
//add on 2019/4/17.

//add on 2020/3/30/
__device__ float4 bgColor = {1.0f, 1.0f, 1.0f, 0.0f};	//transparent white.
__device__ float4 contextColor = { 0.0f, 0.0f, 1.0f, 0.0f};	//transparent blue.
__device__ float4 tarFeatColor = { 1.0f, 0.0f, 0.0f, 0.0f};	//transparent red.
//add on 2020/3/30



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





extern "C" void freeCudaBuffers()
{
	checkCudaErrors(cudaFreeArray(d_normInten));
	checkCudaErrors(cudaFreeArray(d_normGrad));
	checkCudaErrors(cudaFreeArray(d_probVolume));
	//checkCudaErrors(cudaFreeArray(d_colormap));
}




extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


extern "C" void initializeCuda(cudaExtent volumeSize, 
	void *normInten, void *normGrad, void *probVolume,
	void *normNormalX, void *normNormalY, void *normNormalZ)
{
	//����channel������.
	cudaChannelFormatDesc float_ChannelDesc = cudaCreateChannelDesc<float>();
	//cudaChannelFormatDesc float4_ChannelDesc = cudaCreateChannelDesc<float4>();
	

	//����cudaMemcpy3DParms.
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	
	
	//1.1 create a 3D array on gpu, pointed by 'd_normInten'.
	cudaMalloc3DArray(&d_normInten, &float_ChannelDesc, volumeSize);

	//1.2 copy cpu .raw data (pointed by 'normalizedIntensity') to this gpu 3D array (pointed by 'd_normalizedIntensity').
	copyParams.srcPtr = make_cudaPitchedPtr(normInten, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_normInten;
	cudaMemcpy3D(&copyParams);

	//1.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_normInten.normalized = true;	//access with normalized texture coordinates [0.0, 1.0].
	tex3D_normInten.filterMode = cudaFilterModeLinear;	//linear interpolation can only use with float-point.
	tex3D_normInten.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_normInten.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_normInten, d_normInten, float_ChannelDesc);



	//2.1 create a 3D array on gpu, pointed by 'd_normalizedGrad'.
	cudaMalloc3DArray(&d_normGrad, &float_ChannelDesc, volumeSize);

	//2.2 copy cpu .raw data (pointed by 'normalizedGrad') to this gpu 3D array (pointed by 'd_normalizedGrad').
	//cudaMemcpy3DParms is CUDA 3D memory copying parameters.
	copyParams.srcPtr = make_cudaPitchedPtr(normGrad, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_normGrad;
	cudaMemcpy3D(&copyParams);

	//2.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_normGrad.normalized = true;	//access with normalized texture coordinates [0.0,1.0].
	tex3D_normGrad.filterMode = cudaFilterModeLinear;	//linear interpolation can only use with float-point.
	tex3D_normGrad.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_normGrad.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_normGrad, d_normGrad, float_ChannelDesc);



	//3.1 create a 3D array on gpu, pointed by 'd_probVolume'.
	cudaMalloc3DArray(&d_probVolume, &float_ChannelDesc, volumeSize);

	//3.2 copy cpu .raw data (pointed by 'probVolume') to this gpu 3D array (pointed by 'd_probVolume').
	//cudaMemcpy3DParms is CUDA 3D memory copying parameters.
	copyParams.srcPtr = make_cudaPitchedPtr(probVolume, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);	//copyParams_resultOfSelected3DComponent.srcPtr = make_cudaPitchedPtr(resultOfSelected3DComponent, volumeSize.width * sizeof(unsigned char), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_probVolume;
	cudaMemcpy3D(&copyParams);

	//3.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_probVolume.normalized = true;	//access with normalized texture coordinates [0.0, 1.0].
	tex3D_probVolume.filterMode = cudaFilterModeLinear;	//linear interpolation can only use with float-point.	//tex3D_resultOfSelected3DComponent.filterMode = cudaFilterModePoint;
	tex3D_probVolume.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_probVolume.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_probVolume, d_probVolume, float_ChannelDesc);


	/*
	//4.1 create a 1D array on gpu, pointed by 'd_colormap'.
	cudaMallocArray(&d_colormap, &float4_ChannelDesc, numOfRows_colormap, 1);

	//4.2 copy cpu .raw colormap (pointed by 'colormap') to this gpu 1D array.
	cudaMemcpyToArray(d_colormap, 0, 0, colormap, sizeof(float4) * numOfRows_colormap, cudaMemcpyHostToDevice);

	//4.3 (1)set texture parameters for a gpu 1D texture;
	//(2)bind the 1D texture with above 1D colormap array, so that we can use tex1D(transferTex, x) to obtain the x-indexed RGBA color.
	tex1D_colormap.normalized = true;
	tex1D_colormap.filterMode = cudaFilterModeLinear;
	tex1D_colormap.addressMode[0] = cudaAddressModeClamp;
	//bind above 1D colormap array to this 1D texture.
	cudaBindTextureToArray(tex1D_colormap, d_colormap, float4_ChannelDesc);
	*/



	//add on 2019/4/17.
	//5.1 create a 3D array on gpu, pointed by 'd_normNormalX'.
	cudaMalloc3DArray(&d_normNormalX, &float_ChannelDesc, volumeSize);

	//5.2 copy cpu .raw data (pointed by 'normalizedGrad') to this gpu 3D array (pointed by 'd_normalizedGrad').
	//cudaMemcpy3DParms is CUDA 3D memory copying parameters.
	copyParams.srcPtr = make_cudaPitchedPtr(normNormalX, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_normNormalX;
	cudaMemcpy3D(&copyParams);

	//5.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_normNormalX.normalized = true;	//access with normalized texture coordinates [0.0,1.0].
	tex3D_normNormalX.filterMode = cudaFilterModeLinear;	//linear interpolation can only use with float-point.
	tex3D_normNormalX.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_normNormalX.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_normNormalX, d_normNormalX, float_ChannelDesc);



	//6.1 create a 3D array on gpu, pointed by 'd_normNormalY'.
	cudaMalloc3DArray(&d_normNormalY, &float_ChannelDesc, volumeSize);

	//6.2 copy cpu .raw data (pointed by 'normalizedGrad') to this gpu 3D array (pointed by 'd_normalizedGrad').
	//cudaMemcpy3DParms is CUDA 3D memory copying parameters.
	copyParams.srcPtr = make_cudaPitchedPtr(normNormalY, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_normNormalY;
	cudaMemcpy3D(&copyParams);

	//6.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_normNormalY.normalized = true;	//access with normalized texture coordinates [0.0,1.0].
	tex3D_normNormalY.filterMode = cudaFilterModeLinear;	// linear interpolation can only use with float-point.
	tex3D_normNormalY.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_normNormalY.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_normNormalY, d_normNormalY, float_ChannelDesc);



	//7.1 create a 3D array on gpu, pointed by 'd_normNormalZ'.
	cudaMalloc3DArray(&d_normNormalZ, &float_ChannelDesc, volumeSize);

	//7.2 copy cpu .raw data (pointed by 'normalizedGrad') to this gpu 3D array (pointed by 'd_normalizedGrad').
	//cudaMemcpy3DParms is CUDA 3D memory copying parameters.
	copyParams.srcPtr = make_cudaPitchedPtr(normNormalZ, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_normNormalZ;
	cudaMemcpy3D(&copyParams);

	//7.3 (1)set texture parameters for a gpu 3D texture;
	//(2)bind this 3D texture with above 3D array, so that we can use tex3D(tex, x, y, z) to obtain .raw's (x, y, z) voxel.
	tex3D_normNormalZ.normalized = true;	//access with normalized texture coordinates [0.0,1.0].
	tex3D_normNormalZ.filterMode = cudaFilterModeLinear;	// linear interpolation can only use with float-point.
	tex3D_normNormalZ.addressMode[0] = cudaAddressModeClamp;	//clamp texture coordinates
	tex3D_normNormalZ.addressMode[1] = cudaAddressModeClamp;
	//bind above 3D array to this gpu 3D texture.
	cudaBindTextureToArray(tex3D_normNormalZ, d_normNormalZ, float_ChannelDesc);
	//add on 2019/4/17.
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
__global__ void	d_render(int winWidth, int winHeight, cudaExtent volumeSize, int maxVolumeDim, 
int totAniFrames, int *d_aniMatrix, int theta, float3 lightPos, bool contextOpen, bool drawLasso, int ox, int oy,
uint *d_output)
{
	//���thread(x, y).
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//range: [0, winWidth - 1].
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//range: [0, winHeight - 1].
	

	if ((x < winWidth) && (y < winHeight))
	{
		const int maxSteps = 500;	//max number of samples along a ray.
		const float tstep = 0.01f;	//sampling distance between 2 samples along a ray.
		const float opacityThreshold = 0.95f;
		//both boxMin and boxMax ensure the final image displays in correct dimension along 3 directions.
		const float3 boxMin = make_float3(-1.0f * volumeSize.width / maxVolumeDim,
			-1.0f * volumeSize.height / maxVolumeDim,
			-1.0f * volumeSize.depth / maxVolumeDim);
		const float3 boxMax = make_float3(1.0f * volumeSize.width / maxVolumeDim,
			1.0f * volumeSize.height / maxVolumeDim,
			1.0f * volumeSize.depth / maxVolumeDim);
		
		
		//����d_ouput��ÿ��pixel(x, y)��˵��
		//��d_output�ϵ�pixel(x, y)ӳ��Ϊ�ռ��з�ΧΪ[-1.0, 1.0]��(u, v).
		float u = (x / (float)winWidth) * 2.0f - 1.0f;	//u range: [-1.0, 1.0].
		float v = (y / (float)winHeight) * 2.0f - 1.0f;	//v range: [-1.0, 1.0].

		//calculate eye ray in world space.
		Ray eyeRay;
		//����eyeRay origin: eyeRay.o = (0, 0, 4).
		eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		//����eyeRay direction.
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(c_invViewMatrix, eyeRay.d);


		//find intersection with box.
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
		if (hit)	//���eyeRay��box�ཻ.
		{
			if (tnear < 0.0f) tnear = 0.0f;     //clamp to near plane.

			//march along ray from front to back, accumulating color.
			float4 color_xy = bgColor;
			float t = tnear;
			float3 pos = eyeRay.o + eyeRay.d * tnear;	//pos range: [-1, -1, -1] - [1, 1, 1].
			float3 step = eyeRay.d * tstep;


			for (int i = 0; i < maxSteps; i++)
			{
				//(1)���pos(x, y, z) sample point normInten + normGrad + probVolume + numOfEnhancedFrames + x/y/z normals.
				//sample_normInten range: [0, 1].
				float sample_normInten = tex3D(tex3D_normInten,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);
				//sample_normGrad range: [0, 1].
				float sample_normGrad = tex3D(tex3D_normGrad,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);
				//sample_probVolume range: [0, 1].
				float sample_probVolume = tex3D(tex3D_probVolume,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);
				//sample_numOfEnhancedFrames range: [0, 16].
				//int sample_numOfEnhancedFrames = round(sample_probVolume * totAniFrames);	//ceil(sample_probVolume * totAniFrames);

				//sample_normNormal.x range: [0, 1];
				//sample_normNormal.y range: [0, 1];
				//sample_normNormal.z range : [0, 1].
				float3 sample_normNormal;
				sample_normNormal.x = tex3D(tex3D_normNormalX,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);
				sample_normNormal.y = tex3D(tex3D_normNormalY,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);
				sample_normNormal.z = tex3D(tex3D_normNormalZ,
						pos.x * maxVolumeDim / volumeSize.width * 0.5f + 0.5f,
						pos.y * maxVolumeDim / volumeSize.height * 0.5f + 0.5f,
						pos.z * maxVolumeDim / volumeSize.depth * 0.5f + 0.5f);


				//(2)���pos(x, y, z) sample point RGBA color sampleCol, according to contextOpen (true or false).
				//(2.1)���pos(x, y, z) sample point RGBA sampleCol without shading.
				float opacityScaleFactor = 1.5;
				float enhFactor = 900;
				float4 sampleCol = { 0.0f, 0.0f, 0.0f, 0.0f };
				switch (contextOpen)
				{
				case true: //��ʾboth context + tarFeat.
					if (sample_probVolume == 0)	//˵����context.
					{
						sampleCol = contextColor;
						sampleCol.w = sample_normInten * sample_normGrad * opacityScaleFactor;
					}
					else if (sample_probVolume > 0)	//˵����tarfet feature.
					{
						sampleCol = tarFeatColor;
						sampleCol.w = sample_normInten * sample_normGrad * opacityScaleFactor;
						sampleCol.w *= (1.0f + log(sample_probVolume + 1) * enhFactor);
					}
					break;
				case false: //ֻ��ʾtarFeat.
					if (sample_probVolume > 0)
					{
						sampleCol = tarFeatColor;
						sampleCol.w = sample_normInten * sample_normGrad * opacityScaleFactor;
						sampleCol.w *= (1.0f + log(sample_probVolume + 1) * enhFactor);
					}
					break;
				}


				//(2.2)add shading to pos(x, y, z) sample point RGBA sampleCol.
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
				float diffuseFactor = 10;
				float diffuse = sample_normNormal.x * lightDir.x + sample_normNormal.y * lightDir.y + sample_normNormal.z * lightDir.z;
				//add diffuse lighting to sampleCol.
				sampleCol.x += diffuse * diffuseFactor;
				sampleCol.y += diffuse * diffuseFactor;
				sampleCol.z += diffuse * diffuseFactor;


				//(2.3)clamp sampleCol to be [0, 1].
				clamp(&sampleCol);


				/*
				//(3)���pos(x, y, z) sample point enhanced RGB(optional) + oapcity.
				float enhFactor = 9000000;	//30;
				float4 sampleCol_enhanced = sampleCol_default;

				//(3.1)����1(rule-enhanced paper������): ����pos(x, y, z) sample point probVolume, ��ø�sample point enhanced opacity Oe(v).
				sampleCol_enhanced.w = sampleCol_default.w * (1.0f + log(sample_probVolume + 1) * enhFactor);
				//end ����1.
				*/

				/*
				//(3.2)����2(�Լ�����, �뷽��1��ѡ��һ): ����pos(x, y, z) sample point probVolume, ��ø�sample point new RGB + enhanced opacity Oe(v).
				//(3.2.1)���pos(x, y, z) sample point new RGB.
				if (sample_probVolume > 0)
				{
				//specify to be red.
				sampleCol_enhanced.x = 1.0f;
				sampleCol_enhanced.y = 0.0f;
				sampleCol_enhanced.z = 0.0f;
				}
				//(3.2.2)���pos(x, y, z) sample point enhanced opacity Oe(v).
				sampleCol_enhanced.w = sampleCol_default.w * (1 + log(sample_probVolume + 1) * enhFactor);
				//end ����2.
				*/

				/*
				//(3.3)clamp pos(x, y, z) sample point enhanced opacity Oe(v) to be [0, 1].
				if (sampleCol_enhanced.w < 0.0f)
				sampleCol_enhanced.w = 0.0f;
				else if (sampleCol_enhanced.w > 1.0f)
				sampleCol_enhanced.w = 1.0f;
				*/

				/*
				//add on 2019/4/13.
				//(4)���pos(x, y, z) sample point removed opacity.
				float4 sampleCol_removed = sampleCol_default;


				//(4.1)���ݸ�sample point sample_probVolume, ����sampleCol.removed.w.
				//���sampleprobVolumeԽ��, ��sampleCol.removed.wԽС;
				//���sample_probVolumeԽС, ��sampleCol.removed.wԽ��.
				//(Note: (a)remFactor = 0, sample_removed.w = sample_default.w (����ԭ������target feature��ʾ������, ����remFactor = 0);
				//(b)remFactorԽ��, �Ƴ�����Խ��).
				float remFactor = 0;	//90000000;
				sampleCol_removed.w = sampleCol_default.w * (1.0f - log(sample_probVolume + 1) * remFactor);
				//end����2.



				//(4.2)clamp sampleCol_removed.w to be [0, 1].
				if (sampleCol_removed.w < 0.0f)
				sampleCol_removed.w = 0.0f;
				else if (sampleCol_removed.w > 1.0f)
				sampleCol_removed.w = 1.0f;



				//(5)����d_aniMatrix(theta - 1, sample_numOfEnhancedFrames), ȷ���ڸ�theta֡��sampleCol_thetaFrame��ɫ.
				float4 sampleCol_thetaFrame;
				int enhancedOrRemovedValue = d_aniMatrix[(theta - 1) * (totAniFrames + 1) + sample_numOfEnhancedFrames];
				if (enhancedOrRemovedValue == 1)
				{
				//˵���ڸ�theta֡sampleColor_thetaFrame = sampleColor_enhanced.
				sampleCol_thetaFrame = sampleCol_enhanced;
				}
				else if (enhancedOrRemovedValue == 0)
				{
				//˵���ڸ�theta֡sampleColor_thetaFrame = sampleColor_removed.
				sampleCol_thetaFrame = sampleCol_removed;
				}
				//add on 2019/4/13.
				*/


				//(6)accumulate pos(x, y, z) sample point sampleCol to be color_xy.
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
		}//end if intersection.

	}//end if.
}



extern "C" void render_kernel(int winWidth, int winHeight, dim3 blockSize, dim3 gridSize, cudaExtent volumeSize, int maxVolumeDim, 
int totAniFrames, int *d_aniMatrix, int theta, 
float3 lightPos, bool contextOpen, bool drawLasso, int ox, int oy,
uint *d_output)
{
	//render image (pointed by d_output).
	d_render << <gridSize, blockSize >> > (winWidth, winHeight, volumeSize, maxVolumeDim, totAniFrames, d_aniMatrix, theta, lightPos, 
		contextOpen, drawLasso, ox, oy, d_output);
}

#endif


