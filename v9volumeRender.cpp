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

/*
Volume rendering sample

This sample loads a 3D volume from disk and displays it using
ray marching and 3D textures.

Note - this is intended to be an example of using 3D textures
in CUDA, not an optimized volume renderer.

Changes
sgg 22/3/2010
- updated to use texture for display instead of glDrawPixels.
- changed to render from front-to-back rather than back-to-front.
*/

//add on 2019/4/17.
//(1)v9 version adds lighting.
//add on 2019/4/17.

// OpenGL Graphics APPLE
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include "v9volumeRender.h"

using namespace std;


#define MAX_EPSILON_ERROR		5.00f
#define THRESHOLD				0.30f

//#define NUMOFROWS_COLORMAP		64
//#define NUMOFCOLUMNS_COLORMAP	3

#define TOTANIFRAMES			16		//20	
#define ONETHOUSANDMILLISEC		1000


//NOTE: in 'make_cudaExtent',
//1st parameter is rows;
//2nd parameter is columns;
//3rd parameter is depth;
//e.g., make_cudaExtent(width, height, depth).
cudaExtent volumeSize = make_cudaExtent(128, 256, 256);	 
int winWidth = 512, winHeight = 512;	//APP. window's width & height
dim3 blockSize(16, 16);	//blockSize.z = 1 by default.
dim3 gridSize;	//gridSize.x = 1; gridSize.y = 1; gridSize.z = 1.


float3 viewRotation;	//viewRotation.x = 0; viewRotation.y = 0; viewRotation.z = 0.
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];


GLuint pbo = 0;     // OpenGL pixel buffer object, each pixel 4 bytes for RGBA.
GLuint texFromPBO = 0;     // OpenGL texture object, each pixel 4 bytes for RGBA.
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)


int *d_aniMatrix;
int theta = 1;

//compute frames-per-second, according to http://www.lighthouse3d.com/tutorials/glut-tutorial/frames-per-second/.
int numOfFrames = 0;
int lastTime = 0;
int currentTime;
//find out the max. dimension of volume data.
int maxVolumeDim = 0;

//add on 2019/4/17.
float3 lightPos = make_float3(0.0f, -1.0f, 4.0f);
//add on 2019/4/17.

int ox, oy;
int buttonState = 0;

//add on 2020/3/30.
bool contextOpen = true;
bool drawLasso = false;
//add on 2020/3/30.


#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void initializePixelBuffer();
extern "C" void initializeCuda(cudaExtent volumeSize,
	void *normInten, void *normGrad, void *probVolume,
	void *normNormalX, void *normNormalY, void *normNormalZ);
extern "C" void render_kernel(int winWidth, int winHeight, dim3 blockSize, dim3 gridSize, cudaExtent volumeSize, int maxVolumeDim,
	int totAniFrames, int *d_aniMatrix, int theta, 
	float3 lightPos, bool contextOpen, bool drawLasso, int ox, int oy,
	uint *d_output);



//compute frames-per-second, according to http://www.lighthouse3d.com/tutorials/glut-tutorial/frames-per-second/.
void calculateFPS()
{
	//increase the number of frames.
	numOfFrames++;

	//get the elapsed time since the call to glutInit(in milliseconds).
	currentTime = glutGet(GLUT_ELAPSED_TIME);
	if (currentTime - lastTime >= ONETHOUSANDMILLISEC)
	{
		//如果已过去1秒钟.
		//(1)计算1秒钟里的numOfFrames数量.
		float fps = (float)(numOfFrames * ONETHOUSANDMILLISEC) / (currentTime - lastTime);

		//(2)lastTime = currrentTime.
		lastTime = currentTime;

		//(3)set numOfFrames to be 0 again.
		numOfFrames = 0;

		//(4)display & update fps as window title.
		char fpsTitle[256];
		sprintf(fpsTitle, "Vis. = %4.2f fps.", fps);
		glutSetWindowTitle(fpsTitle);
	}
}



// render image using CUDA
// in other words, use cuda kernel to generate image data, which is stored in pbo.
void render(int winWidth, int winHeight, dim3 blockSize, dim3 gridSize, cudaExtent volumeSize, int maxVolumeDim, 
int totAniFrames, int *d_aniMatrix, int theta,
float3 lightPos, bool contextOpen, bool drawLasso, int ox, int oy)
{
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	//printf("width: %d, height: %d, CUDA mapped PBO: May access %ld bytes\n", width, height, num_bytes);

	//changed on 2020/3/30.
	//clear image d_output to be white = 255 (not black = 0).
	checkCudaErrors(cudaMemset(d_output, 255, winWidth * winHeight * 4));
	//changed on 2020/3/30.


	//call CUDA kernel, writing results to PBO.
	render_kernel(winWidth, winHeight, blockSize, gridSize, volumeSize, maxVolumeDim, totAniFrames, d_aniMatrix, theta, 
		lightPos, contextOpen, drawLasso, ox, oy, d_output);


	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}


//display results using OpenGL (called by GLUT).
void display()
{
	//use OpenGL to build view matrix.
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
	glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	render(winWidth, winHeight, blockSize, gridSize, volumeSize, maxVolumeDim, 
		TOTANIFRAMES, d_aniMatrix, theta, lightPos, contextOpen, drawLasso, ox, oy);

	//display results.
	glClear(GL_COLOR_BUFFER_BIT);

	//draw image from PBO.
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
	// draw using glDrawPixels (slower)
	glRasterPos2i(0, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glDrawPixels(winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
	//draw using texture.
	//copy from pbo to texture.
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, texFromPBO);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//draw textured quad.
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	glTexCoord2f(1, 0);
	glVertex2f(1, 0);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(0, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
	
#endif
	glutSwapBuffers();
	glutReportErrors();

	
	//计算frames-per-second.
	calculateFPS();
	

	//printf("theta = %d.\n", theta);		//this print can be removed for speeding up.
	/*
	theta++;
	if (theta > TOTANIFRAMES)
	{
		theta = 1;
	}
	*/
}


//add on 2020/3/30.
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 32: //space key.
		contextOpen = !contextOpen;
		//refresh screen.
		glutPostRedisplay();
		break;
	}
	
}
//add on 2020/3/30.


void idle()
{
	//refresh screen.
	glutPostRedisplay();
}


void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
		//add on 2020/4/1.
		drawLasso = false;
		printf("false.\n");
		//add on 2020/4/1.
	}

	ox = x;
	oy = y;
	//glutPostRedisplay();
}


void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 4)
	{
		//right = rotate.
		viewRotation.x += dy / 5.0f;
		viewRotation.y += dx / 5.0f;
		glutPostRedisplay();
	}
	else if (buttonState == 2)
	{
		//middle = translate.
		viewTranslation.x += dx / 100.0f;
		viewTranslation.y -= dy / 100.0f;
		glutPostRedisplay();
	}
	else if (buttonState == 1)
	{
		//left = draw lasso.
		drawLasso = true;
		glutPostRedisplay();
	}
	

	ox = x;
	oy = y;
}


//add on 2020/4/1.
//scroll mouse wheel, zoom in or out the camera.
void mouseWheel(int button, int dir, int x, int y)
{
	float dy = 0;
	if (dir == -1)
	{
		//zoom in.
		dy += 10;
		
	}
	else if (dir == 1)
	{
		//zoom out.
		dy -= 10;
	}
	viewTranslation.z += dy / 100.0f;
	
	ox = x;
	oy = y;
	glutPostRedisplay();
}
//add on 2020/4/1.


int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


void reshape(int w, int h)
{
	winWidth = w;
	winHeight = h;
	initializePixelBuffer();
	
	// calculate new grid size
	gridSize = dim3(iDivUp(winWidth, blockSize.x), iDivUp(winHeight, blockSize.y));

	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
	freeCudaBuffers();

	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &texFromPBO);
	}
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}


void initGL(int *argc, char **argv)
{
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);	//(GLUT_RGB | GLUT_DOUBLE); //2016/8/4 change from GLUT_RGB to GLUT_RGBA
	glutInitWindowSize(winWidth, winHeight);
	glutCreateWindow("CUDA volume rendering");
	

	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
	{
		printf("Required OpenGL extensions missing.\n");
		exit(EXIT_SUCCESS);
	}
}



void initializePixelBuffer()
{
	//1. if 'pbo'(pixel buffer object) has been existed, delete these old 'pbo' & old 'tex'.
	if (pbo)
	{
		//unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		//delete this old 'pbo' buffer.
		glDeleteBuffersARB(1, &pbo);
		//delete this old texture 'tex'.
		glDeleteTextures(1, &texFromPBO);
	}


	//2. if 'pbo' is not created, then create a OpenGL 'pbo' for writing rendering data.
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, winWidth * winHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB); 
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);


	//3. register this OpenGL 'pbo'buffer object with CUDA, so that both CUDA & OpenGL could share this 'pbo'.
	//'cudaGraphicsMapFlagsWriteDiscard' is to specify
	//that the previous contents in this 'pbo' will be discarded, 
	//making this 'pbo' essentially write-only.
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));	


	//4. create texture, which will copy 'pbo' content & be used for render to the screen.
	glGenTextures(1, &texFromPBO);
	glBindTexture(GL_TEXTURE_2D, texFromPBO);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}



//Load raw data from disk.
void *loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");

	//fail to open this .raw file.
	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		exit(1);	//terminate with error.
	}

	//open this .raw file successfully.
	void *data = malloc(size);
	//printf("size1: %d\n", size);
	size_t read = fread(data, 1, size, fp);
	//size_t read = fread(data, 2, size/2, fp);
	fclose(fp);

	
	printf("Read '%s', %lu bytes\n", filename, read);

	return data;
}



void timerFunc(int value)
{
	//refresh screen.
	glutPostRedisplay();

	glutTimerFunc(ONETHOUSANDMILLISEC, timerFunc, 0);
}




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//1. select a CUDA device (whose compute capability is 1.0 or better), 
	//on which to run both CUDA & OpenGL.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);
	//given CUDA device id 'dev', using this device for both CUDA & OpenGL.
	cudaGLSetGLDevice(dev);



	//2. initialize OpenGL context, so we can properly set the OpenGL for CUDA.
	//This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//These GLUT calls need to be made before other OpenGL calls.
	initGL(&argc, argv);


	/*
	//add on 2019/4/13.
	//3. generate aniMatrix, and copy it to dev_aniMatrix.	
	//(3.1)初始化1个ani2DMatrix.
	int ani2DMatrix[TOTANIFRAMES][TOTANIFRAMES + 1] = { { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第1行.
	{ 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第2行.
	{ 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第3行.
	{ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第4行.
	{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第5行.
	{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第6行.
	{ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第7行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 },	//第8行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 },	//第9行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 },	//第10行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 },	//第11行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 },	//第12行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },	//第13行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 },	//第14行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 },	//第15行.
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }	//第16行.
	};

	//(3.2)reshape ani2DMatrix to be 1D aniMatrix.
	int *aniMatrix;
	aniMatrix = (int *)malloc(sizeof(int)* TOTANIFRAMES * (TOTANIFRAMES + 1));
	for (int i = 0; i < TOTANIFRAMES; i++)
		for (int j = 0; j < (TOTANIFRAMES + 1); j++)
		{
			aniMatrix[i * (TOTANIFRAMES + 1) + j] = ani2DMatrix[i][j];
		}


	//(3.3)copy aniMatrix to d_aniMatrix.
	cudaMalloc((void **)&d_aniMatrix, sizeof(int) * TOTANIFRAMES * (TOTANIFRAMES + 1));
	cudaMemcpy(d_aniMatrix, aniMatrix, sizeof(int) * TOTANIFRAMES * (TOTANIFRAMES + 1), cudaMemcpyHostToDevice);
	*/
	

	//3. load .raw volume data to cpu memory + reshape colormap.
	string dataPath = "D:/research/codes/GMM-rule/VisMale/";
	string colormapPath = "C:/FangCloudV2/Personal files/codes/matlab/saveMatlabColormap/";

	//(3.1)load normalizedIntensity.raw.
	void *normInten = loadRawFile(const_cast<char *>((dataPath + "normInten.raw").c_str()), sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	//(3.2)load normaliazedGrad.raw.
	void *normGrad = loadRawFile(const_cast<char *>((dataPath + "normGrad.raw").c_str()), sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	//(3.3)load probVolume.raw.
	void *probVolume = loadRawFile(const_cast<char *>((dataPath + "probVolume_nsel2D=1_NUMOFCTNGEN=10_NUMOFTOTGEN=40_FITNESSCONV=0.010000_MD=0.000261(lowUpperThres)(percent).raw").c_str()),	//((dataPath + "segResultEval/goldStandardMask_32bits.raw").c_str()), 
		sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	
	/*
	//(3.4)load colormap.raw.
	void *colormap_temp = loadRawFile(const_cast<char *>((colormapPath + "colormap_jet.raw").c_str()), sizeof(float)* NUMOFCOLUMNS_COLORMAP * NUMOFROWS_COLORMAP);
	
	//(3.5)reshape colormap_temp to be 1D(float4 * 64) matrix colormap.
	float4 *colormap;	//float4 colormap_float4[COLORMAP_NUMOFROWS];
	colormap = (float4 *)malloc(sizeof(float4)* NUMOFROWS_COLORMAP);
	for (int i = 0; i < NUMOFROWS_COLORMAP; i++)
	{
		colormap[i].x = ((float *)colormap_temp)[i];
		colormap[i].y = ((float *)colormap_temp)[i + 1 * NUMOFROWS_COLORMAP];
		colormap[i].z = ((float *)colormap_temp)[i + 2 * NUMOFROWS_COLORMAP];
		//colormap_float4.w代表opacity.
		colormap[i].w = 0.0f;

		//test.
		//printf("colormap[%d].x = %f, colormap[%d].y = %f, colormap[%d].z = %f, colormap[%d].w = %f.\n",
		//	i, colormap[i].x, i, colormap[i].y, i, colormap[i].z, i, colormap[i].w);
		//test.
	}
	*/
	
	//add on 2019/4/17.
	//(3.6)load normNormalX + normNormalY + normNormalZ .raw files.
	void *normNormalX = loadRawFile(const_cast<char *>((dataPath + "normNormalX.raw").c_str()),
		sizeof(float)* volumeSize.width * volumeSize.height * volumeSize.depth);
	void *normNormalY = loadRawFile(const_cast<char *>((dataPath + "normNormalY.raw").c_str()),
		sizeof(float)* volumeSize.width * volumeSize.height * volumeSize.depth);
	void *normNormalZ = loadRawFile(const_cast<char *>((dataPath + "normNormalZ.raw").c_str()),
		sizeof(float)* volumeSize.width * volumeSize.height * volumeSize.depth);
	//add on 2019/4/17.


	//4. 
	//(4.1)initializeCuda did 3 things:
	//(4.1.1)copy normInten/normGrad/probVolume .raw from cpu (pointed by 'normInten') 
	//to gpu (pointed by 'd_normInten'), and bind the .raw with a 3D texture (pointed by 'tex3D_normInten'). 
	//As a result, we can use tex3D(tex3D_normInten, x, y, z) to obtain .raw volume data's (x, y, z) voxel.
	//(4.1.2)copy colormap .raw data from cpu (pointed by 'colormap') to gpu (pointed by 'd_colormap'),
	//and bind the colormap data with a 1D texture (pointed by 'tex1D_colormap').
	//As a result, we can use tex1D(tex1D_colormap, x) to obtain the x-indexed RGBA color.
	//(4.1.3)copy normNormalX + normNormalY + normNormalZ .raw from cpu
	//to gpu (pointed by 'd_normalsX/Y/Z'), and bind the .raw with a 3D texture (pointed by 'tex3D_normNormalX/Y/Z'). 
	//As a result, we can use tex3D(tex3D_normNormalX/Y/Z, x, y, z) to obtain .raw volume data's (x, y, z) voxel.
	//NOTE: in this step, 
	//(i)texture <VolumeType, 3, cudaReadModeNormalizedFloat> tex makes range of the data on texture to be [0.0, 1.0]
	//(refer to http://blog.csdn.net/yanghangjun/article/details/5587269);
	//(ii) tex.normalized = true makes the texture coordinates to be [0.0, 1.0].		
	initializeCuda(volumeSize, 
		normInten, normGrad, probVolume,
		normNormalX, normNormalY, normNormalZ);
	

	//(4.2)free cpu memory.
	//(4.2.1)free normalizedIntensity.raw volume data on cpu memory, as it has been copied to gpu memory.
	free(normInten);
	//(4.2.2)free normalizedGrad.raw volume data on cpu memory, as it has been copied to gpu memory. 
	free(normGrad);
	//(4.2.3)free probVolume.raw volume data on cpu memory, as it has been copied to gpu memory.
	free(probVolume);
	//(4.2.4)free colormap.raw on cpu memory, as it has been copied to gpu memory.
	//free(colormap);
	//free(aniMatrix);
	free(normNormalX);
	free(normNormalY);
	free(normNormalZ);
	//2019/4/10: 到此正确。


	//5. initPixelBuffer() did 2 things: 
	//(1)OpenGL allocates an empty pbo (pointed by 'pbo'), & registered this pbo to cuda;
	//so that cuda kernel can write/create rendering image;
	//(2)OpenGL allocated an empty texture (pointed by 'tex'), with width * height (each pixel = RGBA);
	//and we copy 'pbo' image to this texture, which displays final rendering.
	initializePixelBuffer();


	//given blockSize = (16, 16), calculate new grid size.
	gridSize = dim3(iDivUp(winWidth, blockSize.x), iDivUp(winHeight, blockSize.y));

	
	//find out the max dimension of volume data.
	maxVolumeDim = MAX(volumeSize.width, volumeSize.height);
	maxVolumeDim = MAX(maxVolumeDim, volumeSize.depth);
	printf("maxVolumeDim = %d.\n", maxVolumeDim);


	//6. This is the normal rendering path for VolumeRender. In particular, 
	//glutDisplayFunc(display) will call the CUDA kernel.
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	
	//glutIdleFunc(idle);	//remove on 2019/4/13.
	//add on 2019/4/13.
	//glutTimerFunc(ONETHOUSANDMILLISEC, timerFunc, 0);
	//add on 2019/4/13.

	//add on 2020/3/30.
	glutKeyboardFunc(keyboard);
	glutMouseWheelFunc(mouseWheel);
	//add on 2020/3/30.

	glutCloseFunc(cleanup);

	//printf("VolumeType: %d\n", sizeof(VolumeType));
	glutMainLoop();
}
