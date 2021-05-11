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
#include "v12volumeRender.h"


using namespace std;


#define		WINDOWTITLEBARHEIGHT		9	//人工写的window title bar height in pixels.
#define		TOTALTIMESTEPS				50
#define		FILESTARTVAL				0
#define		FILEINCREMENT				1
#define		FIRSTTIMESTEP				1	//0 (at t = 0, normSimilarity_t has no data).
#define		MAX_EPSILON_ERROR			5.00f
#define		ONETHOUSANDMILLISEC			500
#define		TARTIMESTEP					1	//timestep when feature first appears.
#define		TARTIMESTEP_LAST			49	//timestep when feature disappears.
#define		GRAYVALUE_FEATURE			1
//#define NUMOFROWS_COLORMAP			64
//#define NUMOFCOLUMNS_COLORMAP			3


#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif


//global variables.
cudaExtent volumeSize = make_cudaExtent(128, 128, 128);	 
int winWidth = 512, winHeight = 512;	//APP. window's width & height.
dim3 blockSize(16, 16);	//blockSize.z = 1 by default.
dim3 gridSize;	//gridSize.x = 1; gridSize.y = 1; gridSize.z = 1.
dim3 bs(8, 8, 8);
dim3 gs;

float3 viewRotation = make_float3(-90, 0, 0);	//viewRotation.x = 0; viewRotation.y = 0; viewRotation.z = 0.
float3 viewTranslation = make_float3(0, 0, -2);	//(0.0, 0.0, -4.0f);
float invViewMatrix[12];
GLuint pbo = 0;     // OpenGL pixel buffer object, each pixel 4 bytes for RGBA.
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO).
GLuint pboFromFramebuffer = 0;	//OpenGL pixel buffer object, used to copy 'framebuffer' including user-drawn lasso.
GLuint texFromPBO = 0;     // OpenGL texture object, each pixel 4 bytes for RGBA.

//compute frames-per-second, according to http://www.lighthouse3d.com/tutorials/glut-tutorial/frames-per-second/.
int numOfFrames = 0;
int lastTime = 0;
int currentTime;
//find out the max. dimension of volume data.
int maxVolumeDim = 0;
float3 voxelSize = make_float3(1, 1, 1);
float3 lightPos = make_float3(0.0f, -1.0f, 4.0f);

int buttonState = 0;
int lx = 0, ly = 0;
int cx = 0, cy = 0;

//colors.
float4 pink = {1.0f, 0.0f, 1.0f, 1.0f};	//lassoColor: opaque pink.
float4 transWhite = { 1.0f, 1.0f, 1.0f, 0.0f };	//bgColor: transparent white.
float4 transBlue = { 0.0f, 0.0f, 1.0f, 0.0f };	//contextColor: transparent blue.
float4 transRed = { 1.0f, 0.0f, 0.0f, 0.0f };	//tarFeat1Color: transparent red.
float4 transGreen = {0.0f, 1.0f, 0.0f, 0.0f};	//tarFeat2Color: transparent green.

int timeStepCounter = FIRSTTIMESTEP;
char dataSource[200] = "D:/Research/Feature_Tracking_Tornado/Feature_Tracking_Tornado/v6_result/";
char fileName[200];
float *mergeData_t, *dev_mergeData_t;
float *normGrad_t, *dev_normGrad_t;
float *normNormalX_t, *dev_normNormalX_t;
float *normNormalY_t, *dev_normNormalY_t;
float *normNormalZ_t, *dev_normNormalZ_t;


extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void initializePixelBuffer();
extern "C" void render_kernel(int winWidth, int winHeight, dim3 gridSize, dim3 blockSize,
	cudaExtent volumeSize, int maxVolumeDim, float3 voxelSize,
	float3 lightPos, //bool contextOpen,
	float4 bgColor, float4 contextColor, float4 featColor,
	float *dev_mergeData, float *dev_normGrad, float *dev_normNormalX, float *dev_normNormalY, float *dev_normNormalZ,
	uint *d_output);




//refers to: https://lodev.org/cgtutor/floodfill.html.
void push(std::vector<int>& stack, int x, int y)
{
	// C++'s std::vector can act as a stack and manage memory for us
	stack.push_back(x);
	stack.push_back(y);
}


//refers to: https://lodev.org/cgtutor/floodfill.html.
bool pop(std::vector<int>& stack, int& x, int& y)
{
	if (stack.size() < 2) return false; // it's empty
	y = stack.back();
	stack.pop_back();
	x = stack.back();
	stack.pop_back();
	return true;
}


//refer to CUDA 'postProcessGL' example.
void pboRegister(int pbo)
{
	//register this buffer object with CUDA.
	cudaGLRegisterBufferObject(pbo);
}


//refer to CUDA 'postProcessGL' example.
void pboUnregister(int pbo)
{
	//unregister this buffer object with CUDA.
	cudaGLUnregisterBufferObject(pbo);
}



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


//render image using CUDA.
//in other words, use cuda kernel to generate image data, which is stored in pbo.
void render(int winWidth, int winHeight, dim3 gridSize, dim3 blockSize, 
cudaExtent volumeSize, int maxVolumeDim, float3 voxelSize,
float3 lightPos, //bool contextOpen,
float4 bgColor, float4 contextColor, float4 featColor,
float *dev_mergeData, float *dev_normGrad, float *dev_normNormalX, float *dev_normNormalY, float *dev_normNormalZ)
{
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	
	//clear image d_output to be white = 255 (not black = 0).
	checkCudaErrors(cudaMemset(d_output, 0, winWidth * winHeight * 4));
	
	//call CUDA kernel, writing results to PBO.
	// printf("ok\n");
	render_kernel(winWidth, winHeight, gridSize, blockSize, 
		volumeSize, maxVolumeDim, voxelSize,
		lightPos, //contextOpen,
		bgColor, contextColor, featColor,
		dev_mergeData, dev_normGrad, dev_normNormalX, dev_normNormalY, dev_normNormalZ,
		d_output);


	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}


void saveToRawFile(string rawFileName, void *data, char dataType, int size)
{
	//Note: 
	//(i)fopen第一个参数必须加(char *), 否则错误; 
	//(ii)fopen第二个参数必须为"wb"而非"w", 否则错误).
	FILE *fid = fopen((char *)(rawFileName).c_str(), "wb");

	//fail to open .raw file.
	if (fid == NULL)
	{
		printf("Fail to open %s file.\n", rawFileName.c_str());
		exit(1);	//terminate with error.
	}

	//successfully open .raw file.
	size_t numOfVoxels;
	switch (dataType)
	{
	case 'u':	//unsigned char.
		numOfVoxels = fwrite(data, sizeof(unsigned char), size, fid);
		break;
	case 'f':	//float.
		numOfVoxels = fwrite(data, sizeof(float), size, fid);
		break;
	}
	
	if (numOfVoxels != size)
	{
		printf("Writing error!\n");
		exit(1);	//terminate with error.
	}
	fclose(fid);
	printf("%s has been saved.\n\n", rawFileName.c_str());
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
	

	//write rendering to 'pbo', and copy 'pbo' to 'framebuffer'.
	//CUDA render into 'pbo', and 'pbo' content transfers to 'framebuffer'.
	render(winWidth, winHeight, gridSize, blockSize, 
		volumeSize, maxVolumeDim, voxelSize,
		lightPos, //contextOpen,
		transWhite, transBlue,	transRed,
		dev_mergeData_t, dev_normGrad_t, dev_normNormalX_t, dev_normNormalY_t, dev_normNormalZ_t);


	//display results.
	glColor4f(transWhite.x, transWhite.y, transWhite.z, transWhite.w);	//glColor3f(1.0f, 1.0f, 1.0f);	//white.
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
}




void idle()
{
	//refresh screen.
	glutPostRedisplay();
}



void mouse(int button, int state, int x, int y)
{
	lx = cx;
	ly = cy;

	cx = x;
	cy = y;

	
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN))
	{
		buttonState = 1;
		//points.clear();
	}
	else if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_UP))
	{
		buttonState = 2;
		glutPostRedisplay();
	}
	else if ((button == GLUT_MIDDLE_BUTTON) && (state == GLUT_DOWN))
	{
		buttonState = 3;
	}
	else if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN))
	{
		buttonState = 4;
	}
	else
	{
		buttonState = 0;
	}
	
}

void motion(int x, int y)
{
	lx = cx;
	ly = cy;
	
	cx = x;
	cy = y;

	//points.push_back(x);
	//points.push_back(y);

	
	float dx, dy;
	dx = (float)(cx - lx);
	dy = (float)(cy - ly);


	if (buttonState == 1)
	{
		//left = draw lasso.
		glutPostRedisplay();
	}
	else if (buttonState == 3)
	{
		//middle = translate.
		viewTranslation.x += dx / 100.0f;
		viewTranslation.y -= dy / 100.0f;
		glutPostRedisplay();
	}
	else if (buttonState == 4)
	{
		//right = rotate.
		viewRotation.x += dy / 5.0f;
		viewRotation.y += dx / 5.0f;
		glutPostRedisplay();
	}

}



//scroll mouse wheel, zoom in or out the camera.
void mouseWheel(int button, int dir, int x, int y)
{
	buttonState = 0;
	
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
	
	glutPostRedisplay();
}



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
	//free cpu + gpu memory that has not been freed before.

	

	//unregister 'pbo' to CUDA and delete it.
	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &texFromPBO);
	}


	//unregister 'pboFromFramebuffer' to CUDA and delete it.
	if (pboFromFramebuffer)
	{
		pboUnregister(pboFromFramebuffer);
		glDeleteBuffers(1, &pboFromFramebuffer);
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
	glutCreateWindow("animation");
	

	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
	{
		printf("Required OpenGL extensions missing.\n");
		exit(EXIT_SUCCESS);
	}
}



void initializePixelBuffer()
{
	//1. create a OpenGL 'pbo' (pixel buffer object) & register to CUDA, for writing rendering data.
	//(1.1)create a OpenGL 'pbo' for writing rendering data.
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, winWidth * winHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB); 
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//(1.2)register the OpenGL 'pbo' pixel buffer object with CUDA, 
	//so that both CUDA & OpenGL could share these 'pbo'.
	//(Note:'cudaGraphicsMapFlagsWriteDiscard' is to specify
	//that the previous contents in this 'pbo' will be discarded, 
	//making this 'pbo' essentially write-only).
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsNone));	//cudaGraphicsMapFlagsWriteDiscard));	


	//2. create a texture, which will copy 'pbo' content & be used for render to the screen.
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
		//exit(1);	//terminate with error.
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



void animation(int value)
{
	glutTimerFunc(ONETHOUSANDMILLISEC, animation, -1);	//call animation() in ONETHOUSANDMILLISEC.
	
	//at current timestep t:
	printf("t: %d.\n", timeStepCounter);


	//1. at t, load .raw data, and copy them to gpu.
	//(1.1)read mergeData_t: [0-1, 2].
	sprintf(fileName, "%smergeData_%d.raw", dataSource, FILESTARTVAL + timeStepCounter * FILEINCREMENT);
	mergeData_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(1.2)read normGrad.
	sprintf(fileName, "%snormGrad_%d.raw", dataSource, FILESTARTVAL + timeStepCounter * FILEINCREMENT);
	normGrad_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(1.3)read normNormalX.
	sprintf(fileName, "%snormNormalX_%d.raw", dataSource, FILESTARTVAL + timeStepCounter * FILEINCREMENT);
	normNormalX_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(1.4)read normNormalY.
	sprintf(fileName, "%snormNormalY_%d.raw", dataSource, FILESTARTVAL + timeStepCounter * FILEINCREMENT);
	normNormalY_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(1.5)read normNormalZ.
	sprintf(fileName, "%snormNormalZ_%d.raw", dataSource, FILESTARTVAL + timeStepCounter * FILEINCREMENT);
	normNormalZ_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);


	//copy mergeData_t to gpu.
	//cudaMalloc((void **)&dev_mergeData_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_mergeData_t, mergeData_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);

	//copy normGrad_t to gpu.
	//cudaMalloc((void **)&dev_normGrad_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normGrad_t, normGrad_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);

	//copy normNormalX_t to gpu.
	//cudaMalloc((void **)&dev_normNormalX_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalX_t, normNormalX_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//copy normNormalY_t to gpu.
	//cudaMalloc((void **)&dev_normNormalY_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalY_t, normNormalY_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//copy normNormalZ_t to gpu.
	//cudaMalloc((void **)&dev_normNormalZ_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalZ_t, normNormalZ_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//free .raw data (which are useless when copied to gpu).
	free(mergeData_t);
	free(normGrad_t);
	free(normNormalX_t);
	free(normNormalY_t);
	free(normNormalZ_t);


	//2. at t, refresh screen.
	glutPostRedisplay();


	//3. increment t, and loop to play animation.
	timeStepCounter++;
	if (timeStepCounter > (TARTIMESTEP_LAST))
	{
		timeStepCounter = FIRSTTIMESTEP;
	}

}


//used to merge normInten and feature into one data. In particular,
//normInten: [0, 1];
//feature = 2.
void mergeData()
{
	for (int t = FIRSTTIMESTEP; t < TOTALTIMESTEPS; t++)
	{
		//for each timestep t:
		//1. read normInten_t: [0, 1].
		sprintf(fileName, "%snormInten_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
		float *normInten_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

		/*
		//(1.1)this is special edge process to display squared cylinder only (used for squared cylinder data only).
		for (int k = 0; k < volumeSize.depth; k++)
			for (int i = 0; i < volumeSize.height; i++)
				for (int j = 0; j < volumeSize.width; j++)
				{
					if ((i <= 1) || (i >= (volumeSize.height - 1)) || (k == (volumeSize.depth - 1)))
					{
						normInten_t[i * volumeSize.width + j + k * volumeSize.width * volumeSize.height] = 0.000001;
					}
				}
		*/


		if ((t >= TARTIMESTEP) && (t <= TOTALTIMESTEPS -1))
		{
			// 这里49这个时间步的mergeData不知为何没有计算出来
			//2. read extracted feature.
			sprintf(fileName, "%sTornado_%d_classificationFiled.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
			float *feat_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);


			//3. merge normInten and extracted feature.
			for (int k = 0; k < volumeSize.depth; k++)
				for (int i = 0; i < volumeSize.height; i++)
					for (int j = 0; j < volumeSize.width; j++)
					{
						//for each voxel v(i, j, k):
						//if (feat_t[i * volumeSize.width + j + k * volumeSize.width * volumeSize.height] == GRAYVALUE_FEATURE)
						if (feat_t[i * volumeSize.width + j + k * volumeSize.width * volumeSize.height] != 0)
						{
							normInten_t[i * volumeSize.width + j + k * volumeSize.width * volumeSize.height] = 2;
						}
					}
		}



		//4. save mergeData as .raw file.
		sprintf(fileName, "%smergeData_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
		saveToRawFile(fileName, normInten_t, 'f', volumeSize.width * volumeSize.height * volumeSize.depth);
	
	}	//end t.

}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{ 
	int t = FIRSTTIMESTEP;

	
	//1. (optional. use it if not used yet) merge normInten and extracted feature data into one data. In particular,
	//normInten: [0, 1];
	//feature = 2.
	mergeData();

	
	
	//2. initialize OpenGL context, so we can properly set the OpenGL for CUDA.
	//This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	//These GLUT calls need to be made before other OpenGL calls.
	initGL(&argc, argv);


	//3. at t = FIRSTTIMESTEP, load its mergeData_t: [0-1, 2], normGrad, normNormalX/Y/Z.
	//(3.1)read mergeData_t.
	sprintf(fileName, "%smergeData_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
	mergeData_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(3.2)read normGrad.
	sprintf(fileName, "%snormGrad_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
	normGrad_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(3.3)read normNormalX.
	sprintf(fileName, "%snormNormalX_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
	normNormalX_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(3.4)read normNormalY.
	sprintf(fileName, "%snormNormalY_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
	normNormalY_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//(3.5)read normNormalZ.
	sprintf(fileName, "%snormNormalZ_%d.raw", dataSource, FILESTARTVAL + t * FILEINCREMENT);
	normNormalZ_t = (float *)loadRawFile(fileName, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);

	//copy mergeData_t to gpu.
	cudaMalloc((void **)&dev_mergeData_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_mergeData_t, mergeData_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);

	//copy normGrad_t to gpu.
	cudaMalloc((void **)&dev_normGrad_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normGrad_t, normGrad_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);

	//copy normNormalX_t to gpu.
	cudaMalloc((void **)&dev_normNormalX_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalX_t, normNormalX_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//copy normNormalY_t to gpu.
	cudaMalloc((void **)&dev_normNormalY_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalY_t, normNormalY_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//copy normNormalZ_t to gpu.
	cudaMalloc((void **)&dev_normNormalZ_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth);
	cudaMemcpy(dev_normNormalZ_t, normNormalZ_t, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth,
		cudaMemcpyHostToDevice);


	//free .raw data (which are useless when copied to gpu).
	free(mergeData_t);
	free(normGrad_t);
	free(normNormalX_t);
	free(normNormalY_t);
	free(normNormalZ_t);
	
	
	//4. initPixelBuffer() did 3 things: 
	//(1)OpenGL allocates an empty pbo (pointed by 'pbo'), & registered this pbo to cuda;
	//so that cuda kernel can write/create rendering image;
	//(2)OpenGL allocated an empty texture (pointed by 'tex'), with width * height (each pixel = RGBA);
	//and we copy 'pbo' image to this texture, which displays final rendering.
	//(3)OpenGL allocates an empty 'pboFromFramebuffer'(it copies framebuffer image including lasso to it), 
	//so that we can (in parallel) generate lassoBackgroundImg/dev_lassoBackgroundImg(去除该功能).
	initializePixelBuffer();


	//5. given blockSize_win/blockSize_vol,
	//(5.1)calculate APP. window's gridSize.
	gridSize = dim3(iDivUp(winWidth, blockSize.x), iDivUp(winHeight, blockSize.y));
	//(5.2)calculate volume data's gridSize.
	gs = dim3(iDivUp(volumeSize.width, bs.x), iDivUp(volumeSize.height, bs.y), iDivUp(volumeSize.depth, bs.z));
	//(5.3)find out volume data's max dimension.
	maxVolumeDim = MAX(volumeSize.width, volumeSize.height);
	maxVolumeDim = MAX(maxVolumeDim, volumeSize.depth);
	printf("maxVolumeDim = %d.\n", maxVolumeDim);


	//6. This is the normal rendering path for VolumeRender. In particular, 
	//glutDisplayFunc(display) will call the CUDA kernel.
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	//glutKeyboardFunc(keyboard);
	glutMouseWheelFunc(mouseWheel);
	//glutIdleFunc(idle);	//remove on 2019/4/13.
	glutTimerFunc(ONETHOUSANDMILLISEC, animation, -1);
	glutCloseFunc(cleanup);
	glutMainLoop();
}
