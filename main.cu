#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include <cuda.h> 
#include <cufft.h>

#define FLDBL	double
const FLDBL PI=2.0*asin(1.);
const int BLOCK_X=16;
const int BLOCK_Y=16;
__global__ void streamingKernel(FLDBL *utemp,FLDBL *vtemp,FLDBL*un,FLDBL*vn,FLDBL*uup,FLDBL*udown,FLDBL*uleft,FLDBL*uright,FLDBL*vup,FLDBL*vdown,FLDBL*vleft,FLDBL*vright,FLDBL dx,FLDBL dy,FLDBL dt,int x, int y,FLDBL nu)
{
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;
	
	int idx = tx+bx*blockDim.x;
	int jdx = ty+by*blockDim.y;
	if ((idx!=0)&&(idx!=x-1)&&(jdx!=0)&&(jdx!=y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-un[idx+x*(jdx+1)])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
	else if ((idx==0)&&(jdx!=0)&&(jdx!=y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(uleft[jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-un[idx+x*(jdx+1)])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+uleft[jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vleft[jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vleft[jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
	else if ((idx==x-1)&&(jdx!=0)&&(jdx!=y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-uright[jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-un[idx+x*(jdx+1)])/2/dy+nu*( (uright[jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vright[jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vright[jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
	else if ((idx!=0)&&(idx!=x-1)&&(jdx==0))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(uup[idx]-un[idx+x*(jdx+1)])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+uup[idx])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vup[idx]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vup[idx])/dy/dy ));
	}
	else if ((idx!=0)&&(idx!=x-1)&&(jdx==y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-udown[idx])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (udown[idx]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vdown[idx])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vdown[idx]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
	else if ((idx==0)&&(jdx==0))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(uleft[jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(uup[idx]-un[idx+x*(jdx+1)])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+uleft[jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+uup[idx])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vleft[jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vup[idx]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vleft[jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vup[idx])/dy/dy ));
	}
	else if ((idx==0)&&(jdx==y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(uleft[jdx]-un[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-udown[idx])/2/dy+nu*( (un[idx+1+x*jdx]-2*un[idx+x*jdx]+uleft[jdx])/dx/dx + (udown[idx]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vleft[jdx]-vn[idx+1+x*jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vdown[idx])/2/dy+nu*( (vn[idx+1+x*jdx]-2*vn[idx+x*jdx]+vleft[jdx])/dx/dx + (vdown[idx]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
	else if ((idx==x-1)&&(jdx==0))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-uright[jdx])/2/dx+vn[idx+x*jdx]*(uup[idx]-un[idx+x*(jdx+1)])/2/dy+nu*( (uright[jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (un[idx+x*(jdx+1)]-2*un[idx+x*jdx]+uup[idx])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vright[jdx])/2/dx+vn[idx+x*jdx]*(vup[idx]-vn[idx+x*(jdx+1)])/2/dy+nu*( (vright[jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vn[idx+x*(jdx+1)]-2*vn[idx+x*jdx]+vup[idx])/dy/dy ));
	}
	else if ((idx==x-1)&&(jdx==y-1))
	{
		utemp[idx+x*jdx]=un[idx+x*jdx]+dt*(un[idx+x*jdx]*(un[idx-1+x*jdx]-uright[jdx])/2/dx+vn[idx+x*jdx]*(un[idx+x*(jdx-1)]-udown[idx])/2/dy+nu*( (uright[jdx]-2*un[idx+x*jdx]+un[idx-1+x*jdx])/dx/dx + (udown[idx]-2*un[idx+x*jdx]+un[idx+x*(jdx-1)])/dy/dy ));
		vtemp[idx+x*jdx]=vn[idx+x*jdx]+dt*(un[idx+x*jdx]*(vn[idx-1+x*jdx]-vright[jdx])/2/dx+vn[idx+x*jdx]*(vn[idx+x*(jdx-1)]-vdown[idx])/2/dy+nu*( (vright[jdx]-2*vn[idx+x*jdx]+vn[idx-1+x*jdx])/dx/dx + (vdown[idx]-2*vn[idx+x*jdx]+vn[idx+x*(jdx-1)])/dy/dy ));
	}
}
__global__ void lineKernel(FLDBL *utemp,FLDBL *vtemp,FLDBL *f,int x, int y,FLDBL dx,FLDBL dy)
{
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;
	
	int idx = tx+bx*blockDim.x;
	int jdx = ty+by*blockDim.y;

	if ((idx!=0)&&(idx!=x-1)&&(jdx!=0)&&(jdx!=y-1))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(jdx-1)])/2/dy;
	else if ((idx==0)&&(jdx!=0)&&(jdx!=y-1))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[x-1+x*jdx])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(jdx-1)])/2/dy;
	else if ((idx==x-1)&&(jdx!=0)&&(jdx!=y-1))
		f[jdx+idx*y]=(utemp[x*jdx]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(jdx-1)])/2/dy;
	else if ((idx!=0)&&(idx!=x-1)&&(jdx==0))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(y-1)])/2/dy;
	else if ((idx!=0)&&(idx!=x-1)&&(jdx==y-1))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx]-vtemp[idx+x*(jdx-1)])/2/dy;
	else if ((idx==0)&&(jdx==0))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[x-1+x*jdx])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(y-1)])/2/dy;
	else if ((idx==0)&&(jdx==y-1))
		f[jdx+idx*y]=(utemp[idx+1+jdx*x]-utemp[x-1+x*jdx])/2/dx+(vtemp[idx]-vtemp[idx+x*(jdx-1)])/2/dy;
	else if ((idx==x-1)&&(jdx==0))
		f[jdx+idx*y]=(utemp[x*jdx]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx+x*(jdx+1)]-vtemp[idx+x*(y-1)])/2/dy;
	else if ((idx==x-1)&&(jdx==y-1))
		f[jdx+idx*y]=(utemp[x*jdx]-utemp[idx-1+jdx*x])/2/dx+(vtemp[idx]-vtemp[idx+x*(jdx-1)])/2/dy;
}
__global__ void fourieKernel(cufftDoubleComplex * data, FLDBL dx, FLDBL dy, int x, int y)
{
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;
	
	int idx = tx+bx*blockDim.x+1;
	int jdx = ty+by*blockDim.y+1;

	data[jdx+(y/2+1)*idx].x=data[jdx+(y/2+1)*idx].x/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
	if (idx!=x/2)
		data[jdx+(y/2+1)*(x-idx)].x=data[jdx+(y/2+1)*(x-idx)].x/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
	data[(y/2+1)*idx].x=data[(y/2+1)*idx].x/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx);
	if (idx!=x/2)
		data[(y/2+1)*(x-idx)].x=data[(y/2+1)*(x-idx)].x/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx);
	data[jdx].x=data[jdx].x/(-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
	data[jdx+(y/2+1)*idx].y=data[jdx+(y/2+1)*idx].y/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
	if (idx!=x/2)
		data[jdx+(y/2+1)*(x-idx)].y=data[jdx+(y/2+1)*(x-idx)].y/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
	data[(y/2+1)*idx].y=data[(y/2+1)*idx].y/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx);
	if (idx!=x/2)
		data[(y/2+1)*(x-idx)].y=data[(y/2+1)*(x-idx)].y/(-4*sin(dx*idx/2)*sin(dx*idx/2)/dx/dx);
	data[jdx].y=data[jdx].y/(-4*sin(dy*jdx/2)*sin(dy*jdx/2)/dy/dy);
}
__global__ void summaryKernel(FLDBL * p,FLDBL *utemp,FLDBL *vtemp,FLDBL*un,FLDBL*vn,FLDBL*uup,FLDBL*udown,FLDBL*uleft,FLDBL*uright,FLDBL*vup,FLDBL*vdown,FLDBL*vleft,FLDBL*vright,int x, int y, FLDBL dx, FLDBL dy)
{
	int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;
	
	int idx = tx+bx*blockDim.x+1;
	int jdx = ty+by*blockDim.y+1;
	if ((idx<x-1)&&(jdx<y-1))
	{
	un[idx+jdx*x]=utemp[idx+jdx*x]-(p[idx+1+x*jdx]-p[idx-1+x*jdx])/2/dx;

	un[idx]=utemp[idx]-(p[idx+1]-p[idx-1])/2/dx;
	un[idx+x*(y-1)]=utemp[idx+x*(y-1)]-(p[idx+1+x*(y-1)]-p[idx-1+x*(y-1)])/2/dx;
	un[jdx*x]=utemp[jdx*x]-(p[1+x*jdx]-p[x-1+x*jdx])/2/dx;
	un[x-1+jdx*x]=utemp[x-1+jdx*x]-(p[x*jdx]-p[x-2+x*jdx])/2/dx;

	vn[idx+jdx*x]=vtemp[idx+jdx*x]-(p[idx+x*(jdx+1)]-p[idx+x*(jdx-1)])/2/dy;

	vn[idx]=vtemp[idx]-(p[idx+x]-p[idx+x*(y-1)])/2/dy;
	vn[idx+x*(y-1)]=vtemp[idx+x*(y-1)]-(p[idx]-p[idx+x*(y-2)])/dy/2;
	vn[jdx*x]=vtemp[jdx*x]-(p[x*(jdx+1)]-p[x*(jdx-1)])/2/dy;
	vn[x-1+jdx*x]=vtemp[x-1+jdx*x]-(p[x-1+x*(jdx+1)]-p[x-1+x*(jdx-1)])/2/dy;

	un[0+x*0]=utemp[0+x*0]-(p[1]-p[x-1])/dx/2;
	un[x-1+x*0]=utemp[x-1+x*0]-(p[0]-p[x-2])/dx/2;
	un[0+x*(y-1)]=utemp[0+x*(y-1)]-(p[1+x*(y-1)]-p[x-1+x*(y-1)])/dx/2;
	un[x-1+x*(y-1)]=utemp[x-1+x*(y-1)]-(p[x*(y-1)]-p[x-2+x*(y-1)])/dx/2;

	vn[0+x*0]=vtemp[0+x*0]-(p[x]-p[x*(y-1)])/dy/2;
	vn[x-1+x*0]=vtemp[x-1+x*0]-(p[x-1+x]-p[x-1+x*(y-1)])/dy/2;
	vn[0+x*(y-1)]=vtemp[0+x*(y-1)]-(p[0]-p[x*(y-2)])/dy/2;
	vn[x-1+x*(y-1)]=vtemp[x-1+x*(y-1)]-(p[x-1+0]-p[x-1+x*(y-2)])/dy/2;

	uup[idx]=utemp[idx+x*(y-1)]-(p[idx+1+x*(y-1)]-p[idx-1+x*(y-1)])/2/dx;
	udown[idx]=utemp[idx]-(p[idx+1]-p[idx-1])/2/dx;
	uleft[jdx]=utemp[x-1+jdx*x]-(p[x*jdx]-p[x-2+x*jdx])/2/dx;
	uright[jdx]=utemp[jdx*x]-(p[1+x*jdx]-p[x-1+x*jdx])/2/dx;

	vup[idx]=vtemp[idx+x*(y-1)]-(p[idx]-p[idx+x*(y-2)])/dy/2;
	vdown[idx]=vtemp[idx]-(p[idx+x]-p[idx+x*(y-1)])/2/dy;
	vleft[jdx]=vtemp[x-1+jdx*x]-(p[x-1+x*(jdx+1)]-p[x-1+x*(jdx-1)])/2/dy;
	vright[jdx]=vtemp[jdx*x]-(p[x*(jdx+1)]-p[x*(jdx-1)])/2/dy;

	udown[0]=utemp[0+x*0]-(p[1]-p[x-1])/dx/2;
	uright[0]=utemp[0+x*0]-(p[1]-p[x-1])/dx/2;
	uleft[0]=utemp[x-1+x*0]-(p[0]-p[x-2])/dx/2;
	udown[x-1]=utemp[x-1+x*0]-(p[0]-p[x-2])/dx/2;
	uup[0]=utemp[0+x*(y-1)]-(p[1+x*(y-1)]-p[x-1+x*(y-1)])/dx/2;
	uright[y-1]=utemp[0+x*(y-1)]-(p[1+x*(y-1)]-p[x-1+x*(y-1)])/dx/2;
	uup[x-1]=utemp[x-1+x*(y-1)]-(p[x*(y-1)]-p[x-2+x*(y-1)])/dx/2;
	uleft[y-1]=utemp[x-1+x*(y-1)]-(p[x*(y-1)]-p[x-2+x*(y-1)])/dx/2;

	vright[0]=vtemp[0+x*0]-(p[x]-p[x*(y-1)])/dy/2;
	vdown[0]=vtemp[0+x*0]-(p[x]-p[x*(y-1)])/dy/2;
	vleft[0]=vtemp[x-1+x*0]-(p[x-1+x]-p[x-1+x*(y-1)])/dy/2;
	vdown[x-1]=vtemp[x-1+x*0]-(p[x-1+x]-p[x-1+x*(y-1)])/dy/2;
	vright[y-1]=vtemp[0+x*(y-1)]-(p[0]-p[x*(y-2)])/dy/2;
	vup[0]=vtemp[0+x*(y-1)]-(p[0]-p[x*(y-2)])/dy/2;
	vup[y-1]=vtemp[x-1+x*(y-1)]-(p[x-1+0]-p[x-1+x*(y-2)])/dy/2;
	vleft[x-1]=vtemp[x-1+x*(y-1)]-(p[x-1+0]-p[x-1+x*(y-2)])/dy/2;
	}
}


	//__shared__ FLDBL us[BLOCK_X][BLOCK_Y];
	//__shared__ FLDBL uns[BLOCK_X][BLOCK_Y];
	//__shared__ FLDBL vs[BLOCK_X][BLOCK_Y];
	//__shared__ FLDBL vns[BLOCK_X][BLOCK_Y];

	//for(i=1;i<x+1;i++)
	//	for(j=1;j<y+1;j++)
	//	{
	//		un[i+(x+2)*j]=u[i+(x+2)*j]+dt*(u[i+(x+2)*j]*(u[i-1+(x+2)*j]-u[i+1+(x+2)*j])/2/dx+v[i+(x+2)*j]*(u[i+(x+2)*(j-1)]-u[i+(x+2)*(j+1)])/2/dy+nu*( (u[i+1+(x+2)*j]-2*u[i+(x+2)*j]+u[i-2+(x+2)*j])/dx/dx + (u[i+(x+2)*(j+1)]-2*u[i+(x+2)*j]+u[i+(x+2)*(j-1)])/dy/dy ));
	//		vn[i+(x+2)*j]=v[i+(x+2)*j]+dt*(u[i+(x+2)*j]*(v[i-1+(x+2)*j]-v[i+1+(x+2)*j])/2/dx+v[i+(x+2)*j]*(v[i+(x+2)*(j-1)]-v[i+(x+2)*(j+1)])/2/dy+nu*( (v[i+1+(x+2)*j]-2*v[i+(x+2)*j]+v[i-2+(x+2)*j])/dx/dx + (v[i+(x+2)*(j+1)]-2*v[i+(x+2)*j]+v[i+(x+2)*(j-1)])/dy/dy ));
	//}
	//for(i=1;i<x+1;i++)
	//{
	//	un[i]=un[i+(x+2)*y];
	//	un[i+(x+2)*(y+1)]=un[i+x+2];
	//	vn[i]=vn[i+(x+2)*y];
	//	vn[i+(x+2)*(y+1)]=vn[i+x+2];
	//}
	//for(j=0;j<y+2;j++)
	//{
	//	un[(x+2)*j]=un[x+(x+2)*j];
	//	un[x+1+(x+2)*j]=un[1+(x+2)*j];
	//	vn[(x+2)*j]=vn[x+(x+2)*j];
	//	vn[x+1+(x+2)*j]=vn[1+(x+2)*j];
	//}
	////********FOURIE********


	////********RESULT********
	//for (i=1;i<x+1;i++)
	//	for(j=1;j<y+1;j++)
	//	{
	//		u[i+(x+2)*j]=un[i+(x+2)*j];
	//		v[i+(x+2)*j]=vn[i+(x+2)*j];
	//	}
	//for (int i=1;i<x+1;i++)
	//{
	//	u[i+(x+2)*0]=u[i+(x+2)*y];
	//	u[i+(x+2)*(y+1)]=u[i+(x+2)*1];
	//	v[i+(x+2)*0]=v[i+(x+2)*y];
	//	v[i+(x+2)*(y+1)]=v[i+(x+2)*1];
	//}
	//for (int j=0;j<y+2;j++)
	//{
	//	u[0+(x+2)*j]=u[x+(x+2)*j];
	//	u[x+1+(x+2)*j]=u[1+(x+2)*j];
	//	v[0+(x+2)*j]=v[x+(x+2)*j];
	//	v[x+1+(x+2)*j]=v[1+(x+2)*j];
	//}

	//us[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx];
	//vs[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx];

	//__syncthreads();
	//if ((tx>0)&&(tx<blockDim.x)&&(ty>0)&&(ty<blockDim.y))
	//{
	//	uns[tx][ty]=us[tx][ty]+dt*( us[tx][ty]*(us[tx-1][ty]-us[tx+1][ty])/2/dx+vs[tx][ty]*(us[tx][ty-1]-us[tx][ty+1])/2/dy+nu*( (us[tx+1][ty]-2.*us[tx][ty]+us[tx-1][ty])/dx/dx+(us[tx][ty+1]-2.*us[tx][ty]+us[tx][ty-1])/dy/dy ) );
	//	vns[tx][ty]=vs[tx][ty]+dt*( us[tx][ty]*(vs[tx-1][ty]-vs[tx+1][ty])/2/dx+vs[tx][ty]*(vs[tx][ty-1]-vs[tx][ty+1])/2/dy+nu*( (vs[tx+1][ty]-2.*vs[tx][ty]+vs[tx-1][ty])/dx/dx+(vs[tx][ty+1]-2.*vs[tx][ty]+vs[tx][ty-1])/dy/dy ) );
	//}
	//else if (!(((bx==0)&&(tx==0))||((bx==x/blockDim.x)&&(tx==blockDim.x))||((by==0)&&(ty==0))||((by==y/blockDim.y)&&(ty==blockDim.y))))
	//{
	//	uns[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//	vns[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//}
	//else if ((bx==0)&&(tx==0))
	//{
	//	uns[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(uleft[ty+by*blockDim.y]-un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+uleft[ty+by*blockDim.y])/dx/dx+(un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//	vns[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vleft[ty+by*blockDim.y]-vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vleft[ty+by*blockDim.y])/dx/dx+(vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//}
	//else if ((bx==x/blockDim.x)&&(tx==blockDim.x))
	//{
	//	uns[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-uright[ty+by*blockDim.y])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (uright[ty+by*blockDim.y]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//	vns[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-vright[ty+by*blockDim.y])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (vright[ty+by*blockDim.y]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//}
	//else if ((by==0)&&(ty==0))
	//{
	//	uns[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(uup[tx+bx*blockDim.x]-un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(un[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+uup[tx+bx*blockDim.x])/dy/dy ) );
	//	vns[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vup[tx+bx*blockDim.x]-vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx])/2/dy+nu*( (vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(vn[(ty+by*blockDim.y+1)*x+tx+blockDim.x*bx]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vup[tx+bx*blockDim.x])/dy/dy ) );
	//}
	//else if ((by==y/blockDim.y)&&(ty==blockDim.y))
	//{
	//	uns[tx][ty]=un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-udown[tx+bx*blockDim.x])/2/dy+nu*( (un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(udown[tx+bx*blockDim.x]-2.*un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+un[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//	vns[tx][ty]=vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+dt*( un[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1]-vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1])/2/dx+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]*(vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx]-vdown[tx+bx*blockDim.x])/2/dy+nu*( (vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx+1]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx-1])/dx/dx+(vdown[tx+bx*blockDim.x]-2.*vn[(ty+by*blockDim.y)*x+tx+blockDim.x*bx]+vn[(ty+by*blockDim.y-1)*x+tx+blockDim.x*bx])/dy/dy ) );
	//}
	//__syncthreads();
	//un[tx+bx*blockDim.x+x*(ty+by*blockDim.y)]=uns[tx][ty];
	//vn[tx+bx*blockDim.x+x*(ty+by*blockDim.y)]=vns[tx][ty];
	//__syncthreads();
	//if ((by==y/blockDim.y)&&(ty==blockDim.y))
	//{
	//	uup[tx+bx*blockDim.x]=uns[tx][ty];
	//	vup[tx+bx*blockDim.x]=vns[tx][ty];
	//}
	//else if ((by==0)&&(ty==0))
	//{
	//	udown[tx+bx*blockDim.x]=uns[tx][ty];
	//	vdown[tx+bx*blockDim.x]=vns[tx][ty];
	//}
	//else if ((bx==x/blockDim.x)&&(tx==blockDim.x))
	//{
	//	uleft[ty+by*blockDim.y]=uns[tx][ty];
	//	vleft[ty+by*blockDim.y]=vns[tx][ty];
	//}	
	//else if ((bx==0)&&(tx==0))
	//{
	//	uright[ty+by*blockDim.y]=uns[tx][ty];
	//	vright[ty+by*blockDim.y]=vns[tx][ty];
	//}
	//__syncthreads();



void streaming (FLDBL*u,FLDBL*v,int x, int y,FLDBL dx, FLDBL dy, int t,FLDBL dt,  FLDBL nu );
void field_div(FLDBL * un, FLDBL * vn, int x, int y, FLDBL dx, FLDBL dy);
int main(int argc, char * argv[])
{
	FLDBL lx=2*PI;
	int x=256;
	int y=256;
	FLDBL dx=lx/x;
	FLDBL ly=2*PI;
	FLDBL dy=ly/y;
	int t=100;
	FLDBL dt=0.001;
	//FLDBL lt=t*dt;
	FLDBL nu=0.001;

	FLDBL *u=new FLDBL[(x+2)*(y+2)];
	FLDBL *v=new FLDBL[(x+2)*(y+2)];

	for (int i=0;i<x+2;i++)
		for (int j=0;j<y+2;j++)
		{
			u[i+(x+2)*j]=-sin(2*PI*(j-1)/y);
			v[i+(x+2)*j]=sin(2*PI*(i-1)/x);
		};

	streaming(u,v,x,y,dx,dy,t,dt,nu);
	FILE *qout = fopen("q.dat", "w");
	for(int i = 1; i < x+1; i++)
	{
		for(int j = 1; j < y+1; j++)
		{
			if ((u[i+(x+2)*j]>2.)||(u[i+(x+2)*j]<-2.))
				u[i+(x+2)*j]=3.;
			fprintf(qout, "%lf\t%lf\t%lf\n", 2*PI*(i-1)/x, 2*PI*(j-1)/y, -(u[i+(x+2)*(j+1)]-u[i+(x+2)*(j-1)])/2/dy+(v[i+1+(x+2)*j]-v[i-1+(x+2)*j])/2/dx);
		}
		fprintf(qout, "\n");
	}
	fclose(qout);

	FILE *pout = fopen("p.dat", "w");
	for(int i = 0; i < x+2; i++)
	{
		for(int j = 0; j < y+2; j++)
		{
			if ((v[i+(x+2)*j]>2.)||(v[i+(x+2)*j]<-2.))
				v[i+(x+2)*j]=3.;
			fprintf(pout, "%lf\t%lf\t%lf\n", (double)i, (double)j, u[i+(x+2)*j]+sin(2*PI*(j-1)/y)*exp(-nu*dt*t));
		}
		fprintf(pout, "\n");
	}
	fclose(pout);

	delete [] u;
	delete [] v;

	return 0;
}
void streaming (FLDBL*u,FLDBL*v,int x, int y,FLDBL dx, FLDBL dy, int t,FLDBL dt, FLDBL nu )
{
	FLDBL * uup=new FLDBL[x];
	FLDBL * udown=new FLDBL[x];
	FLDBL * uleft=new FLDBL[y];
	FLDBL * uright=new FLDBL[y];
	FLDBL * vup=new FLDBL[x];
	FLDBL * vdown=new FLDBL[x];
	FLDBL * vleft=new FLDBL[y];
	FLDBL * vright=new FLDBL[y];
	FLDBL * un=new FLDBL[x*y];
	FLDBL * vn=new FLDBL[x*y];
	FLDBL * f=new FLDBL[x*y];
	FLDBL * p=new FLDBL[x*y];

	int i=0;
	int j=0;
	for (i=1;i<x+1;i++)
		for(j=1;j<y+1;j++)
		{
			un[i-1+x*(j-1)]=u[i+(x+2)*j];
			vn[i-1+x*(j-1)]=v[i+(x+2)*j];
		}
	//FILE *pout = fopen("p.dat", "w");
	//for(int i = 0; i < x; i++)
	//{
	//	for(int j = 0; j < y; j++)
	//	{
	//		fprintf(pout, "%lf\t%lf\t%lf\n", (double)i, (double)j, vn[i+x*j]);
	//	}
	//	fprintf(pout, "\n");
	//}
	//fclose(pout);

	for(i=1;i<x+1;i++)
	{
		uup[i-1]=u[i];
		udown[i-1]=u[i+(x+2)*(y+1)];
		vup[i-1]=v[i];
		vdown[i-1]=v[i+(x+2)*(y+1)];
	}
	for(j=1;j<y+1;j++)
	{
		uleft[j-1]=u[(x+2)*j];
		uright[j-1]=u[x+1+(x+2)*j];
		vleft[j-1]=v[(x+2)*j];
		vright[j-1]=v[x+1+(x+2)*j];
	}

	int numBytes = x*y*sizeof(FLDBL);
	FLDBL*unDev=NULL;
	FLDBL*vnDev=NULL;
	FLDBL*uleftDev=NULL;
	FLDBL*urightDev=NULL;
	FLDBL*uupDev=NULL;
	FLDBL*udownDev=NULL;
	FLDBL*vleftDev=NULL;
	FLDBL*vrightDev=NULL;
	FLDBL*vupDev=NULL;
	FLDBL*vdownDev=NULL;
	FLDBL*utempDev=NULL;
	FLDBL*vtempDev=NULL;
	FLDBL*fDev=NULL;
	FLDBL*pDev=NULL;
	dim3 threads = dim3(BLOCK_X,BLOCK_Y);
	dim3 blocks = dim3(x/threads.x,y/threads.y);
	dim3 sblocks = dim3(x/2/threads.x,y/2/threads.y);
	dim3 lsblocks = dim3((x-2)/threads.x+1, (y-2)/threads.y+1);
	cudaMalloc((void**)& unDev,numBytes);
	cudaMalloc((void**)& vnDev,numBytes);
	cudaMalloc((void**)& uupDev,x*sizeof(FLDBL));
	cudaMalloc((void**)& udownDev,x*sizeof(FLDBL));
	cudaMalloc((void**)& uleftDev,y*sizeof(FLDBL));
	cudaMalloc((void**)& urightDev,y*sizeof(FLDBL));
	cudaMalloc((void**)& vupDev,x*sizeof(FLDBL));
	cudaMalloc((void**)& vdownDev,x*sizeof(FLDBL));
	cudaMalloc((void**)& vleftDev,y*sizeof(FLDBL));
	cudaMalloc((void**)& vrightDev,y*sizeof(FLDBL));
	cudaMalloc((void**)& utempDev,numBytes);
	cudaMalloc((void**)& vtempDev,numBytes);
	cudaMalloc((void**)& fDev,numBytes);
	cudaMalloc((void**)& pDev,numBytes);
	cudaMemcpy(unDev,un,numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(vnDev,vn,numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(uupDev,uup,x*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(udownDev,udown,x*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(uleftDev,uleft,y*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(urightDev,uright,y*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(vupDev,vup,x*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(vdownDev,vdown,x*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(vleftDev,vleft,y*sizeof(FLDBL),cudaMemcpyHostToDevice);
	cudaMemcpy(vrightDev,vright,y*sizeof(FLDBL),cudaMemcpyHostToDevice);
		cufftDoubleComplex *data;
		cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*x*y);
	for (int ti=0;ti<t;ti++)
	{
		if (!(ti%10))
			printf("%d from %d\n",ti/10,t/10);
		cufftHandle plan;

		streamingKernel<<<blocks,threads>>>(utempDev,vtempDev,unDev,vnDev,uupDev,udownDev,uleftDev, urightDev,vupDev,vdownDev,vleftDev, vrightDev,dx,dy,dt,x,y,nu);
		//printf("Start: %s\n",cudaGetErrorString(cudaGetLastError()));
		lineKernel<<<blocks,threads>>>(utempDev,vtempDev,fDev,x,y,dx,dy);
		//printf("After drift: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftPlan2d(&plan, y, x, CUFFT_D2Z);
		//printf("Making a plan: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftExecD2Z(plan, fDev, data);
		//printf("Exec: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftDestroy(plan);
		//printf("Destroying: %s\n",cudaGetErrorString(cudaGetLastError()));
		fourieKernel<<<sblocks,threads>>>(data, dx,dy,x,y);
		//printf("Fourie: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftPlan2d(&plan, y, x, CUFFT_Z2D);
		//printf("Making a plan: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftExecZ2D(plan,data,pDev);
		//printf("Exec: %s\n",cudaGetErrorString(cudaGetLastError()));
		cufftDestroy(plan);
		//printf("Destroying: %s\n",cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(p,pDev,numBytes,cudaMemcpyDeviceToHost);
		for (i=0;i<x;i++)
			for (j=0;j<y;j++)
			{
				p[j+x*i]=p[j+y*i]/x/y;
			}
		FLDBL pom=0;
		for (i=0;i<x/2;i++)
			for (j=0;j<y/2;j++)
			{
				pom=p[i+x*j];
				p[i+x*j]=p[j+y*i];
				p[j+y*i]=pom;
			}
	FILE *pressout = fopen("press.dat", "w");
	for(int i = 0; i < x; i++)
	{
		for(int j = 0; j < y; j++)
		{
			fprintf(pressout, "%lf\t%lf\t%lf\n", (double)i, (double)j, p[i+x*j]);
		}
		fprintf(pressout, "\n");
	}
	fclose(pressout);

		cudaMemcpy(pDev,p,numBytes,cudaMemcpyHostToDevice);
		//printf("Norm: %s\n",cudaGetErrorString(cudaGetLastError()));
		summaryKernel<<<lsblocks, threads>>>(pDev,utempDev,vtempDev,unDev,vnDev,uupDev,udownDev,uleftDev, urightDev,vupDev,vdownDev,vleftDev, vrightDev,x,y,dx,dy);
		//printf("Last: %s\n",cudaGetErrorString(cudaGetLastError()));

	//FILE *rout = fopen("r.dat", "w");
	//for(int i = 0; i < x; i++)
	//{
	//	for(int j = 0; j < y; j++)
	//	{
	//		fprintf(rout, "%lf\t%lf\t%lf\n", (double)i, (double)j, data[i+x*j].x);
	//	}
	//	fprintf(rout, "\n");
	//}
	//fclose(rout);


	}
		cudaFree(data);
	cudaMemcpy(un,unDev,numBytes,cudaMemcpyDeviceToHost);
	cudaMemcpy(vn,vnDev,numBytes,cudaMemcpyDeviceToHost);
	cudaMemcpy(uup,uupDev,x*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(udown,udownDev,x*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(uleft,uleftDev,y*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(uright,urightDev,y*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(vup,vupDev,x*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(vdown,vdownDev,x*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(vleft,vleftDev,y*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaMemcpy(vright,vrightDev,y*sizeof(FLDBL),cudaMemcpyDeviceToHost);
	cudaFree(unDev);
	cudaFree(vnDev);
	cudaFree(uupDev);
	cudaFree(udownDev);
	cudaFree(uleftDev);
	cudaFree(urightDev);
	cudaFree(vupDev);
	cudaFree(vdownDev);
	cudaFree(vleftDev);
	cudaFree(vrightDev);
	cudaFree(utempDev);
	cudaFree(vtempDev);
	cudaFree(fDev);
	field_div(un,vn,x,y,dx,dy);


	for (i=1;i<x+1;i++)
		for(j=1;j<y+1;j++)
		{
			u[i+(x+2)*j]=un[i-1+x*(j-1)];
			v[i+(x+2)*j]=vn[i-1+x*(j-1)];
		}
	for(i=1;i<x+1;i++)
	{
		u[i]=uup[i-1];
		u[i+(x+2)*(y+1)]=udown[i-1];
		v[i]=vup[i-1];
		v[i+(x+2)*(y+1)]=vdown[i-1];
	}
	for(j=1;j<y+1;j++)
	{
		u[(x+2)*j]=uleft[j-1];
		u[x+1+(x+2)*j]=uright[j-1];
		v[(x+2)*j]=vleft[j-1];
		v[x+1+(x+2)*j]=vright[j-1];
	}
	delete [] un;
	delete [] vn;
	delete [] uleft;
	delete [] uright;
	delete [] uup;
	delete [] udown;
	delete [] vleft;
	delete [] vright;
	delete [] vup;
	delete [] vdown;
	delete [] f;
	delete [] p;
}
void field_div(FLDBL * un, FLDBL * vn, int x, int y, FLDBL dx, FLDBL dy)
{
	FLDBL divu=0;
	int maxi=0;
	int maxj=0;
	int i=0;
	int j=0;
	divu=0;
	for (i=1;i<x-1;i++)
		for (j=1;j<y-1;j++)
			if(divu<fabs((un[i+1+x*j]-un[i-1+x*j])/dx/2+(vn[i+x*(j+1)]-vn[i+x*(j-1)])/dy/2))
			{
				divu=fabs((un[i+1+x*j]-un[i-1+x*j])/dx/2+(vn[i+x*(j+1)]-vn[i+x*(j-1)])/dy/2);
				maxi=i;
				maxj=j;
			}
	printf("~~~i=%d,j=%d~\n",maxi,maxj);
	printf("~~~~div = %f;\n",divu);
	FLDBL omega=0;
	for (i=0;i<x-1;i++)
		for (j=0;j<y-1;j++)
			if(omega<abs(-(un[i+x*(j+1)]-un[i+x*j])/dy+(vn[i+1+x*j]-vn[i+x*j])/dx))
				omega=abs(-(un[i+x*(j+1)]-un[i+x*j])/dy+(vn[i+1+x*j]-vn[i+x*j])/dx);
	printf("~~~~omega=%f\n",omega);

}