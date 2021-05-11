# animation

这个库主要是用来动画化一些体数据，如3D流数据，tornado数据，hurricane数据，earthquake数据

运行顺序：
  （1）运行normGradAndNormNormalsGen.m, 产生normGrad和法向量数据
  （2）运行v12volumeRender.cpp，动画化
  
  注：该版本代码适用于地震数据，如果要应用其他数据，请适当修改参数
