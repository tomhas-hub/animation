clear all;
clc;

%计算程序运行时间.
tic;

%---------%only information here needs to be provided by user.----------%
noFiles = 50;
fileStartVal = 0;
fileIncrement = 1;
firstTimeStep = 1;
numOfRows = 128;
numOfColumns = 128;
numOfSlices = 128;
dataPath = 'D:\Research\Feature_Tracking_Tornado\Feature_Tracking_Tornado\v6_result\'
saveDataPath = 'D:\Research\Feature_Tracking_Tornado\Feature_Tracking_Tornado\v6_result\';
singleDataType = 'single';
%method和normFactor_method参见:
%(1)https://uk.mathworks.com/help/images/ref/imgradientxyz.html#bu4cewe-1-method
%(2)https://uk.mathworks.com/help/images/ref/imgradient3.html
method = 'sobel';
normFactor_method = 1/44; 
%---------%only information here needs to be provided by user.----------%

for t = firstTimeStep: noFiles - 1  %t: [0, 49].
    %1.读取原始数据normInten.
    fileName = sprintf('normInten_%d.raw', fileStartVal + t * fileIncrement);
    fid = fopen(strcat(saveDataPath, fileName), 'r');
    oneColumnMatrix = fread(fid, numOfRows * numOfColumns * numOfSlices, singleDataType);
    fclose(fid);
    B = reshape(oneColumnMatrix, [numOfColumns, numOfRows, numOfSlices]);
    normInten = permute(B, [2, 1, 3]);  %the array is the same as in C.
    clear oneColumnMatrix;


    %2. 已知normInten:
    %(2.1)计算x, y, z三个方向的directional gradients, 作为法线normals.
    [Gx, Gy, Gz] = imgradientxyz(normInten, method);


    %(2.2)计算gradient magnitude.
    [Gmag, Gdir, Gelevation] = imgradient3(Gx, Gy, Gz);


    %(2.3)normalize Gx, Gy, Gz and Gmag.
    %查看Gx, Gy, Gz and Gmag min. and max. values.
    min_Gx = min(min(min(Gx)));
    max_Gx = max(max(max(Gx)));
    min_Gy = min(min(min(Gy)));
    max_Gy = max(max(max(Gy)));
    min_Gz = min(min(min(Gz)));
    max_Gz = max(max(max(Gz)));
    min_Gmag = min(min(min(Gmag)));
    max_Gmag = max(max(max(Gmag)));

    %normalization.
    normGx = Gx * normFactor_method;
    normGy = Gy * normFactor_method;
    normGz = Gz * normFactor_method;
    normGmag = Gmag * normFactor_method; %(Gmag - min_Gmag)/(max_Gmag - min_Gmag);

    %查看normalized Gx, Gy, Gz and Gmag min. and max. values.
    min_normGx = min(min(min(normGx)));
    max_normGx = max(max(max(normGx)));
    min_normGy = min(min(min(normGy)));
    max_normGy = max(max(max(normGy)));
    min_normGz = min(min(min(normGz)));
    max_normGz = max(max(max(normGz)));
    min_normGmag = min(min(min(normGmag)));
    max_normGmag = max(max(max(normGmag)));


    %3. save normalized Gx, Gy, Gz和Gmag.
    %(3.1)save normGx.
    normGx = permute(normGx, [2, 1, 3]);
    fileName = sprintf('normNormalX_%d.raw', fileStartVal + t * fileIncrement);
    fid = fopen(strcat(saveDataPath, fileName), 'w');
    numOfVoxels = fwrite(fid, normGx, singleDataType);
    fclose(fid);
    fprintf('%s has been saved.\n', fileName);


    %(3.2)save normGy.
    normGy = permute(normGy, [2, 1, 3]);
    fileName = sprintf('normNormalY_%d.raw', fileStartVal + t * fileIncrement);
    fid = fopen(strcat(saveDataPath, fileName), 'w');
    numOfVoxels = fwrite(fid, normGy, singleDataType);
    fclose(fid);
    fprintf('%s has been saved.\n', fileName);


    %(3.3)save normGz.
    normGz = permute(normGz, [2, 1, 3]);
    fileName = sprintf('normNormalZ_%d.raw', fileStartVal + t * fileIncrement);
    fid = fopen(strcat(saveDataPath, fileName), 'w');
    numOfVoxels = fwrite(fid, normGz, singleDataType);
    fclose(fid);
    fprintf('%s has been saved.\n', fileName);


    %(3.4)save norm Gmag.
    normGmag = permute(normGmag, [2, 1, 3]);
    fileName = sprintf('normGrad_%d.raw', fileStartVal + t * fileIncrement);
    fid = fopen(strcat(saveDataPath, fileName), 'w');
    numOfVoxels = fwrite(fid, normGmag, singleDataType);
    fclose(fid);
    fprintf('%s has been saved.\n', fileName);
    
end %end t.


%计算程序运行时间.
executionTime = toc;
fprintf('normGradAndNormNormalsGen.m execution time = %fs.\n', executionTime);


