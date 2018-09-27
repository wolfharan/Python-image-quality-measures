#LIBRARIES

import cv2
import numpy as np

#-----------------------------------------

#FUNCTIONS

#Mean square error
def MSE(imgA,imgB):
    rows,cols=imgA.shape
    sumimg=0
    for i in range(rows):
        for j in range(cols):
            sumimg=sumimg+((float(imgA[i,j])-float(imgB[i,j]))**2)
    return sumimg/(rows*cols)
    
#Peak signal to noise ratio
def PSNR(imgA,imgB):
    rows,cols=imgA.shape
    sumimg=0
    for i in range(rows):
        for j in range(cols):
            sumimg=sumimg+((float(imgA[i,j])-float(imgB[i,j]))**2)
    mse=sumimg/(rows*cols)
    const=25 5**2
    frac=float(const/mse)
    return 10*np.log10(frac)
    

#average difference
def avgdiff(imgA,imgB):
    rows,cols=imgA.shape
    sumimg=0
    for i in range(rows):
        for j in range(cols):
            sumimg=sumimg+(float(imgA[i,j])-float(imgB[i,j]))
    return sumimg/(rows*cols)
    

#Normalized correlation
def normalizedcorrelation(imgA,imgB):
    rows,cols=imgA.shape
    sqrAsum=0
    productsum=0
    for i in range(rows):
        for j in range(cols):
            productsum=productsum+(float(imgA[i,j])*float(imgB[i,j]))
            sqrAsum=sqrAsum+((float(imgA[i,j]))**2)
   
    return productsum/sqrAsum
    

#Maximum Difference

def maxdiff(imgA,imgB):
    diff=np.absolute(imgA.astype('float')-imgB.astype('float'))
    return np.amax(diff)

#Normalized Absolute error
def normalizedabsoluterr(imgA,imgB):
    rows,cols=imgA.shape
    Asum=0
    productsum=0
    for i in range(rows):
        for j in range(cols):
            product=abs(float(imgA[i,j])-float(imgB[i,j]))
            productsum=productsum+product
            Asum=Asum+abs(float(imgA[i,j]))
    return productsum/(Asum)

#Structural Content
def structuralcontent(imgA,imgB):
    rows,cols=imgA.shape
    Asum=0
    Bsum=0
    for i in range(rows):
        for j in range(cols):
            Asum=Asum+(float(imgA[i,j]))**2
            Bsum=Bsum+(float(imgB[i,j]))**2
   
    return Asum/Bsum
