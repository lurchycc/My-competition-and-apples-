try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
from scipy.spatial import voronoi_plot_2d
import matplotlib.pyplot as plt
#解析图片，获取图片标注位置
def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc=[x1,y1,x2,y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet
crossing = GetAnnotBoxLoc('C:/Users/lurchy/Desktop/EasyDL/8230-4502-0.5.xml')
#读取图片,计算矩形框中心坐标
img = cv2.imread('C:/Users/lurchy/Desktop/EasyDL/8230-4502-0.5.jpg')
cross = crossing['crossing']
#读取图片

spot_add = []
qian = []
m=0
for i in cross:
    qian.append(int((i[m]+i[m+2])/2))
    qian.append(int((i[m+1]+i[m+3])/2))
    spot_add.append(qian)
    qian = []
#图像显示函数
def cv_imshow(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#双边滤波
d = cv2.getTrackbarPos("d","image")
sigmaColor = cv2.getTrackbarPos("sigmaColor","image")
sigmaSpace = cv2.getTrackbarPos("sigmaSpace","image")
img_ = cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)
# cv_imshow(img_,'a')
#自动色阶
def ComputeHist(img):
    h,w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1,w*h), bins=list(range(257)))
    return hist
    
def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i
            
def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255-i]
        if (sum >= (pnum * rate * 0.01)):
            return 255-i
            
def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):    #获取阈值外的像素值 i< minlevel，i> maxlevel
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i-minlevel)/(maxlevel-minlevel) * 255
        return newmap
        
def CreateNewImg(img):
    h,w,d = img.shape
    newimg = np.zeros([h,w,d])
    for i in range(d):
        imgmin = np.min(img[:,:,i])
        imgmax = np.max(img[:,:,i])
        imghist = ComputeHist(img[:,:,i])
        minlevel = ComputeMinLevel(imghist, 8.3, h*w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h*w)
        newmap = LinearMap(minlevel,maxlevel)
        if (newmap.size ==0 ):
            continue
        for j in range(h):
            newimg[j,:,i] = newmap[img[j,:, i]]
    return newimg
newimg = CreateNewImg(img)
def grad_(m,n):#输入两个坐标，计算坐标之间梯度
    crop_img = newimg[m[1]:n[1],m[0]:n[0]]
    try:
        sobely = cv2.Sobel(crop_img,cv2.CV_64F,0,1,ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        s=0
        m=0
        for i in sobely:
            for j in i:
                for k in j:
                    if k!=0:
                        m+=1
                        s+=k
        grad = s/m
        return grad
    except cv2.error:
        print('error again!')
    # sobely = cv2.convertScaleAbs(sobely)
    # s=0
    # m=0
    # for i in sobely:
    #     for j in i:
    #         for k in j:
    #             if k!=0:
    #                 m+=1
    #                 s+=k
    # grad = s/m#梯度转化为与阈值判断的标准数据格式
    # return grad
del_b = []
for i in range(len(spot_add)):
    for j in range(i,len(spot_add)):
        if(i!=j):
            if(isinstance(grad_(spot_add[i],spot_add[j]),float)):
                if(grad_(spot_add[i],spot_add[j])>41.81):
                    gd=[]
                    gd.append(spot_add[i])
                    gd.append(spot_add[j])
                    del_b.append(gd)