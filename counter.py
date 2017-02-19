# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:50:50 2017

@author: Danilo
"""
import cv2
import numpy as np

from vector import distance
from skimage.measure import label
from scipy import ndimage

from skimage.measure import regionprops
from skimage.morphology import *
from vector import pnt2line2
from sklearn.datasets import fetch_mldata
from skimage import color

videoName="data/video-9.avi"
cap = cv2.VideoCapture(videoName)


##########NOVI PODACI NOVI PODACI 


minLineLength = 600;

def houghTransformtion(frame,grayImg):
    global minLineLength;
    edges = cv2.Canny(grayImg,50,150,apertureSize = 3)#canny vraca ivice cele slike 
    minx,miny,maxx,maxy=advancedHoughTransformation(frame,edges,minLineLength)

    return minx,miny, maxx,maxy


def advancedHoughTransformation(frame,edges,minLineLength):

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength,8)#vraca vektora linija i smest ih u linije 

    minx=1500
    miny=1500
    maxy=-5
    maxx=-5

    for i in  range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if x2>maxx : #and y2>52
                maxy=y2
                maxx=x2            
            if x1<minx :  
                minx=x1
                miny=y1

    return minx,miny,maxx,maxy

mnist = fetch_mldata('MNIST original')
new_mnist_set=[]    
    
def findLineParams(videoName):
    cap = cv2.VideoCapture(videoName)
    kernel = np.ones((2,2),np.uint8)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img0 = cv2.dilate(gray, kernel)  

        cap.release()
        cv2.destroyAllWindows()
        return houghTransformtion(frame,img0)                
       

#######################################################

lineX1,lineY1,lineX2,lineY2=findLineParams(videoName)

m=0;
while m < 70000:
    img_from_mnist=mnist.data[m].reshape(28,28)
    img_from_mnist_BW=((color.rgb2gray(img_from_mnist)/255.0)>0.88).astype('uint8')
    
    try:
        newImg="newImage"
        minx=500
        miny=500
        maxx=-1
        maxy=-1
        label_img = label(img_from_mnist_BW)
        regions = regionprops(label_img)
        for region in regions:
            bbox = region.bbox
            if bbox[0]<minx:
                minx=bbox[0]
            if bbox[1] <miny:
                miny=bbox[1]
            if bbox[2]>maxx:
                maxx=bbox[2]
            if bbox[3]>maxy:
                maxy=bbox[3]
    
        

        
    #   print "visina broja je: " + format(height);
    #print "****************************************";
        newImg = np.zeros((28, 28))
       #print "*********************************";
        height = maxy - miny
        
        width = maxx - minx
       # print "sirina broja je: " +format(width);
       #print "*********************************";       
        newImg[0:width, 0:height] = newImg[0:width, 0:height] + img_from_mnist_BW[minx:maxx, miny:maxy]
        new_mnist_img = newImg;
    except ValueError: 
        pass    
    
    new_mnist_set.append(new_mnist_img)
    m=m+1
    
passedSum = 0
idNumber = -1

#def currentId():
#    global idNumber
#    idNumber += 1
#    return idNumber

def inRange(items, item):
    retVal = []
    for obj in items:
        mdist = distance(obj['center'],item['center'])
        
        if(mdist<20):
            retVal.append(obj)        
    return retVal


def main():
    kernel = np.ones((2,2),np.uint8)
    donjaGranica = np.array([220, 220, 220],dtype="uint8")
    gornjaGranica = np.array([255, 255, 255],dtype="uint8")
    
    
    elements = []
    t = 0
    counter = 0

    while (1):
        ret, img = cap.read()
        if not ret:
            break
        

        mask = cv2.inRange(img, donjaGranica, gornjaGranica)
#        cv2.circle(img, ((lineX1,lineY1)), 4, (25, 25, 255), 1)
#        cv2.circle(img, ((lineX2,lineY2)), 4, (25, 25, 255), 1)
        #cv2.waitKey()
        img0 = 1.0* mask
        img01 = 1.0* mask
        
        img0 = cv2.dilate(img0, kernel)  
        img0 = cv2.dilate(img0, kernel)
 
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        
        numberOfObject = range(nr_objects)
        for i in numberOfObject:
            loc = objects[i]
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)


            if (dxc > 11 or dyc > 11):
                elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                lst = inRange(elements, elem)
                nn = len(lst)
                if nn == 0:
                    global idNumber;
                    idNumber += 1;
                    x1=xc-14
                    x2=xc+14
                    y1=yc-14
                    y2=yc+14
                    elem['id'] = idNumber;
                    elem['t'] = t
                    elem['pass'] = False
                    img_BW=color.rgb2gray(img01[y1:y2,x1:x2]) >= 0.88
                    img_BW=(img_BW).astype('uint8')
                    
                    try:
                        label_img = label(img_BW)
                        regions = regionprops(label_img)
                
                        newImg="newImage"
                        minx=500
                        miny=500
                        maxx=-1
                        maxy=-1
                        for region in regions:
                            bbox = region.bbox
                            if bbox[0]<minx:
                                minx=bbox[0]
                            if bbox[1] <miny:
                                miny=bbox[1]
                            if bbox[2]>maxx:
                                maxx=bbox[2]
                            if bbox[3]>maxy:
                                maxy=bbox[3]
                    
                        
                        width = maxx - minx
                       # print "sirina broja je: " +format(width);
                       #print "*********************************";
                        height = maxy - miny
                    #   print "visina broja je: " + format(height);
                    #print "****************************************";
                        newImgLeft = np.zeros((28, 28))
                        
                        newImgLeft[0:width, 0:height] = newImgLeft[0:width, 0:height] + img_BW[minx:maxx, miny:maxy]
                        newImg = newImgLeft;
                    except ValueError: 
                        pass   
                    
                    minDifference = 8888
                    number = -10
                    for i in range(70000):
                        difference=0
                        mnist_img=new_mnist_set[i]
                        diff = mnist_img!=newImg
                        difference=np.sum(diff)
                        if difference < minDifference:
                            minDifference = difference
                            number = mnist.target[i]
                    elem['number'] = number;
                    elements.append(elem)
                    
                else:
                    elementForUpdate = lst[0]
                    for obj in lst:
                        if distance(elem['center'],elementForUpdate['center']) > distance(obj['center'],elem['center']):
                            elementForUpdate = obj
                    el = elementForUpdate;
                    el['center'] = elem['center']
                    el['t'] = t
                

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                dist, pnt, r = pnt2line2(el['center'], (lineX1,lineY1), (lineX2,lineY2))
                if r > 0:
                    if (dist < 10):
                        if el['pass'] == False:
                            el['pass'] = True
                            counter += 1
                            (x,y)=el['center']
                            x1=x-14
                            x2=x+14
                            y1=y-14
                            y2=y+14
                            global passedSum
                            passedSum += el['number']
                            print "Prosao broj :  " + format(el['number'])
                            
                            #findNumber(img01[y1:y2,x1:x2])
                            
        cv2.putText(img, 'Counter: ' + str(counter), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

        t += 1
        if t % 100 == 0:
            print "Frame :  " + format(t);

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    print "ZBIR JE: " + format(passedSum)
    cap.release()
    cv2.destroyAllWindows()
main()
