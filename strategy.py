import sys
import math
import os
import numpy as np
import cv2

def getgrasp(Item,depth_map,point_map,baseHcam,xmin,ymin,w,h):    #return (flag,cX,cY,rZ)
    flag = 1
    div_num = 6
    up = np.ones(div_num,int)*-1
    down = np.ones(div_num,int)*-1
    div = np.ones(div_num,int)*-1
    if (Item=='plastic bag' or Item=='paper' ):
        cY = ymin+h/2
        cX = xmin+w/2
        if(w > 0.9*h):
            rZ=-math.pi/2
        else:
            rZ=0
    elif (Item == 'bottle'):
        cY = ymin + h / 2
        cX = xmin + w / 2
        if (w > h):
            rZ = -math.pi / 2
        else:
            rZ = 0
    elif (Item=='cup' or Item=='citrus'):
        maxedge = 15
        if Item=='cup':
            edge = 15
        else:
            edge = 5
        if (w > h):
            temp = xmin + w / 4
            divline = np.array(int(temp))
            for i in range(div_num):
                if (i > 0):
                    divline = np.append(divline, int(temp))
                temp2 = int(ymin)
                while (temp2 <= int(ymin + h)):
                    if (str(depth_map[temp2, int(temp)]) == 'nan'):
                        depth_map[temp2, int(temp)] = depth_map[temp2 - 1, int(temp)]
                    if (depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)] < -edge):
                        if (down[i]==-1 and up[i] == -1):
                            up[i] = temp2 - 2
                            div[i] = int(temp)
                            print('up', depth_map[temp2, int(temp)], depth_map[temp2 - 5, int(temp)])
                            if (depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)]<-maxedge):
                                maxedge = -(depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)])
                    elif (depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)] > edge):
                        if (up[i] != -1):
                            down[i] = temp2 - 2
                            div[i] = int(temp)
                            print('down', depth_map[temp2, int(temp)], depth_map[temp2 - 5, int(temp)])
                            if (depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)]>maxedge):
                                maxedge= depth_map[temp2, int(temp)] - depth_map[temp2 - 5, int(temp)]
                    temp2 = temp2 + 1
                temp = temp + w / 2 / (div_num - 1)
        else:
            flag = 0
            temp = ymin + h / 4
            divline = np.array(int(temp))
            for i in range(div_num):
                if (i > 0):
                    divline = np.append(divline, int(temp))
                temp2 = int(xmin)
                while (temp2 <= int(xmin + w)):
                    if (str(depth_map[int(temp), temp2]) == 'nan'):
                        depth_map[int(temp), temp2] = depth_map[int(temp), temp2 - 1]
                    if (depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5] < -edge):
                        if (down[i] == -1 and up[i] == -1):
                            up[i] = temp2 - 2
                            div[i] = int(temp)
                            print('up', depth_map[int(temp), temp2], depth_map[int(temp), temp2 - 5])
                            if (depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5]<-maxedge):
                                maxedge = -(depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5] )
                    elif (depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5] > edge):
                        if (up[i] != -1):
                            down[i] = temp2 - 2
                            div[i] = int(temp)
                            print('down', depth_map[int(temp), temp2], depth_map[int(temp), temp2 - 5])
                            if (depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5] > maxedge):
                                maxedge = depth_map[int(temp), temp2] - depth_map[int(temp), temp2-5]
                    temp2 = temp2 + 1
                temp = temp + h / 2 / (div_num - 1)
        firup = 0
        firdown = 0
        endup = -1
        enddown = -1
        while (up[firup] == -1 and firup < div_num / 2):
            firup = firup + 1
        while (down[firdown] == -1 and firdown < div_num / 2):
            firdown = firdown + 1
        while (up[endup] == -1 and endup > -(div_num-firup-1)):
            endup = endup - 1
        while (down[enddown] == -1 and enddown > -(div_num-firdown-1)):
            enddown = enddown - 1
        fir = max(firup, firdown)
        end = min(endup, enddown)
        if (flag):
            if (up[fir] == -1 or up[end] == -1 or down[fir] == -1 or down[end] == -1):
                flag=0
                cX = div[int(div_num / 2)]
                cY = (up[int(div_num / 2)] + down[int(div_num / 2)]) / 2
                return (flag, cX, cY, -math.pi/2)
            if (div[fir]==-1 or div[end]==-1 or up[int((fir+end+div_num)/2)]==-1 or down[int((fir+end+div_num)/2)]==-1 ):
                cY = ymin + h / 2
                cX = xmin + w / 2
            else:
                cX = (div[fir] + div[end]) / 2
                cY = (up[int((fir + end + div_num) / 2)] + down[int((fir + end + div_num) / 2)]) / 2
                if (cX==-1):
                    cX = xmin + w / 2
                if (cY==-1):
                    cY = ymin + h / 2
            print('cX,xY ', cX, cY)
            print('maxedge:',maxedge)
            if (maxedge > 50):
                return (3, cX, cY, -math.pi / 2)
            elif (maxedge > 65):
                return (2, cX, cY, -math.pi / 2)
            tempX = np.array([])
            tempY = np.array([])
            for i in range(fir, div_num + end + 1):
                coincam = point_map[int((up[i] + down[i]) / 2), div[i]]
                coincam[3] = 1
                coinbase = np.dot(baseHcam, coincam)
                if (str(coinbase[0])!='nan' and str(coinbase[1])!='nan'):
                    tempX = np.append(tempX, coinbase[0])
                    tempY = np.append(tempY, coinbase[1])
            print('tempX',tempX)
            print('tempY',tempY)
            k = (np.sum(tempY * tempX) - len(tempY) * np.mean(tempY) * np.mean(tempX)) / (
                        np.sum(tempX * tempX) - len(tempY) * np.mean(tempX) * np.mean(tempX))
            if (k < 0):
                rZ = math.atan(k)
            else:
                rZ = math.atan(k) - math.pi
            print('rZ',rZ)
        else:
            flag = 1
            if (up[fir] == -1 or up[end] == -1 or down[fir] == -1 or down[end] == -1):
                flag = 0
                cY = div[int(div_num / 2)]
                cX = (up[int(div_num / 2)] + down[int(div_num / 2)]) / 2
                return (flag, cX, cY, 0)
            if (div[fir]==-1 or div[end]==-1 or up[int((fir+end+div_num)/2)]==-1 or down[int((fir+end+div_num)/2)]==-1 ):
                cY = ymin + h / 2
                cX = xmin + w / 2
            else:
                cY = (div[fir] + div[end]) / 2
                cX = (up[int((fir + end + div_num) / 2)] + down[int((fir + end + div_num) / 2)]) / 2
                if (cX==-1):
                    cX = xmin + w / 2
                if (cY==-1):
                    cY = ymin + h / 2
            print('cX,xY ', cX, cY)
            print('maxedge:',maxedge)
            if (maxedge > 50):
                return (3, cX, cY, -math.pi / 2)
            elif (maxedge > 65):
                return (2, cX, cY, -math.pi / 2)
            tempX = np.array([])
            tempY = np.array([])
            for i in range(fir, div_num + end + 1):
                coincam = point_map[div[i], int((up[i] + down[i]) / 2)]
                coincam[3] = 1
                coinbase = np.dot(baseHcam, coincam)
                if(str(coinbase[0])!='nan' and str(coinbase[1])!='nan'):
                    tempX = np.append(tempX, coinbase[0])
                    tempY = np.append(tempY, coinbase[1])
            k = (np.sum(tempY * tempX) - len(tempY) * np.mean(tempY) * np.mean(tempX)) / (
                    np.sum(tempX * tempX) - len(tempY) * np.mean(tempX) * np.mean(tempX))
            if (k < 0):
                rZ = math.atan(k)
            else:
                rZ = math.atan(k) - math.pi
            print('rZ ',rZ)
        print('div,up,down', div, up, down)
        print('fir', fir, 'end', end)
    else:
        cY = ymin + h / 2
        cX = xmin + w / 2
        return (0, cX, cY, -math.pi/2)
    return (flag,cX,cY,rZ)





