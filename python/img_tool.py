import numpy as np
import cv2
import random
# image tools based on libs or from scratch using numpy
#Reference: Opencv Tutorial:https://docs.opencv.org/4.2.0/d6/d00/tutorial_py_root.html
#Have Done:
#+ rotation matrix generation
#+ affine transform from scratch
#ToDo:
#+ Interpolations
#+ crop
#+ resize
#+ color normalization
#+ elastic transformation
#+ code simlification and acceleration
### write from scratch based on numpy###
def gen_rot_mat(ang,h,w,s=1):
    ''' generate rotation matrix with input in degrees,pad to keep all img'''
    ang = ang % 360
    if ang > 180:
        ang -= 360
    ang = np.pi * ang/180 # map to [-pi,pi]
    mat = np.zeros([3,3])
    mat[0,0] = np.cos(ang)
    mat[0,1] = -np.sin(ang)
    mat[1,0] = np.sin(ang)
    mat[1,1] = np.cos(ang)
    mat[2,2] = 1
    
    br = np.dot(mat,np.array([w,h,1]).T)
    bl = np.dot(mat,np.array([0,h,1]).T)
    ur = np.dot(mat,np.array([w,0,1]).T)
    ul = np.dot(mat,np.array([0,0,1]).T)
    nw = int(round(max(br[0],bl[0],ur[0],ul[0])-min(br[0],bl[0],ur[0],ul[0])))
    nh = int(round(max(br[1],bl[1],ur[1],ul[1])-min(br[1],bl[1],ur[1],ul[1])))
    nh = max(nh,h)
    nw = max(nw,w)
    trans = np.eye(3)
    trans[0,2] = -w/2
    trans[1,2] = -h/2
    mat = np.dot(mat,trans)
    trans[0,2] = w/2
    trans[1,2] = h/2
    mat = np.dot(trans,mat)
    return mat,(nh,nw)
### interpolations ###
def bilinear():
    pass
inters={'bilinear':bilinear}
def affine_trans2d(src,mat,tsize=None,interpolation='bilinear'):
    # [x1,y1] = mat.*[x,y,1]
    if len(src.shape)<3:
       np.expand_dims(src,axis=2)
    h,w,c=src.shape
    nh,nw = tsize
    #inter = inters[interpolation] 
    if tsize==None:
        target = np.zeros(src.shape,dtype=src.dtype)
    else:
        target = np.zeros([nh,nw,c],dtype=src.dtype)
    for i in range(nh):
        for j in range(nw):
            pt = np.dot(mat,np.array([j,(nh-i-1),1]).T)
            x = pt[0]
            y = pt[1]
            if 0<=y<=h and 0<=x<w:
                y = int(min(max(round(y),0),h-1))
                x = int(min(max(round(x),0),w-1))
                target[i,j,:]=src[h-y-1,x,:]
    return target   
def rotate(img,ang):
    h,w,c = img.shape
    mat,tsize = gen_rot_mat(ang,h,w)
    print(np.linalg.inv(mat))
    inv = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
    print(inv)
    nh,nw = tsize
    dh,dw = (nh-h)//2,(nw-w)//2
    target = np.zeros((nh,nw,c),dtype=img.dtype)
    target[dh:dh+h,dw:dw+w,:] = img
    #padding to keep all pixels
    res = cv2.warpAffine(img,inv[:2,:],(w,h))
    #show_img(res)
    #res = affine_trans2d(target,inv,(nh,nw))
    print(res.shape)
    return res

def show_img(img,tag='result'):
    cv2.imshow(tag,img)
    cv2.waitKey()


if __name__ == "__main__":
    img = cv2.imread("lena2.jpg")
    print(img.shape)
    ang = random.uniform(0,1)*360
    print(ang)
    res = rotate(img,20)
    nh,nw,_=res.shape
    cv2.circle(res,(nw//2,nh//2),2,(255,0,0))
    cv2.circle(res,(192,256),2,(0,255,0))
    show_img(res)