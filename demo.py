from infer import vaild
import os
import random
import cv2

class val(object):
    def __init__(self,epoch):
        super(val, self).__init__()
        vaild(epoch)


def genRandImg1(size):
    path='./gg'
    picture_rand=os.listdir(path)
    len_rand_picture=len(picture_rand)
    x=random.randint(0,len_rand_picture-1)
    name_image=picture_rand[x]
    picture=cv2.imread(path+'/'+name_image,0)
    picture = cv2.resize(picture, size)
    # print(picture)
    _,mask_pict=cv2.threshold(picture,150,255,cv2.THRESH_BINARY)
    #
    # cv2.imshow('image',mask_pict)
    # cv2.waitKey()

genRandImg1(size=(128,128))