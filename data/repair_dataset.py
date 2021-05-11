import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2 as cv
import numpy as np
import os


class ROI_transforms(object):
    def __init__(self, size=(128, 128)):
        super(ROI_transforms).__init__()
        self.size = size
        self.area = self.size[0] * self.size[1]
        self.prosepect_pmax = 0.1
        self.morph_method = [cv.MORPH_DILATE, cv.MORPH_ERODE, cv.MORPH_OPEN, cv.MORPH_CLOSE]
        self.gray_mask_method = [self.rand_mask, self.gauss_mask, self.const_mask]

    def transform_bit(self, x):
        x = self.get_shape_mask(x)
        x = self.rotate(x, angle=[0, 359])

        if random.random() > 0.5:
            x = self.scale(x, [0.5, 2], [0.5, 2])
        if random.random() > 0.5:
            x = self.random_morphological_processing(x, k_size=[3, 11])
        # x = self.align_img(x)
        x = self.crop_or_pad(x)
        _, x = cv.threshold(x, 20, 255, cv.THRESH_BINARY)
        return x

    def transform_gray(self, x):
        x = self.get_shape_mask(x)
        x = self.rotate(x, angle=[0, 359])
        if random.random() > 0.5:
            x = self.scale(x, [0.5, 2], [0.5, 2])
        # x = self.align_img(x)
        x = self.crop_or_pad(x)
        return x


    def get_new_roi(self, mask):
        """

        :param mask: (ndarray.uint8)[Height, Width]
        :return:
        """
        # 增加灰度图和二值图的判断
        # 存在20-230灰度值像素则认定为灰度图
        _, mask_dist = cv.threshold(mask, 20, 255, cv.THRESH_TOZERO)
        _, mask_dist = cv.threshold(mask_dist, 230, 255, cv.THRESH_TOZERO_INV)
        if np.count_nonzero(mask_dist) < 5:
            # 二值图处理
            # 1.mask增强
            mask = self.transform_bit(mask)
            # 2.灰度赋值
            mask = mask.astype(np.float32) / 255
            low = np.random.randint(0, 150)
            high = low + np.random.randint(50, 105)
            mask = self.gray_mask_method[np.random.randint(0, len(self.gray_mask_method) - 1)](mask, low, high)
            if np.random.random() < 0.7:
                # 平滑随机噪声
                k = np.random.randint(1, 5) * 2 + 1
                cv.GaussianBlur(mask, (k, k), k, mask)
        else:
            # 灰度图处理
            # 增强
            mask = self.transform_gray(mask)
            scale = 0.8 + 0.4 * np.random.rand()
            offset = np.random.randint(-10, 10)
            # 随机线性变换
            cv.convertScaleAbs(mask, mask, scale, offset)
        mask = mask.astype(np.uint8)
        # if mask.shape != self.size:
        #     cv.imshow("1", mask)
        #     cv.imshow("2", self.crop_or_pad(mask))
        #     cv.waitKey(0)

        return mask


    def get_shape_mask(self, x):
        if np.count_nonzero(x) < 20:
            return np.ones((np.random.randint(5, 15), np.random.randint(5, 15)), dtype=np.uint8) * 255
        Row = np.argwhere(np.sum(x, axis=0) != 0)
        Col = np.argwhere(np.sum(x, axis=1) != 0)
        x = x[np.min(Col): np.max(Col) + 1, np.min(Row): np.max(Row) + 1]
        # 控制像素数量
        while np.count_nonzero(x) > self.area * self.prosepect_pmax:
            scale = np.random.random()
            scale = scale if scale > 0.5 else 0.5
            x = cv.resize(src=x, dsize=(int(x.shape[1]*scale), int(x.shape[0]*scale)), interpolation=cv.INTER_NEAREST)
        return x

    # 旋转

    def rotate(self, x, angle=0):
        H, W = x.shape
        if isinstance(angle, list):
            assert len(angle) == 2
            angle = np.random.randint(angle[0], angle[1])

        x = np.pad(x, ((W//2, W//2), (H//2, H//2)), mode="constant", constant_values=0)
        H, W = x.shape
        m = cv.getRotationMatrix2D((W//2, H//2), angle, scale=1)
        x = cv.warpAffine(x, m, (x.shape[1], x.shape[0]))
        x = self.get_shape_mask(x)
        return x

    # 形态学处理
    def random_morphological_processing(self, x, k_size=3):
        if isinstance(k_size, list):
            k_size = np.random.randint(k_size[0], k_size[1])
        k_size = k_size // 2 * 2 + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        param = {"src": x, "kernel": element}
        param["op"] = self.morph_method[random.randint(0, len(self.morph_method) - 1)]
        y = cv.morphologyEx(**param)
        if np.sum(y)//255 < 10:
            return x
        return y

    # 放缩
    def scale(self, x, scaleX_factor=1, scaleY_factor=1):
        if isinstance(scaleX_factor, list):
            assert len(scaleX_factor) == 2
            scaleX_factor = scaleX_factor[0] + (scaleX_factor[1] - scaleX_factor[0]) * np.random.rand()
        if isinstance(scaleY_factor, list):
            assert len(scaleY_factor) == 2
            scaleY_factor = scaleY_factor[0] + (scaleY_factor[1] - scaleY_factor[0]) * np.random.rand()
        cv.resize(x, (int(x.shape[1] * scaleX_factor), int(x.shape[0] * scaleY_factor)), x,
                  interpolation=cv.INTER_LINEAR)
        return x

    # 回归尺寸
    # def align_img(self, x):
    #     # if np.random.random() < 0.2:
    #     #     x = self.resize(x)
    #     # else:
    #     #     x = self.crop_or_pad(x)
    #     #
    #     x = self.crop_or_pad(x)
    #     cv.threshold(x, 20, 255, cv.THRESH_BINARY, x)
    #     return x

    def resize(self, x):
        x = np.resize(x, self.size)
        return x

    def crop_or_pad(self, x):
        y = None
        cnt = 0
        while y is None or np.sum(y)//255 < 10:
            H = x.shape[0] - self.size[0]
            W = x.shape[1] - self.size[1]
            if H < 0:
                H = -H
                pad_top = random.randint(0, H)
                y = np.pad(x, ((pad_top, H - pad_top), (0, 0)), mode="constant", constant_values=0)
            else:
                crop_top = random.randint(0, H)
                y = x[crop_top: crop_top + self.size[0]]
            if W < 0:
                W = -W
                pad_left = random.randint(0, W)
                y = np.pad(y, ((0, 0), (pad_left, W - pad_left)), mode="constant", constant_values=0)
            else:
                crop_left = random.randint(0, W)
                y = y[:, crop_left: crop_left + self.size[1]]
            # crop有时只裁剪到黑色区域,此时直接resize
            if np.sum(y)//255 < 10:
                cnt += 1
                if cnt >= 5:
                    return np.resize(x, self.size).astype(np.uint8)
        return y

    # 随机mask灰度值
    def rand_mask(self, mask, low, high):
        gray_mask = np.random.randint(low, high, mask.shape) * mask
        return gray_mask

    def gauss_mask(self, mask, low, high):
        mask = self.get_shape_mask(mask)
        gauss_x = cv.getGaussianKernel(mask.shape[1], mask.shape[1])
        gauss_y = cv.getGaussianKernel(mask.shape[0], mask.shape[0])
        kyx = np.multiply(gauss_y, np.transpose(gauss_x))

        mask = mask * kyx
        Max = np.max(mask)
        Min = np.min(np.where(mask == 0, Max, mask))


        gray_mask = low + (mask - Min) / (Max - Min) * (high - low)
        gray_mask = np.where(gray_mask > 0, gray_mask, 0)
        gray_mask = self.crop_or_pad(gray_mask)
        return gray_mask

    def const_mask(self, mask, *args):
        gray_mask = mask * np.random.randint(0, 255)
        return gray_mask


# def genRandImg1(size,mask):
#     path = './gg'
#     picture_rand = os.listdir(path)
#     len_rand_picture = len(picture_rand)
#     x = random.randint(0, len_rand_picture - 1)
#     name_image = picture_rand[x]
#     picture = cv.imread(path + '/' + name_image, 0)
#     # print(type(picture))
#     picture = cv.resize(picture, (128,128))
#     # print(picture)
#     # _, mask_pict = cv.threshold(picture, 150, 255, cv.THRESH_BINARY)
#     #
#     # cv2.imshow('image',mask_pict)
#     # cv2.waitKey()
#     picture = picture.astype(np.float)
#     return picture


def get_new_image(img, gray_mask):
    gray_mask = gray_mask.astype(np.float32)
    mask = np.where(gray_mask > 0, 1, 0)
    # mask = np.where(gray_mask > 0, 255, 0)
    # mask1=mask.astype(np.uint8)
    # cv.imshow('mask',mask1)
    # cv.waitKey(5000)
    # cover
    if random.random() > 0.8:

        new_img = (img * (1 - mask) + gray_mask * mask)
    else:

        # new_img = (img * (1 - mask)) + gray_mask * mask * (255 - np.mean(img)) / 255
        new_img = (img * (1 - mask)) + mask * img * (1 + (gray_mask - 127.5) / 127.5)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img

def smooth_edge(new_img, mask):
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, element)
    mask_erode = cv.morphologyEx(mask, cv.MORPH_ERODE, element)
    mask_edge = ((mask_dilate - mask_erode) / 255).astype(np.float32)
    new_img_gauss = cv.GaussianBlur(new_img, (5, 5), 5)
    return (new_img * (1 - mask_edge) + new_img_gauss * mask_edge).astype(np.uint8)


class DefectiveGenerator(object):
    def __init__(self,dir_Database,shape_Img,Limit_ROI=(20,10000),withDatabase=True):
        """

        :param dir_Database:  缺陷ROI路径
        :param shape_Img:   图片大小[height,width]
        :param Limit_ROI:   ROI外接矩形大小[lower ,upper]
        :param withDatabase:   true:从硬盘读入ROI  false：算法生成ROI
        """
        self.dir_Database=dir_Database
        self.height_Img = shape_Img[0]
        self.width_Img=shape_Img[1]
        self.lowerLimit_ROI=Limit_ROI[0]
        self.upperLimit_ROI=Limit_ROI[1]
        #
        self.roi_transform = ROI_transforms()
        #从数据库读入ROI
        self.names_ROIs,self.num_ROIs=self.loadROIs(self.dir_Database)
        if self.num_ROIs<1:
            print("the dataset is empty!")


    def loadROIs(self,dir):
        # ROIs=os.listdir(dir)
        # 递归遍历文件
        ROIs = list()
        for root, dirs, files in os.walk(dir):
            # print(root)
            for file in files:
                if file.endswith(".bmp") or file.endswith(".PNG"):
                    ROIs.append(os.path.join(root, file))

        num_ROI=len(ROIs)
        print('采用本地ROI个数为{}'.format(num_ROI))
        return ROIs,num_ROI

    def genRandImg(self,size):
        mean=random.randint(-125,125)
        fluct=random.randint(1,100)
        low=mean-fluct  #+(mean-fluct<0)*abs(mean-fluct)
        high=mean+fluct   #-(mean+fluct>255)*abs(255-(mean+fluct))
        img=np.random.randint(low,high,size)
        img=img.astype(np.float)
        return img

    def genRandImg1(self,size):
        path = './gg'
        picture_rand = os.listdir(path)
        len_rand_picture = len(picture_rand)
        x = random.randint(0, len_rand_picture - 1)
        name_image = picture_rand[x]
        picture = cv.imread(path + '/' + name_image, 0)
        # print(type(picture))
        picture = cv.resize(picture, (128,128))
        # cv.imshow('image',picture)
        # cv.waitKey()
        # print(picture)
        # _, mask_pict = cv.threshold(picture, 150, 255, cv.THRESH_BINARY)
        #
        # cv.imshow('image',picture)
        # cv.waitKey(5000)
        # picture = picture.astype(np.float)
        # cv.imshow('image',picture)
        # cv.waitKey(5000)
        return picture

    def apply(self,img):
        ROI=self.randReadROI()
        # 灰度mask处理
        # 1.最小矩形提取
        # 2.随机旋转和放缩
        # 3.尺寸回归

        # 二值mask处理
        # roi增强
        # 1.最小矩形提取形状
        # 2.随机旋转和放缩
        # 3.形态学处理
        # 4.回归尺寸
        # 返回二值掩模图

        # 返回灰度roi
        ROI_new = self.roi_transform.get_new_roi(ROI)
        ROI_new = np.where(ROI_new > 0, 1, 0).astype(np.uint8)
        #
        # cv.imshow('mask',ROI_new)
        # cv.waitKey(5000)
        randd = random.randint(0, 1)
        if randd == 0:
            img_rand = self.genRandImg([self.height_Img, self.width_Img])
        else:
            img_rand = self.genRandImg1([self.height_Img, self.width_Img])
        img_new = img.astype(np.float)
        rand = random.randint(0, 1)
        if rand == 0:
            img_new = img_new * (1 - ROI_new) + img_rand * ROI_new
        else:
            img_new = img_new + img_rand * ROI_new
        #  img_new = img_new * (1 - ROI_new) + img_rand * ROI_new
        img_new = np.clip(img_new, 0, 255).astype(np.uint8)
        # ROI_new = (ROI_new * 255).astype(np.uint8)
        # cv.imshow('mask',img_new)
        # cv.imshow('mask', img_rand)
        # cv.waitKey(5000)
        return img_new, ROI_new

        # img_new = get_new_image(img, ROI_new)
        #
        # img_new = smooth_edge(img_new, ROI_new)
        # cv.imshow("img", img)
        # cv.imshow("ROI", ROI)
        # cv.imshow("ROI_new", ROI_new)
        # cv.imshow("img_new", img_new)
        # cv.waitKey(0)


        #
        # img_rand=self.genRandImg([self.height_Img, self.width_Img])
        # img_new=img.astype(np.float)


        # rand = np.random.randint(0, 1)

        # if rand==0:
        #    img_new=img_new*(1-ROI_new)+img_rand*ROI_new
        # else:
        #     img_new = img_new + img_rand * ROI_new
      #  img_new = img_new * (1 - ROI_new) + img_rand * ROI_new
      #   img_new=np.clip(img_new, 0, 255).astype(np.uint8)
      #   ROI_new=(ROI_new*255).astype(np.uint8)
      #   return img_new, ROI_new

    def randReadROI(self):

        while(1):
            rand = random.randint(0, self.num_ROIs - 1)
            name_Img = self.names_ROIs[rand]
            img_Label = cv.imread(name_Img, 0)
            cv.threshold(img_Label, 20, 255, cv.THRESH_TOZERO, img_Label)
            if np.sum(img_Label) > 5:
                return img_Label



    def randVaryROI(self,ROI):


        return ROI

    # def randMoveROI(self,ROI):
    #     #求图像的域的大小
    #     Height_Domain =  self.height_Img
    #     Width_Domain= self.width_Img
    #     #求ROI区域的坐标
    #     Rows,Cols = np.nonzero(ROI)
    #     #求ROI区域的外接矩形大小
    #     Width_ROI=np.max(Cols)-np.min(Cols)
    #     Height_ROI=np.max(Rows)-np.min(Rows)
    #     #随机设置ROI的起始坐标
    #     Row_Upleft=random.randint(0,Height_Domain-Height_ROI-1)
    #     Col_Upleft = random.randint(0, Width_Domain - Width_ROI-1)
    #     Rows=Rows-np.min(Rows)+Row_Upleft
    #     Cols=Cols-np.min(Cols)+Col_Upleft
    #     ROI_new=np.zeros([Height_Domain,Width_Domain])
    #     ROI_new[Rows,Cols]=1
    #     return ROI_new

    # def genRandImg(self,size):
    #     mean=random.randint(-125,125)
    #     fluct=random.randint(1,100)
    #     low=mean-fluct  #+(mean-fluct<0)*abs(mean-fluct)
    #     high=mean+fluct   #-(mean+fluct>255)*abs(255-(mean+fluct))
    #     img=np.random.randint(low,high,size)
    #     img=img.astype(np.float)
    #     #
    #     return img





class RepairDataset(BaseDataset):
    """
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size =  self.A_size  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform=self.get_transform()
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        # ztb1数据换为128, 128
        self.defectGen =DefectiveGenerator("../datasets/masks", (128, 128))
        self.phase=opt.phase
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
       # print(A_img.size)

       #  if self.phase=="train":
       #      B_img = Image.open(A_path).convert('L')
       #      # 正负采样:
       #      if np.random.random() < 0.5:
       #          B_img = self.transform(B_img)
       #      A_img,_ = self.defectGen.apply((np.array(B_img)))
       #      A_img=Image.fromarray(A_img)
       #      # apply image transformation
       #      A = self.transform_A(A_img)
       #      B = self.transform_B(B_img)
       #      return {'A': A, 'B': B, 'A_paths': A_path}

        if self.phase == "train":
            B_img = Image.open(A_path).convert('L')
            B_img = self.transform(B_img)
            if index % 2 == 0:
                A_img, _ = self.defectGen.apply((np.array(B_img)))
                # cv.imshow('image',A_img)
                # cv.waitKey(5000)
                A_img = Image.fromarray(A_img)
                # apply image transformation
                A = self.transform_A(A_img)
                B = self.transform_B(B_img)
                # print('缺陷样本')
                return {'A': A, 'B': B, 'A_paths': A_path}
            else:
                B = self.transform_B(B_img)
                # print('正样本')
                return {'A': B, 'B': B, 'A_paths': A_path}
        elif self.phase=="test":
            A_img = Image.open(A_path).convert('L')
            A = self.transform_A(A_img)
            # B_mask_path=A_path.replace("testA","mask")
            # dirname,fname=os.path.split(B_mask_path)
            # B_mask_path=os.path.join(dirname,"groundT_"+fname)
            # B_mask= Image.open(B_mask_path).convert('L')
            # B_mask = self.transform_A(B_mask)
            # 测试级无mask,使用原图
            B_mask = A
            return {'A': A, 'B': A, 'A_paths': A_path,'B_mask': B_mask}
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size


    def get_transform(self):
        import torchvision.transforms as transforms
        l=[]
        l.append(transforms.RandomHorizontalFlip())
        l.append(transforms.RandomVerticalFlip())
        l.append(transforms.Resize([128, 128]))
       # l.append(transforms.RandomResizedCrop( 256, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)))
        return   transforms.Compose(l)