# coding=UTF-8
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import  DefectDetector
from unit import submit
import cv2

def vaild(epoch):
    opt = TestOptions().parse()  # get test options
    opt.eval=True
    opt.load_iter=epoch
    opt.checkpoints_dir='./checkpoints/checkpoints-runzt'
    opt.name='fabric_pix2pix'
    opt.dataroot='../datasets/fabric'
    opt.direction='AtoB'
    opt.model='pix2pix'
    opt.netG='unet_256'
    opt.input_nc=1
    opt.output_nc=1
    opt.dataset_mode='repair'
    opt.preprocess='resize'
    opt.residual=True
    opt.norm='instance'
    opt.n_epochs=3000
    # opt.lr_policy='cosine'
    opt.continue_train=True
    # print('!!!!!!!!!!!')
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    kwargs = {
        "thred_dyn": 50,
        "ksize_dyn": 100,
        "ksize_close": 30,
        "ksize_open": 3,
    }
    detector = DefectDetector(**kwargs)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        print(i)
        paths = data["A_paths"]
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        image_batch = visuals["real_A"]
        reconst_batch = visuals["fake_B"]
        # B_mask=data["B_mask"]

        image_batch = image_batch.detach().cpu().numpy()
        # label_batch = B_mask.detach().cpu().numpy()
        reconst_batch = reconst_batch.detach().cpu().numpy()
        # batchs=detector.apply(image_batch,label_batch,reconst_batch)
        batchs = detector.apply(image_batch, image_batch, reconst_batch)
        for idx, path in enumerate(paths):
            visual_imgs = []
            for batch in batchs:
                visual_imgs.append(batch[idx].squeeze())
            img_visual = detector.concatImage(visual_imgs, offset=None)
            # print(img_visual.size)
            visualization_dir = opt.checkpoints_dir + "/infer_epoch{}/".format(opt.load_iter)
            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir)
            img_visual.save(visualization_dir + "_".join(path.split("/")[-2:]))


def main():
    opt = TestOptions().parse()  # get test options
    gen_jison = False
    if gen_jison:
        vis = False
        opt.num_test=16000
    else:
        vis = True
        opt.num_test = 300
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    kwargs = {
        "thred_dyn": 50,
        "ksize_dyn": 100,
        "ksize_close": 30,
        "ksize_open": 3,
    }
    detector = DefectDetector(**kwargs)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        print(i)
        paths = data["A_paths"]
        # print(paths)
        # print(len(paths))
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        image_batch = visuals["real_A"]
        reconst_batch = visuals["fake_B"]
        # B_mask=data["B_mask"]

        image_batch = image_batch.detach().cpu().numpy()
        # label_batch = B_mask.detach().cpu().numpy()
        reconst_batch = reconst_batch.detach().cpu().numpy()
        # batchs=detector.apply(image_batch,label_batch,reconst_batch)
        batchs = detector.apply(image_batch, image_batch, reconst_batch)



        image_high=128
        image_width=128
        image_name=paths[0].split('/')[-1].split('.')[0]
        image=batchs[-1].squeeze()
        # print(image.shape)
        # cv2.imshow('image',image)
        # cv2.waitKey(3)
        submit(image_high,image_width,image_name,image)
        if vis:
            for idx, path in enumerate(paths):
                visual_imgs = []
                for batch in batchs:
                    visual_imgs.append(batch[idx].squeeze())
                img_visual = detector.concatImage(visual_imgs, offset=None)
                # print(img_visual.size)
                visualization_dir = opt.checkpoints_dir + "/infer_epoch{}/".format(opt.load_iter)
                if not os.path.exists(visualization_dir):
                    os.makedirs(visualization_dir)
                img_visual.save(visualization_dir + "_".join(path.split("/")[-2:]))


if __name__ == '__main__':
    main()
