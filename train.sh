
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 500 --gan_mode lsgan  --preprocess resize



#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 500 --gan_mode wgangp  --preprocess resize --no_flip




#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  basic --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 1000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02

#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG AE --netD  basic --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 1000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02

#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 5000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02

#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 5000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02 --residual


#
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 3000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02 --residual

#7
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \ 7
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 10000 --gan_mode lsgan  --preprocess resize --no_flip  --init_gain 0.02 --residual \
      # --continue_train --epoch_count 5000
#8
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 3000 --gan_mode lsgan  --preprocess resize --no_flip  --weight_decay_G 0.001 --residual \
#9
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 3000 --gan_mode lsgan  --preprocess resize --no_flip  \
#       --weight_decay_G 0.001  --lr_decay_iters 100 --residual

#10
#python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
#       --netG unet_256 --netD  basic --input_nc 1 --output_nc 1  --dataset_mode repair \
#       --batch_size 4 --n_epochs 3000 --gan_mode lsgan  --preprocess resize --no_flip  \
#       --weight_decay_G 0.001  --lr_decay_iters 100 --residual  --norm  batch

python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
       --batch_size 8 --n_epochs 10000 --gan_mode lsgan  --preprocess resize   \
       --weight_decay_G 0.001  --lr_decay_iters 100 --residual  --norm  instance


python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
       --batch_size 32 --n_epochs 20000 --gan_mode lsgan  --preprocess resize   \
       --weight_decay_G 0.001  --lr_decay_iters 100 --residual  --norm  batch


python train.py --dataroot ./datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB \
       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode repair \
       --batch_size 64 --n_epochs 100000 --gan_mode lsgan  --preprocess resize   \
       --weight_decay_G 0.001  --lr_decay_iters 100 --residual  --norm  batch --lambda_L1 100
















































