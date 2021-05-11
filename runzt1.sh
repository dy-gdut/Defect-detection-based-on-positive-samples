
python train.py --dataroot ../datasets/fabric --name fabric_pix2pix --model pix2pix --direction AtoB  --dataset_mode repair \
       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --residual  --norm  instance \
       --batch_size 10 --n_epochs 3000 --gan_mode lsgan  --preprocess resize   \
       --weight_decay_G 0.0001  --lr_decay_iters 300     --checkpoints_dir  ./checkpoints/checkpoints-runzt \
       --lambda_L1  200 --lr_G 0.0002

#python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
# --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
# --load_iter 2000 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt  --norm  instance


#python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
# --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
# --load_iter 2500 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance

 #python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 #--netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 #--load_iter 3000 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance



 # python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 #--netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 #--load_iter 200 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance
