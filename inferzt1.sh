
python infer.py --dataroot ../datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 3100 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt  --norm  instance


 #python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 #--netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 #--load_iter 1000 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance

# python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
# --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
# --load_iter 1500 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance



#  python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
# --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
# --load_iter 2200 --norm instance   --checkpoints_dir   ./checkpoints/checkpoints-runzt --norm  instance
