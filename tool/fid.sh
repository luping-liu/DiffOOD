for method in DDIM S-PNDM F-PNDM FON PF;
do
  echo $method
  mkdir -p ./temp/sample
  mpiexec -np 4 python main.py --runner sample --method $method --config pf_deep_cifar10.yml --model_path temp/models/pf_deep_cifar10.ckpt
  pytorch-fid ./temp/sample ~/llp/Datasets/fid_cifar10_train.npz --device cuda:3
  mv ./temp/sample ./temp/pf_deep/$method
done

for step in 24 28 32 36 40 44 48 52 56 60; do
  rm ./temp/sample/*
  torchrun --nproc_per_node 2 main.py --runner fid --method PNDM4 --sample_step 50 \
  --device cuda --config config/ddim_cifar10_cond.yml --image_path temp/sample \
  --model_path temp/train/ddim_cifar100_cond_1/ema-$240000.ckpt
  pytorch-fid temp/sample temp/fid/fid_cifar100_train.npz --device cuda:0
done

python main.py --runner fid --method PNDM4 --sample_step 50 \
  --device cuda --config config/ddim_cifar10_cond.yml --image_path temp/sample \
  --model_path temp/train/ddim_cifar100_cond_1/ema-$240000.ckpt