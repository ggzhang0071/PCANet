timestamp=`date +%Y%m%d%H%M%S`


batch_size=10

for i in $(seq 1 6 2);
do
   #Training:
   python -m pdb run_PCANet.py  --gpu 3 --mode train --dataset "CIFAR10" --batchSize $i --Numlayers 2 2>&1  |tee Logs/PCANet_batchsize_${i}_train_${cl}_$timestamp.log
   #Testing:

   python -m pdb run_PCANet.py --gpu 3 --mode test --dataset "MNIST" --batchSize $i --Numlayers 2 2>&1  |tee Logs/PCANet_batchsize_${i}_test_${cl}_$timestamp.log  

done