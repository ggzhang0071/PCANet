timestamp=`date +%Y%m%d%H%M%S`

#python run_mnist.py  --mode train --dataset "CIFAR10" 2>&1  |tee run_mnist_${i}_train_${cl}_$timestamp.log


python run_mnist.py  --mode test --dataset "CIFAR10" 2>&1  |tee run_mnist_${i}_test_${cl}_$timestamp.log