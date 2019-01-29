PyTorch implementation of the article [Convolutional Neural Networks with Layer Reuse](https://128.84.21.199/pdf/1901.09615.pdf), codes and pretrained models.

# CNNs with Layer Reuse

<p align="center"><img src="https://github.com/okankop/CNN-layer-reuse/blob/master/pretrained/LRU-arch.jpg" align="middle" width="350" title="LRU architecture" /><figcaption>Fig. 1:  (a) Conventional design of CNNs,  (b) CNN design with layer reuse. Instead of stacking convolutional layers and feeding one layer’s output as input to another layer, we feed the output of a convolutional block as input to itself for N-times before passing it to the next block.</figcaption></figure></p>

## Paper Abstract

A convolutional layer in a Convolutional Neural Network (CNN) consists of many filters which apply convolution operation to the input, capture some special patterns and pass the result to the next layer. If the same patterns also occur at the deeper layers of the network, why wouldn't the same convolutional filters be used also in those layers? In this paper, we propose a CNN architecture, Layer Reuse Network (LruNet), where the convolutional layers are used repeatedly without the need of introducing new layers to get a better performance. This approach introduces several advantages: (i) Considerable amount of parameters are saved since we are reusing the layers instead of introducing new layers, (ii) the Memory Access Cost (MAC) can be reduced since reused layer parameters can be fetched only once, (iii) the number of nonlinearities increases with layer reuse, and (iv) reused layers get gradient updates from multiple parts of the network. The proposed approach is evaluated on CIFAR-10, CIFAR-100 and Fashion-MNIST datasets for image classification task, and layer reuse improves the performance by 5.14%, 5.85% and 2.29%, respectively.

## Running the Code

You can simply modify the --dataset and --model arguments to select one of the [lrunet, mobilenet, mobilenetv2, shufflenet, shufflenetv2] models and on of the [CIFAR10, CIFAR100, fashion-MNIST] datasets, respectively.

  ```shell
#### Some example training configurations
# Train 14-LruNet-1x on CIFAR10 with dropout 0.5
python main.py --model lrunet --dataset cifar10 --layer_reuse 14 --width_mult 1.0 --drop 0.5 --batch_size 256 --lr 0.1

# Train Shufflenet 0.5x (g=3) on fashion-MNIST
python main.py --model shufflenet --dataset fashionmnist --width_mult 0.5 --groups 3 --batch_size 256 --lr 0.1

# Train MobilenetV2 0.4x on CIFAR100
python main.py --model mobilenetv2 --dataset cifar100 --width_mult 0.4 --batch_size 256 --lr 0.1

# Resume training for 14-LruNet-1x on CIFAR10 using the scheckpoint "checkpoint/ckpt.t7" with learning rate of 0.01
python main.py --model lrunet --dataset cifar10 --layer_reuse 14 --width_mult 1.0 --drop 0.5 --batch_size 256 --lr 0.01 --resume_path checkpoint/ckpt.t7
  ```
  
### Learning rate adjustment
Learning rate (--lr) is manually changed during training:

- 0.1 for epoch [0,200)
- 0.01 for epoch [200,250)
- 0.001 for epoch [250,300)


