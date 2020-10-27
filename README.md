# Fashion MNIST

In this project, we explore a Deep Neural Network classification problem using the [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist).


<p align="center">
  <img width="460" height="300" src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/embedding.gif">
</p>

Fashion-MNIST is a collection of images of fashion items, and it can be used as drop-in replacement for the original MNIST digits dataset. It shares the same image size (28x28x1 grayscale) and has 60,000 training and 10,000 testing images. It has 10 categories of output labels. It was created as a replacement for the standard Digits-MNIST dataset, due to the simplicity of the latter (Convolutional Neural Networks (CNNs) easily achieve 99.7% accuracy on MNIST, and it can not represent modern computer vision tasks).

### Requirements

* Python 3.x
* Installing the following packages:
	* Tensorflow, h5py, Yaml, Numpy
 
### Train the classifier

- Clone this repo to your local machine using `git clone ...`.
- Edit the `config.yml` file with your preferences (you can leave the default settings).
- You can edit the network structure in `utils/deepnetwork.py`.
- Train the classifier using the `main.py` script.

### Results

This project represents a baseline to the Fashion-MNIST problem, to show that a simple CNN achieves over 90% accuracy without any optimization.
Folder `models` contains the trained model using the `main.py` script.

Here we show a prediction of a random sample of 50 images, using the resultant model (red title represents incorrect predictions):
<p align="center">
  <img width="460" height="300" src="https://miro.medium.com/max/875/1*UGwDzdgQ1ygzUOQbsNrp2w.png">
</p>

## License

- **MIT license**
