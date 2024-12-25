# CIFAR-10 Image Classification with CNN

## Project Overview
This project involves developing a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. CIFAR-10 is a well-known dataset used in the computer vision community, consisting of 60,000 32x32 color images in 10 different classes. The goal of this project is to accurately classify these images into their respective categories.

## Model Architecture
The model is a simple CNN with the following layers:
- **Convolutional Layer 1**: Applies 6 filters of size 5x5.
- **Max Pooling Layer 1**: Uses a 2x2 pool size.
- **Convolutional Layer 2**: Applies 16 filters of size 5x5.
- **Max Pooling Layer 2**: Uses a 2x2 pool size.
- **Fully Connected Layer 1**: 120 neurons.
- **Fully Connected Layer 2**: 84 neurons.
- **Output Layer**: 10 neurons (one for each class).

## Technologies Used
- **Python**: Primary programming language.
- **PyTorch**: Used for building and training the neural network.
- **Torchvision**: Utilized for data transformation and loading the CIFAR-10 dataset.
- **Matplotlib**: For visualizing images and results.

## Data Preprocessing
Data is normalized using mean and standard deviation of [0.5, 0.5, 0.5] for RGB channels respectively. Data augmentation techniques like random crop and horizontal flip are applied to the training data to improve model generalization.

## Training
The model is trained using:
- **Optimizer**: SGD with a learning rate of 0.001 and momentum of 0.9.
- **Loss Function**: Cross-Entropy Loss.
- **Epochs**: 2 (Note: Increasing the number of epochs may improve accuracy).

## Results
Initial testing provided the following accuracies:
- **Plane**: 56.1%
- **Car**: 63.2%
- **Bird**: 46.1%
- **Cat**: 39.5%
- **Deer**: 52.7%
- **Dog**: 34.6%
- **Frog**: 75.0%
- **Horse**: 56.3%
- **Ship**: 61.0%
- **Truck**: 73.2%

Class-wise accuracies indicate room for improvement, particularly for classes with lower performance.

## Improvements
Future enhancements will include:
- Increasing the complexity of the CNN architecture.
- Extending the training duration.
- Implementing more advanced data augmentation.
- Experimenting with different batch sizes and optimizers.
- Utilizing regularization techniques like dropout and batch normalization.

## How to Run
Instructions on how to set up the environment and run the project are detailed below:
1. Install Python and necessary libraries (PyTorch, torchvision, matplotlib).
2. Clone this repository.
3. Run the training script to train the model:
4. Evaluate the model performance using:


For more detailed usage, refer to the documentation provided in the respective scripts.

