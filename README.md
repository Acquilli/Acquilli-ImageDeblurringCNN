# Acquilli-ImageDeblurringCNN
Deep learning model that restores sharp images from blurred inputs using convolutional neural networks.

## How the Deblurring Model Works 🧠 
This project uses a Convolutional Neural Network (CNN) to learn how to restore blurred images to a sharper form. The dataset consists of paired images: sharp originals (train_y) and their artificially blurred versions (train_x), created using Gaussian blur.

## Data Preparation
All images are resized to a fixed target size (100, 250) and normalized (scaled to values between 0 and 1). The sharp images are stored separately from their blurred counterparts. The blurred images are generated directly from the sharp originals using OpenCV's Gaussian blur.

## Train/Test Split & Preprocessing 
The dataset is split into training and testing sets (80/20). Images are reshaped into 4D tensors with shape (samples, height, width, channels) for input into the CNN.

## Model Architecture 
The model is a sequential CNN with: Three convolutional layers to extract and encode spatial features from the blurred image. Three transposed convolutional layers (deconvolution) to reconstruct the sharpened image. The final layer uses a sigmoid activation to produce pixel values in the range [0, 1].

## Training 
The model is compiled with the Adam optimizer and mean squared error (MSE) as the loss function. It is trained over 10 epochs using the blurred images as input and the original sharp images as ground truth.

## Inference 
Given a new blurred image, the trained model can predict a deblurred version by reconstructing fine image details.

## Outputs 
The trained model is saved in HDF5 format (.h5), along with its weights. During testing, visual results are displayed using matplotlib: original, blurred, and deblurred versions are shown side-by-side for comparison.

![deblur](https://github.com/user-attachments/assets/6c5647dd-b10f-4b13-acc4-97078921a726)
