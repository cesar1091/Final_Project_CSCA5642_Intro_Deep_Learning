# Final_Project_CSCA5642_Intro_Deep_Learning
Final project of the course CSCA5642 - Introduction to Deep Learning

In this project, we will explore the use of deep learning techniques for image classification. The goal is to build a model that can accurately classify images of vehicles based on their damage status. We will use a dataset of vehicle images with different levels of damage and train a convolutional neural network (CNN) to perform the classification task.

## Project Overview

The project consists of the following main components:
- Data Preprocessing: We will preprocess the dataset by resizing the images, normalizing pixel values, and splitting the data into training and validation sets.
- Model Architecture: We will design a CNN architecture suitable for image classification tasks. The model will consist of convolutional layers, pooling layers, and fully connected layers.
- Model Training: We will train the CNN model using the training dataset and validate its performance on the validation dataset. We will also implement techniques such as data augmentation to improve model generalization.
- Model Evaluation: We will evaluate the trained model's performance using metrics such as accuracy, precision, recall, and F1-score. We will also visualize the model's predictions on sample images.
- Model Deployment: We will deploy the trained model using Flask, allowing users to upload images and receive predictions on the damage status of vehicles.

## Dataset

The dataset used for this project is the "VEHICULE DAMAGE INSURANCE VERIFICATION" dataset from Kaggle. The dataset contains images of vehicles with different levels of damage, along with labels indicating the damage status. The dataset is divided into training and validation sets.
- Dataset Size: The dataset contains a total of 10,000 images, with 8,000 images for training and 2,000 images for validation.
- Image Size: The images are of varying sizes, but we will resize them to a uniform size of 224x224 pixels for input to the CNN model.
- Image Format: The images are in JPEG format and are stored in a directory structure based on their labels.
- Labels: The labels for the images are six classes representing different levels of damage: "No Damage", "Minor Damage", "Moderate Damage", "Severe Damage", "Total Loss", and "Unknown".

The link to the dataset is provided below:

Dataset: [VEHICULE DAMAGE INSURANCE VERIFICATION](https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest/data)

## Requirements

- Python 3.x
- TensorFlow 2.x    
- Keras
- NumPy
- OpenCV
- Matplotlib
- Flask
- scikit-learn
- Pillow
- tqdm
- requests
