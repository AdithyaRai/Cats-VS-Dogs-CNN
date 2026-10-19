# Cats vs. Dogs Image Classification using Transfer Learning (MobileNetV2)

This project implements a binary image classifier to distinguish between Cats and Dogs using TensorFlow and Keras, comparing a custom Convolutional Neural Network (CNN) built from scratch against a Transfer Learning model using MobileNetV2.

## 🚀 Project Highlights
- **Dataset**: ~2,000 images divided into training and testing splits.
- **Custom CNN**: Constructed a 4-block deep CNN with Max Pooling, Flattening, and Dropout regularization.
- **Transfer Learning**: Integrated **MobileNetV2** pre-trained on ImageNet as a feature extractor, with a customized classifier head.
- **Data Augmentation**: Applied dynamic random flips, rotations, and zooms to minimize overfitting.
- **Optimization**: Used performance optimizations like `tf.data` caching, prefetching (`AUTOTUNE`), and callbacks (`EarlyStopping` & `ModelCheckpoint`).

## 📊 Performance Comparison
- **Custom CNN**: Reached baseline accuracy using custom convolutional blocks.
- **MobileNetV2 Transfer Learning**: Achieved vastly superior accuracy and faster convergence, leveraging ImageNet's pre-trained weights.

## 🛠️ Tech Stack
- Python, TensorFlow, Keras, Matplotlib, NumPy
