# Kitchenware-classification-using-AI
Kitchware-classification-using-CNN-machine-learning-approach

# Kitchenware Classification Using CNN and Q-Learning

This project is a machine learning application that classifies kitchenware items using a Convolutional Neural Network (CNN) and an adaptive Q-learning algorithm. It is structured into three main components:

1. **Data Preprocessing** (`ImageProcess.py`)
2. **CNN Model Training** (`Main.py`)
3. **Self-Improving Algorithm with Q-Learning** (`Q_learning.py`)

## Project Overview

This project was developed as part of a machine learning class to explore object classification using deep learning techniques. The primary goal is to classify images of various kitchenware items, but the framework can easily be adapted to classify other types of images.

---

## Components

### 1. Data Preprocessing (`ImageProcess.py`)

The data preprocessing step prepares images for model training and classification. This script:
- Accepts a dataset of images for kitchenware classification.
- Can be adapted to other types of data by changing the data source path and categories.

#### Configuration
Update the following parameters to configure your dataset:

- Set the path to your image data by modifying the `Datasrc` variable:
  ```python
  Datasrc = 'C:/Users/...'
