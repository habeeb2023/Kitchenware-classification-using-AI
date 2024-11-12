
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

## Dataset and Preprocessing Details

The dataset consists of images of kitchenware items, organized into categories (e.g., plates, cups, bowls, and spoons). You can adapt the dataset by:
- Adding or removing categories in the `CATEGORIES` list in `ImageProcess.py`.
- Specifying the dataset's path using the `Datasrc` variable.

The preprocessing script (`ImageProcess.py`) converts images into a structured format suitable for CNN training. The script:
1. Loads images from the specified dataset path.
2. Labels each image according to its category.
3. Resizes and normalizes images for compatibility with the CNN model.
4. Saves processed data to two files: `X` (features) and `y` (labels).

This step ensures that the CNN receives standardized data, which is crucial for achieving accurate classification results.

---

## CNN Model Architecture

The CNN model in `Main.py` is designed with layers that help the network learn complex patterns in image data. The architecture includes:
- **Convolutional Layers**: Extract features from images by detecting edges, textures, and shapes.
- **Pooling Layers**: Reduce the dimensionality of feature maps to make the model computationally efficient.
- **Fully Connected Layers**: Enable the model to learn associations between extracted features and output categories.

The model's training parameters, such as learning rate and batch size, are adjustable to improve performance.

---

## Q-Learning Algorithm and Self-Improvement

The Q-learning algorithm implemented in `Q_learning.py` enables the model to adapt based on new data over time. This process involves:
1. Predicting the class of input images.
2. Calculating the reward based on prediction accuracy.
3. Updating the model based on the reward to improve future predictions.

The algorithm utilizes reinforcement learning principles, allowing the model to self-optimize through trial and error. With each iteration, the model learns from its mistakes, enhancing classification accuracy.

### Graphical User Interface (GUI)

The GUI in `Q_learning.py` is designed to make the self-improving prediction process user-friendly. The interface provides:
- An input section for uploading images.
- A display area for prediction results.
- Options for users to manually correct misclassified items, aiding in the Q-learning feedback loop.

This user-friendly interface makes it easier to visualize the classification process and interact with the model.

---

## Example Usage

1. **Data Preprocessing**:  
   Run `ImageProcess.py` to preprocess the images, and ensure `X` and `y` files are generated.

   ```bash
   python ImageProcess.py
