import os #Provides a way to interact with the file system.
import cv2 #Used for reading and processing images.
import numpy as np #Used for numerical operations and handling arrays.
import pickle #Used for serializing and deserializing Python objects.
import random

# Define the source directory and categories of images stored
Datasrc = 'C:/Users/Rugged_Storm/Desktop/Task 8/data'
CATEGORIES = ['plates', 'cups', 'bowls', 'plates']
IMG_SIZE = 100  # Increase the image size for better feature representation

# Initialize an empty list to store training data
training_data = []

# Function to create the training dataset
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(Datasrc, category)  # Create the path to the current category
        class_num = CATEGORIES.index(category) # Assign a numerical label to the current category
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Read the image in grayscale
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # Resize the image to the specified size
                training_data.append([new_array, class_num])  # Append the resized image and its label to the training data
            except Exception as e:
                pass

# Call the function to create the training dataset
create_training_data()

# Shuffle the training data to ensure randomness
random.shuffle(training_data)

X = []  # Image data
y = []  # Labels

# Separate image data and labels from the training data
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert image data and labels to NumPy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


# Save the NumPy arrays to pickle files for later use in training
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
