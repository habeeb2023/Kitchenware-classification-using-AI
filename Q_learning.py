import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
import numpy as np
import random

# Define the actions (class predictions)
ACTIONS = ['forks', 'cups', 'bowls', 'plates']

# Q-learning parameters
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
EPISODES = 1000

# Initialize the Q-table with zeros
Q_table = np.zeros((len(ACTIONS), len(ACTIONS)))

# Load the pre-trained model
model = tf.keras.models.load_model('64x3-CNN.model')

# Function to preprocess an image
def preprocess_image(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Function to choose an action (class prediction) based on epsilon-greedy policy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)  # Explore
    else:
        return ACTIONS[np.argmax(Q_table[state])]

# Function to update the Q-values based on rewards
def update_q_value(state, action, reward, next_state):
    max_next_action_value = np.max(Q_table[next_state])
    current_q_value = Q_table[state][ACTIONS.index(action)]
    new_q_value = (1 - LEARNING_RATE) * current_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_action_value)
    Q_table[state][ACTIONS.index(action)] = new_q_value

# Function to predict and reward the agent
def predict_and_reward():
    file_path = filedialog.askopenfilename()
    if file_path:
        state = ACTIONS.index(random.choice(ACTIONS))
        done = False

        while not done:
            # Choose an action based on epsilon-greedy policy
            epsilon = 0.1  # Exploration rate
            action = choose_action(state, epsilon)

            # Simulate the environment (in this case, use the model to predict)
            prediction = model.predict([preprocess_image(file_path)])
            predicted_class = ACTIONS[int(prediction[0][0])]

            # Reward the agent based on the prediction
            if action == predicted_class:
                reward = 1  # Correct prediction
                done = True  # End the episode
            else:
                reward = 0  # Incorrect prediction

            # Update the Q-value only for correct predictions
            if reward == 1:
                next_state = ACTIONS.index(predicted_class)
                update_q_value(state, action, reward, next_state)

            result_label.config(text=f'Predicted Class: {predicted_class}\nReward: {reward}')

# Function to display the Q-table
def display_q_table():
    q_table_window = tk.Toplevel(root)
    q_table_window.title("Q-Table")

    q_table_label = tk.Label(q_table_window, text="Q-Table")
    q_table_label.pack()

    q_table_text = tk.Text(q_table_window, height=len(ACTIONS), width=len(ACTIONS) * 8)
    q_table_text.pack()

    q_table_text.insert(tk.END, "Actions: " + ", ".join(ACTIONS) + "\n\n")
    for i, row in enumerate(Q_table):
        q_table_text.insert(tk.END, f'State {ACTIONS[i]}: {row}\n')

# Create the main window
root = tk.Tk()
root.title("Image Classification with Q-learning")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=predict_and_reward)
select_button.pack(pady=10)

# Create a button to display the Q-table
q_table_button = tk.Button(root, text="Display Q-Table", command=display_q_table)
q_table_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

# Run the GUI application
root.mainloop()
