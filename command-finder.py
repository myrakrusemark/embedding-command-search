import numpy as np
import json
from scipy.spatial.distance import cosine
import gensim.downloader as api

# Load embeddings from JSON file
with open('embeddings.json', 'r') as f:
    embeddings = json.load(f)

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Define a function to embed a list of strings using Word2Vec
def embed_strings(strings):
    word_vecs = [model[word] for word in strings if word in model]
    return np.mean(word_vecs, axis=0)

# Output available commands
print("Available commands:")
for command in embeddings.keys():
    print(f"- {command}")

while True:
    # Accept goal as input from the user
    goal = input("Enter your task: ")

    # Convert goal to a list of word embeddings
    word_vecs = [model[word] for word in goal.split() if word in model]

    if not word_vecs:
        print("Goal contains no valid words. Please enter a different goal.")
        continue

    # Compute the mean of the word embeddings as the goal embedding
    goal_vec = np.mean(word_vecs, axis=0)

    # Find the command with the smallest cosine distance to the goal
    best_command = None
    next_best_command = None
    best_command_distance = np.inf
    next_best_command_distance = np.inf
    for command, command_data in embeddings.items():
        examples = command_data['examples']
        command_vecs = [np.array(example['embedding']) for example in examples]
        command_mean_vec = np.mean(command_vecs, axis=0)

        if np.isnan(command_mean_vec).any():
            continue

        distance = cosine(goal_vec, command_mean_vec)

        if distance < best_command_distance:
            next_best_command_distance = best_command_distance
            next_best_command = best_command
            best_command_distance = distance
            best_command = command
        elif distance < next_best_command_distance:
            next_best_command_distance = distance
            next_best_command = command

        # Compute percentage match for each command
        command_match = 100 * (1 - distance)
        print(f"Command: {command}, Match: {command_match:.2f}%")

    # Output the best command for achieving the goal
    print("Task:", goal)

    if not np.isnan(best_command_distance):
        best_command_match = 100 * (1 - best_command_distance)
        print(f"Best command ({best_command_match:.2f}%): {best_command}")

    # Output the next best command for achieving the goal
    if not np.isnan(next_best_command_distance):
        next_best_command_match = 100 * (1 - next_best_command_distance)
        print(f"Next best command ({next_best_command_match:.2f}%): {next_best_command}")

