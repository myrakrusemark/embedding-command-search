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

while True:
    # Accept goal as input from the user
    goal = input("Enter your goal: ")

    # Convert goal to a list of word embeddings
    word_vecs = [model[word] for word in goal.split() if word in model]

    # Compute the mean of the word embeddings as the goal embedding
    goal_vec = np.mean(word_vecs, axis=0)

    # Find the tool with the smallest cosine distance to the goal
    best_tool = None
    min_distance = np.inf
    for tool, tool_data in embeddings.items():
        examples = tool_data['examples']
        tool_vec = embed_strings([example['text'] for example in examples])
        if tool_vec is not None:
            distance = cosine(goal_vec, tool_vec)
            if distance < min_distance:
                best_tool = tool
                min_distance = distance

    # Find the example search term with the smallest cosine distance to the goal
    best_example = None
    min_distance = np.inf
    for example_data in embeddings[best_tool]['examples']:
        example_vec = np.array(example_data['embedding'])
        distance = cosine(goal_vec, example_vec)
        if distance < min_distance:
            best_example = example_data['text']
            min_distance = distance

    # Output the best tool and example search term for achieving the goal
    print("Goal:", goal)
    print("Best tool:", best_tool)
    print("Best matching term:", best_example)
