import numpy as np
import json
import os.path
import gensim.downloader as api
from scipy.spatial.distance import cosine

# Check if embeddings file exists, create it if not
if not os.path.isfile('embeddings.json'):
    with open('embeddings.json', 'w') as f:
        json.dump({}, f)

# Load embeddings from JSON file
with open('embeddings.json', 'r') as f:
    embeddings = json.load(f)

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Define a function to embed a list of strings using Word2Vec
def embed_strings(strings):
    word_vecs = []
    for string in strings:
        words = string.split()
        word_vecs.extend([model[word] for word in words if word in model])
    if not word_vecs:
        return None
    embedding = np.mean(word_vecs, axis=0)
    if np.isnan(embedding).any():
        return None
    return embedding.tolist() # convert ndarray to list


# Accept inputs from the user
while True:
    tool_name = input("Enter a tool name (or 'quit' to exit): ")
    if tool_name == 'quit':
        break
    if tool_name in embeddings:
        print("Tool already exists in embeddings. Adding new examples.")
        examples = embeddings[tool_name]['examples']
    else:
        examples = []
    while True:
        example = input("Enter an example search term (or 'done' to finish): ")
        if example == 'done':
            break
        example_embedding = embed_strings([example])
        if example_embedding is not None:
            examples.append({'text': example, 'embedding': example_embedding})
        else:
            print("Skipping invalid embedding for example:", example)
    if examples:
        if tool_name in embeddings:
            embeddings[tool_name]['examples'].extend(examples)
        else:
            embeddings[tool_name] = {
                'tool': {'text': tool_name},
                'examples': examples
            }
    else:
        print("Skipping tool with no valid example embeddings:", tool_name)


# Output the embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    # Convert the embeddings to lists before outputting
    embeddings_copy = embeddings.copy()
    for tool in embeddings_copy.values():
        for example in tool['examples']:
            example_embedding = example['embedding']
            example['embedding'] = example_embedding if example_embedding is not None else None
    json.dump(embeddings_copy, f, indent=4)
    
# Print the contents of the JSON file
with open('embeddings.json', 'r') as f:
    print(f.read())
