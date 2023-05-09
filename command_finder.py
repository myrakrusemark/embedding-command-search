import json
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings():
    embeddings = {}
    for filename in os.listdir('./output'):
        if not filename.endswith('.json'):
            continue
        with open(f'./output/{filename}', 'r') as f:
            command_name = filename[:-5]
            command_data = json.load(f)
            embeddings[command_name] = {
                'embeddings': [example['embedding'] for example in command_data],
                'examples': [example['text'] for example in command_data]
            }
    return embeddings



def load_bert_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_strings(strings, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(string) for string in strings])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Shape: [batch_size, sequence_length, hidden_size]
        embeddings = torch.mean(last_hidden_states, dim=1)  # Take the mean of the sequence to get a single vector
    return embeddings.tolist()

def compute_distance(goal_vec, command_mean_vec):
    distance = cosine(goal_vec, command_mean_vec)
    return distance

def find_best_command(goal_vec, embeddings):
    best_command = None
    best_command_distance = float("inf")
    next_best_command = None
    next_best_command_distance = float("inf")

    for command_name in embeddings.keys():
        command_distance = 0.0
        command_data = embeddings[command_name]['embeddings']
        for emb in command_data:
            command_distance -= cosine_similarity(np.array(goal_vec).reshape(1, -1), np.array(emb).reshape(1, -1))
        command_distance /= len(command_data)

        if command_distance < best_command_distance:
            next_best_command_distance = best_command_distance
            best_command_distance = command_distance
            next_best_command = best_command
            best_command = command_name
        elif command_distance < next_best_command_distance:
            next_best_command_distance = command_distance
            next_best_command = command_name

    return best_command, best_command_distance, next_best_command, next_best_command_distance





def print_available_commands(embeddings):
    print("Available commands:")
    for command in embeddings.keys():
        print(f"- {command}")

def main():
    # Load embeddings from files
    embeddings = load_embeddings()

    # Load pre-trained BERT model and tokenizer
    tokenizer, model = load_bert_model('bert-base-uncased')

    # Output available commands
    print_available_commands(embeddings)

    while True:
        # Accept goal as input from the user
        goal = input("Enter your task: ")

        # Convert goal to a list of sentence embeddings
        goal_vecs = embed_strings([goal], tokenizer, model)

        if not goal_vecs:
            print("Goal contains no valid words. Please enter a different goal.")
            continue

        # Use the first sentence embedding as the goal embedding
        goal_vec = goal_vecs[0]

        # Find the command with the smallest cosine distance to the goal
        (best_command, best_command_distance, next_best_command,
        next_best_command_distance) = find_best_command(goal_vec, embeddings)

        # Output the best command for achieving the goal
        print("Task:", goal)

        if not np.isnan(best_command_distance):
            best_command_match = 100 * (1 - best_command_distance)
            print(f"Best command ({best_command_match.item():.2f}%): {best_command}")

        # Output the next best command for achieving the goal
        if not np.isnan(next_best_command_distance):
            next_best_command_match = 100 * (1 - next_best_command_distance)
            print(f"Next best command ({next_best_command_match.item():.2f}%): {next_best_command}")

if __name__ == '__main__':
    main()
