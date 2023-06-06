import json
import os.path
import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

json_file = 'embeddings.json'


def load_embeddings(embeddings_file=json_file):
    # Check if embeddings file exists, create it if not
    if not os.path.isfile(embeddings_file):
        with open(embeddings_file, 'w') as f:
            json.dump({}, f)

    # Load embeddings from JSON file
    with open(embeddings_file, 'r') as f:
        return json.load(f)


def save_embeddings(embeddings, tool_name):
    with open(f'embeddings-{tool_name}.json', 'w') as f:
        # Convert the embeddings to lists before outputting
        embeddings_copy = embeddings.copy()
        for example in embeddings_copy:
            example_embedding = example['embedding']
            example['embedding'] = example_embedding if example_embedding is not None else None
        json.dump(embeddings_copy, f, indent=4)
    print(f"Embeddings saved to embeddings-{tool_name}.json file.")


def embed_string(string):
    input_ids = torch.tensor([tokenizer.encode(string)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Shape: [batch_size, sequence_length, hidden_size]
        embedding = torch.mean(last_hidden_states, dim=1)  # Take the mean of the sequence to get a single vector
    return embedding.tolist()[0]


def add_tool(embeddings, tool_name, testing=False):
    examples = []
    while True:
        example = None

        if testing:
            example = "Find cats"
        else:
            example = input("Enter an example search term (or 'done' to finish): ")
            if example == 'done':
                break

        example_embedding = embed_string(example)
        if example_embedding is not None:
            examples.append({'text': example, 'embedding': example_embedding})
        else:
            print("Skipping invalid embedding for example:", example)

        if testing:
            break
        
    if examples:
        module_name = input("Enter the module name: ")
        module_description = input("Enter the module description: ")
        module_argument = input("Enter the module argument: ")

        if tool_name in embeddings:
            embeddings[tool_name]['examples'].extend(examples)
        else:
            embeddings[tool_name] = {
                'module': {
                    'name': module_name,
                    'description': module_description,
                    'argument': module_argument
                },
                'embeddings': examples
            }
        print(f"{len(examples)} examples added to {tool_name}.")
    else:
        print(f"No valid example embeddings found for {tool_name}. Skipping tool.")

    return embeddings


def list_tools(embeddings):
    for tool in embeddings.values():
        print(tool['module']['name'])
        for example in tool['embeddings']:
            print(f"  {example['text']}")


def save_embeddings(embeddings, tool_name, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/module-{tool_name}.json"
    
    if os.path.isfile(file_name):
        # Load existing embeddings from file
        with open(file_name, 'r') as f:
            existing_embeddings = json.load(f)
    else:
        existing_embeddings = []

    # Merge existing embeddings with new embeddings
    new_embeddings = embeddings[tool_name]['embeddings']
    combined_embeddings = existing_embeddings + new_embeddings
    
    # Convert the embeddings to lists before outputting
    embeddings_copy = combined_embeddings.copy()
    for example in embeddings_copy:
        example_embedding = example['embedding']
        example['embedding'] = example_embedding if example_embedding is not None else None
    
    # Save combined embeddings to file
    with open(file_name, 'w') as f:
        json.dump({
            'module': embeddings[tool_name]['module'],
            'embeddings': embeddings_copy
        }, f, indent=4)
    print(f"Embeddings saved to {file_name} file.")


def run_prompt():
    embeddings = load_embeddings()
    while True:
        command = input("Enter a command (add, list, save, quit): ")
        if command == 'add':
            tool_name = input("Enter a tool name: ")
            add_tool(embeddings, tool_name)
        elif command == 'list':
            list_tools(embeddings)
        elif command == 'save':
            for tool_name in embeddings:
                save_embeddings(embeddings, tool_name)
            break
        elif command == 'quit':
            save_response = input("Do you want to save your changes before exiting? (y/n) ")
            if save_response.lower() == 'y':
                for tool_name in embeddings:
                    save_embeddings(embeddings, tool_name)
            break
        else:
            print("Invalid command. Please try again.")


if __name__ == "__main__":
    run_prompt()
