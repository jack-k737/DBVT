# -*- coding:utf-8 -*-
"""
Generate variable description embeddings using Bio_ClinicalBERT.
This script reads JSON files containing medical variable descriptions and 
encodes them into embeddings using a pre-trained clinical BERT model.
"""
import json
import torch
import os

# Bypass torch.load security check for older model formats
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

from transformers import AutoTokenizer, AutoModel

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Path to Bio_ClinicalBERT model
model_path = '/home/xiejun/llm/Bio_ClinicalBERT'

# Dataset configurations: (json_file, output_file)
datasets = {
    'physionet_2012': {
        'json_path': 'physionet2012.json',
        'output_path': 'physionet_2012_var_rep.pt'
    },
    'physionet_2019': {
        'json_path': 'physionet2019.json',
        'output_path': 'physionet_2019_var_rep.pt'
    },
    'mimic_iii': {
        'json_path': 'mimic.json',
        'output_path': 'mimic_iii_var_rep.pt'
    }
}

# Load tokenizer and model
print(f"Loading Bio_ClinicalBERT from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load model (environment variable bypasses security check)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()
print("Model loaded successfully!")

# Ensure output directory exists
os.makedirs('./kedgn_embeddings', exist_ok=True)

for dataset_name, config in datasets.items():
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}...")
    
    # Load JSON file
    json_path = config['json_path']
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get variable names and descriptions
    var_names = list(data.keys())
    descriptions = list(data.values())
    
    print(f"Number of variables: {len(var_names)}")
    print(f"Sample variable: {var_names[0]}")
    print(f"Sample description: {descriptions[0][:100]}...")
    
    # Generate embeddings
    with torch.no_grad():
        # Tokenize all descriptions
        tokens = tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get model outputs
        outputs = model(**tokens)
        
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Save embeddings
    output_path = config['output_path']
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path}")
    
    # Also save variable name to index mapping for reference
    var_to_idx = {name: idx for idx, name in enumerate(var_names)}
    mapping_path = output_path.replace('.pt', '_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(var_to_idx, f, indent=2)
    print(f"Saved variable mapping to {mapping_path}")

print(f"\n{'='*50}")
print("All embeddings generated successfully!")
