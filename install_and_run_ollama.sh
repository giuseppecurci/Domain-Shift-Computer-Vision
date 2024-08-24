#!/bin/bash

# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server in the background
ollama serve &

# Pull the specified model
# To add more models: ollama pull [model_name]
# List of available models can be found at https://ollama.com/library 
ollama pull llama3.1

# Display the list of availavle model
# Used as a success message when the script completes
ollama list

# Run the specified model
# Execute the following command ONLY if you want to run ollama from terminal
# ollama run llama3.1