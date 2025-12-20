#!/bin/bash

# Start Ollama app
open /Applications/Ollama.app

# select vm
source ./ollama_env/bin/activate

# Open HTML file in default browser
open ChatApp/chat_ui.html

# Start FastAPI server
fastapi dev ChatApi/main.py