#!/bin/bash

# Name of the virtual environment directory
VENV_NAME=".research-agent-env"

# Python packages to include in requirements.txt
PACKAGES=(
    "anthropic"              # For Claude API
    "python-dotenv"          # For managing API keys
    "requests"               # For downloading papers
    "pypdf"                  # For reading PDF files (replaces PyPDF2)
    "sentence-transformers"  # For generating text embeddings
    "faiss-cpu"              # For local vector store (efficient similarity search)
    "numpy"                  # Numerical operations, often a dependency for ML libs
    "nltk"                   # For text processing (e.g., sentence tokenization for chunking)
    # "pymupdf"              # Optional: Alternative PDF reader, very robust. Might need separate install steps for some systems.
    # "chromadb"             # Optional: Alternative vector store, good for persistence and ease of use.
    # "gradio"               # Optional: For building a quick UI later
    # "streamlit"            # Optional: Alternative for UI building
)

# --- Script Start ---

echo "Setting up Python environment in current directory..."

# 1. Check if we are in a directory (sanity check)
if [ ! -d "." ]; then
    echo "Error: Please run this script from within your project directory."
    exit 1
fi

# 2. Create Python virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating Python virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Make sure python3 and venv are installed."
        exit 1
    fi
    echo "Virtual environment '$VENV_NAME' created."
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# 3. Create requirements.txt
echo "Creating/updating requirements.txt..."
REQUIREMENTS_FILE="requirements.txt"
touch "$REQUIREMENTS_FILE" # Clear the file or create it if it doesn't exist

for pkg in "${PACKAGES[@]}"; do
    echo "$pkg" >> "$REQUIREMENTS_FILE"
done
echo "requirements.txt created with the following packages:"
cat "$REQUIREMENTS_FILE"

# 4. Print activation and installation instructions
echo ""
echo "---------------------------------------------------------------------"
echo "Environment Setup Complete!"
echo ""
echo "Next Steps:"
echo "1. Activate the virtual environment:"
echo "   source \"$VENV_NAME/bin/activate\""  # Quoted to handle the dot
echo ""
echo "2. Install the required packages (if you haven't already):"
echo "   pip install -r requirements.txt"
echo ""
echo "3. You may also need to download NLTK data (e.g., 'punkt' for sentence tokenization):"
echo "   After activating and installing, run python and then:"
echo "   >>> import nltk"
echo "   >>> nltk.download('punkt')"
echo ""
echo "4. To deactivate the virtual environment later, simply type:"
echo "   deactivate"
echo "---------------------------------------------------------------------"

# --- Script End ---