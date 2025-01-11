#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

CONDA_DIR="$HOME/miniconda3"

# Install Miniconda if not already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
    echo "Miniconda installed successfully at $CONDA_DIR"
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# Initialize Conda
echo "Initializing Conda..."
"$CONDA_DIR/bin/conda" init bash

# Configure Conda
echo "Configuring Conda..."
"$CONDA_DIR/bin/conda" config --remove channels https://repo.anaconda.com/pkgs/main || true
"$CONDA_DIR/bin/conda" config --remove channels https://repo.anaconda.com/pkgs/r || true
"$CONDA_DIR/bin/conda" config --add channels conda-forge
"$CONDA_DIR/bin/conda" config --set channel_priority strict
"$CONDA_DIR/bin/conda" config --set auto_activate_base true
cp -f "${HOME}/.condarc" "${CONDA_DIR}/.condarc"

# Update Conda and install Mamba
"$CONDA_DIR/bin/conda" update -n base -c defaults conda -y
"$CONDA_DIR/bin/conda" clean -a -y
"$CONDA_DIR/bin/conda" install -n base -c conda-forge mamba -y
"$CONDA_DIR/bin/mamba" shell init --shell bash --root-prefix=~/.local/share/mamba

# Optionally, you can inform the user to source their ~/.bashrc
echo "Installation and configuration complete."
echo "Please run 'source ~/.bashrc' or restart your terminal to apply the changes."
