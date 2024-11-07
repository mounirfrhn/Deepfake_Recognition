#!/bin/bash
echo "Installation des dépendances système..."

# Installation de libGL
sudo apt-get update
sudo apt-get install -y libgl1

# Installation des packages Python
pip install -r requirements.txt

echo "Installation terminée."
