from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import os

def download_kaggle_dataset(dataset, path):
    api = KaggleApi()
    api.authenticate()
    
    # Récupérer la liste des fichiers
    files = api.dataset_list_files(dataset).files
    total_files = len(files)
    
    # Télécharger chaque fichier avec une barre de progression
    for file in tqdm(files, desc="Téléchargement des fichiers", total=total_files):
        api.dataset_download_file(dataset, file.name, path=path)

if __name__ == "__main__":
    download_kaggle_dataset("nom_du_dataset", "/chemin/vers/le/dossier")
