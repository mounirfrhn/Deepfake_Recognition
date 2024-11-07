import os
import subprocess
import zipfile


def setup_kaggle_auth():
    """Sets up Kaggle authentication by checking for the kaggle.json file or creating it."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Check if kaggle.json exists
    if not os.path.exists(kaggle_json_path):
        print("kaggle.json not found. Make sure you have your Kaggle API key.")
        print("Download it from your Kaggle account: https://www.kaggle.com/account")
        return False

    # Ensure correct permissions
    os.chmod(kaggle_json_path, 0o600)
    return True


def download_kaggle_dataset(competition_name):
    name = competition_name
    """Downloads dataset from a Kaggle competition."""
    if setup_kaggle_auth():
        # Run Kaggle CLI command to download the competition data
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", name], check=True)
            print(f"Data for competition '{competition_name}' downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading data: {e}")
    else:
        print("Authentication setup failed. Please ensure kaggle.json is in ~/.kaggle.")


def unzip_file(zip_path, extract_to="data"):
    """Unzips a file to the specified directory and deletes the zip file."""
    # Assure que le dossier de destination existe
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Décompression du fichier zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to {extract_to}")

    # Supprime le fichier .zip après extraction
    os.remove(zip_path)
    print(f"Deleted the zip file: {zip_path}")


if __name__ == "__main__":
    # Nom du fichier téléchargé par l'API Kaggle
    competition_name = "deepfake-detection-challenge"
    zip_filename = f"{competition_name}.zip"  # Nom du fichier téléchargé
    download_kaggle_dataset(competition_name)

    # Décompresse le fichier téléchargé dans le dossier "data"
    unzip_file(zip_filename, extract_to="/home/onyxia/work/Deepfake_Recognition/data")
