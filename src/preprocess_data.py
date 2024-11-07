# Imports
import cv2
# !pip install mtcnn
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available:
    print("Using CUDA")
else:
    print("Using CPU")

mtcnn = MTCNN(device=device)


def find_absolute_path(folder_name, start_dir='/'):
    for root, dirs, files in os.walk(start_dir):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None


# Usage
root = find_absolute_path('Deepfake_Recognition')
print(f"Absolute path: {root}")


def crop_faces_from_frame(frame):
    """
    Detects and crops faces from a frame using MTCNN and resizes them to 224x224.

    Parameters:
    -----------
    frame : np.ndarray
        The frame from which faces will be detected and cropped.
    target_size : tuple of int
        The desired size (width, height) for each cropped face.

    Returns:
    --------
    cropped_faces : list of PIL.Image
        List of cropped face images as PIL.Image objects, resized to target_size.
    """

    # Convert the frame (OpenCV image) to PIL format
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect faces and extract bounding boxes
    boxes, _ = mtcnn.detect(img)

    # List to store cropped and resized face images
    cropped_faces = []

    # If faces are detected
    if boxes is not None:
        for box in boxes:
            # Extract coordinates of bounding box
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Crop the face from the frame
            cropped_face = img.crop((x1, y1, x2, y2))

            # Resize the cropped face to 224x224
            cropped_face = cropped_face.resize((224, 224), Image.LANCZOS)
            cropped_faces.append(cropped_face)

    return cropped_faces[0]


def extract_frames(video_path, n):
    """
     Extracts n frames evenly distributed across the video.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total_frames // n, 1)

    frames = []
    for i in range(total_frames):
        success, frame = vidcap.read()

        if not success:
            break

        # Capture uniquement les frames selon l'intervalle
        if i % interval == 0:
            frames.append(frame)

            # Arrête l'extraction une fois qu'on atteint n frames
            if len(frames) >= n:
                break

    vidcap.release()
    # print(f"Nombre de frames extraites : {len(frames)}")

    return frames


def create_cropped_video(video_path, n_frames, output_path):

    frames = extract_frames(video_path, n_frames)
    cropped_frames = []

    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Crop the face from the frame
            cropped_face = img.crop((x1, y1, x2, y2))

            # Resize the cropped face to 224x224
            cropped_face = cropped_face.resize((224, 224), Image.LANCZOS)
            cropped_frames.append(cropped_face)

    fps = 10

    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialiser l'écriture de la vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec pour le format MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))

    # Étape 4 : Écrire chaque frame du visage dans la vidéo de sortie
    for frame in cropped_frames:
        # Si nécessaire, convertir la frame en format BGR pour OpenCV
        face_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(face_bgr)

    # Libérer les ressources
    out.release()
    # print("Vidéo de sortie créée avec succès :", output_path)

# Train videos preprocessing


input_root_train = root + "/data/train_sample_videos"
output_root_train = root + "/data/train_cropped_videos"

train_videos = [file for file in os.listdir(input_root_train) if file.endswith('.mp4')]

for video in tqdm(train_videos):
    path = input_root_train+"/"+video

    name_without_extension = os.path.splitext(video)[0]
    output_path = output_root_train + "/" + name_without_extension + "_cropped.mp4"
    create_cropped_video(path, 30, output_path)

# Test videos preprocessing

input_root_test = root + "/data/test_videos"
output_root_test = root + "/data/test_cropped_videos"

test_videos = [file for file in os.listdir(input_root_test) if file.endswith('.mp4')]

for video in tqdm(test_videos):
    path = input_root_test+"/"+video

    name_without_extension = os.path.splitext(video)[0]
    output_path = output_root_test + "/" + name_without_extension + "_cropped.mp4"
    if not os.path.exists(output_path):
        create_cropped_video(path, 30, output_path)
    else:
        print(f"{output_path} already exists, skipping.")
