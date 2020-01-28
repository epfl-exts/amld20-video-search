# Import libraries
from PIL import Image, ImageDraw, ImageFont, ExifTags
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as patches
from IPython.display import display
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib import ticker
import seaborn as sns
from math import ceil
import pandas as pd
import numpy as np
import requests
import shutil
import errno
import dlib
import json
import glob
import dlib
import time
import sys
import cv2
import os

# Locate folders
VIDEO_DIR = os.path.join('data', 'videos')
FRAMES_DIR = os.path.join('data', 'frames')
IMAGES_DIR = os.path.join('data', 'images')

def PillowExifOpen(path):
    img = Image.open(path)
    
    # Handle .jpg "orientation" EXIF code (Pillow doesn't atm)
    try:
        # Get EXIF "orientation" code
        tags_code = {v: k for k, v in ExifTags.TAGS.items()}
        code = tags_code['Orientation']
                
        # Check image EXIF orit
        img_exif = dict(img._getexif().items())
        
        if img_exif[code] == 3:
            img=img.rotate(180, expand=True)
            
        elif img_exif[code] == 6:
            img=img.rotate(270, expand=True)
            
        elif img_exif[code] == 8:
            img=img.rotate(90, expand=True)

    except:
        pass
    
    return img

def download_file(url, name, target_dir):
    try:
        # Create stream object
        response = requests.get(url, stream=True)

        # Define download parameters
        chunk_size, unit = 2**20, 'MB'
        content_length = response.headers.get('content-length')
        if content_length is not None:
            total_iter = ceil(int(content_length)/chunk_size)
        else:
            total_iter = None
        file_path = os.path.join(target_dir, name)

        with open(file_path, 'wb') as handle:
            # Create progress bar
            progress_bar = lambda it: tqdm(it, desc='Downloading',total=total_iter, unit=unit)

            # Download content
            for data in progress_bar(response.iter_content(chunk_size)):
                handle.write(data)
        return file_path
    
    except:
        print('Cannot download file at', url)

def download_video(url, name):
    return download_file(url, name, VIDEO_DIR)

def download_image(url, name):
    return download_file(url, name, IMAGES_DIR)

# Function to extract frames
def extract_frames(video_name, n_frames=150, max_frame_size=1080):
    # Create a new directory to store frames
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.mkdir(FRAMES_DIR)
    
    # Save frames metadata in a variable
    frames_data = defaultdict(list)
    
    # Locate video
    video_path = os.path.join(VIDEO_DIR, video_name)
    if os.path.isfile(video_path):
        # Load
        video = cv2.VideoCapture(video_path)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        frame_indexes = np.linspace(0, video_frame_count-1, n_frames, dtype=int)

        for frame_index in tqdm(frame_indexes, desc='Extracting frames', unit='frame'):
            # Read frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame_retrieved, frame = video.read()

            if frame_retrieved:
                # Convert to Pillow, had issues with OpenCV saving images depending on computer
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize
                frame.thumbnail([max_frame_size, max_frame_size])

                # Save
                frame_path = os.path.join(FRAMES_DIR, 'frame-{:09}.png'.format(frame_index))
                frame.save(frame_path)

                # Log metadata
                frames_data['path'].append(frame_path)
                frames_data['frame_index'].append(frame_index)
                frames_data['sec'].append(frame_index / video_fps)
                frames_data['width'].append(frame.size[0])
                frames_data['height'].append(frame.size[1])

    else:
        print('Cannot find:', video_path)

    # Load metadata into a DataFrame
    frames_df = pd.DataFrame(frames_data)

    # Save to csv file
    frames_df.to_csv(os.path.join(FRAMES_DIR, 'metadata.csv'), index=False)

    return frames_df

def get_dlib_path(path):
    return os.path.join('data', 'dlib-models', path)

# Function to extract faces information from an image
def perform_face_recognition(img, face_detector, resize_img):
    # Resize to avoid issues with too large images
    if resize_img is not None:
        width, height = img.size
        r = max(height/resize_img, width/resize_img)
        width, height = int(width/r), int(height/r)
        img = img.resize([width, height], resample=Image.BILINEAR)
        
    # Convert to Numpy array
    img = np.array(img)
    
    # Additional dlib models for face recognition
    shape_predictor = dlib.shape_predictor(get_dlib_path('shape_predictor_5_face_landmarks.dat')) # Faces landmarks (points)
    face_recognizer = dlib.face_recognition_model_v1(get_dlib_path('dlib_face_recognition_resnet_model_v1.dat')) # Embedding vector (abstract)

    # Faces, landmarks, face chips and embeddings
    rectangles = [x if isinstance(x, dlib.rectangle) else x.rect for x in face_detector(img, 1)]
    landmarks = [shape_predictor(img, r) for r in rectangles]
    face_chips = [dlib.get_face_chip(img, l) for l in landmarks]
    embeddings = [face_recognizer.compute_face_descriptor(c) for c in face_chips]

    return {
        'rectangles': rectangles,
        'landmarks': landmarks,
        'face_chips': face_chips,
        'embeddings': embeddings,
        'resized_img': img
    }

def face_recognition(path, face_detector, resize_img=512):
    # Locate image
    if not os.path.isfile(path):
        print('Cannot find:', path)
        return None

    # Load image, convert to RGB
    original_img = PillowExifOpen(path)
    original_img = original_img.convert('RGB')

    # Perform face recognition
    faces = perform_face_recognition(original_img, face_detector, resize_img)
    
    # Define image size
    display_size = 512
    resized_img = Image.fromarray(faces['resized_img'])
    width, height = resized_img.size
    r = max(height/display_size, width/display_size)
    width, height = int(width/r), int(height/r)
    transform = lambda p: tuple([int(x/r) for x in p])

    # Prepare image
    img = original_img.resize([width, height], resample=Image.BILINEAR)
    img_draw = ImageDraw.Draw(img, mode='RGB')

    # Draw bounding box + landmarks
    rectangle_width = 3
    landmark_radius = 1
    boxes = []
    for rectangle, landmarks in zip(faces['rectangles'], faces['landmarks']):
        # Crop bounding box before drawing rectangle
        box_xy = [rectangle.left(), rectangle.top(),  rectangle.right(), rectangle.bottom()]
        boxes.append(resized_img.crop(box_xy))
        
        box_xy = transform(box_xy)
        img_draw.rectangle(box_xy, fill=None, outline='red', width=rectangle_width)

        # Draw landmarks
        for p in landmarks.parts():
            x, y = transform([p.x, p.y])
            landmarks_xy = [(x - landmark_radius, y - landmark_radius), (x + landmark_radius, y + landmark_radius)]
            img_draw.rectangle(landmarks_xy, fill='red')

    # Generate face chips image
    face_size = 64
    margin = 12
    panel_width = len(boxes) * (face_size + margin) + margin
    panel_height = 2 * (face_size + margin) + margin
    faces_img = Image.new('RGB', (panel_width, panel_height), color='white')
    for i, (box, face_chip) in enumerate(zip(boxes, faces['face_chips'])):
        # Resize images
        box = box.resize([face_size, face_size], resample=Image.NEAREST)
        face_chip = Image.fromarray(face_chip).resize([face_size, face_size], resample=Image.NEAREST)

        # Paste them
        x = i * face_size + (i + 1) * margin
        faces_img.paste(box, (x, margin))
        faces_img.paste(face_chip, (x, face_size + 2 * margin))
        
    # Add legend
    legend_img = Image.new('RGB', (80, panel_height), color='white')
    legend_draw = ImageDraw.Draw(legend_img, mode='RGB')
    font = ImageFont.truetype(os.path.join('fonts', 'Inconsolata-Bold.ttf'), 16)
    legend_draw.text((margin, margin), 'Original', fill='black', font=font)
    legend_draw.text((margin, 2*margin + face_size), 'Aligned', fill='black', font=font)
    
    # Display both images
    display(img)
    display(Image.fromarray(np.hstack([legend_img, faces_img])))

def extract_faces(src_dir, face_detector, resize_img=None):
    # Create directory
    faces_dir = os.path.join('data', 'faces')
    if os.path.exists(faces_dir):
        shutil.rmtree(faces_dir)
    os.mkdir(faces_dir)

    faces_data = defaultdict(list)
    for src_img_path in tqdm(glob.glob(src_dir + '/*.png'), desc='Extracting faces', unit='face'):
        # Extract faces
        img = PillowExifOpen(src_img_path)
        faces = perform_face_recognition(img, face_detector, resize_img)

        # Save them
        for embedding, chip_img in zip(faces['embeddings'], faces['face_chips']):
            face_img_path = os.path.join(faces_dir, 'face-{:09}.png'.format(len(faces_data['src_img_path'])))
            Image.fromarray(chip_img).save(face_img_path)

            faces_data['src_img_path'].append(src_img_path)
            faces_data['face_img_path'].append(face_img_path)
            faces_data['width'].append(chip_img.shape[0])
            faces_data['height'].append(chip_img.shape[1])
            faces_data['json_embedding'].append(json.dumps(list(embedding)))

    # Load metadata into a DataFrame
    faces_df = pd.DataFrame(faces_data)

    # Save it to csv
    faces_df.to_csv(os.path.join(faces_dir, 'metadata.csv'), index=False)
    
def plot_embeddings(src_dir):
    # Load embeddings
    faces_df = pd.read_csv(os.path.join(src_dir, 'metadata.csv'))
    faces_df['embedding'] = faces_df['json_embedding'].apply(json.loads)
    X = np.array([x for x in faces_df['embedding']])

    # Compress data to 2d space
    X_2d = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(X_2d[:, 0], X_2d[:, 1])
    plt.axis('off')
    plt.show()

def cluster_faces(src_dir):
    # Load face metadata
    faces_df = pd.read_csv(os.path.join(src_dir, 'metadata.csv'))
    
    # Check if clustering already exists
    if 'cluster' not in faces_df.columns:
        # Chinese whispers clustering
        faces_df['embedding'] = faces_df['json_embedding'].apply(json.loads)
        X = np.array([x for x in faces_df['embedding']])
        faces_df['cluster'] = dlib.chinese_whispers_clustering([dlib.vector(x) for x in X], 0.5)
    
        # Persist clustering
        faces_df.to_csv(os.path.join(src_dir, 'metadata.csv'), index=False)

def plot_clusters(src_dir, image_size=(32, 32), ncols=20, clip=False, top_N=-1):
    # Cluster faces
    cluster_faces(src_dir)
    
    # Load face metadata
    faces_df = pd.read_csv(os.path.join(src_dir, 'metadata.csv'))
    
    # Group faces by cluster
    clusters = faces_df.groupby('cluster')
    
    # Validate parameters
    top_N = len(clusters) if top_N == -1 else max(1, min(len(clusters), top_N))
    cluster_order = clusters.size().sort_values(ascending=False)[:top_N]
    
    # Compute image size
    nrows = {cluster: (1 if clip else ceil(size/ncols)) for cluster, size in cluster_order.items()}
    tot_rows = sum(nrows.values())
    grid_width = ncols * image_size[0]
    grid_height = tot_rows * image_size[1]

    # Create image
    faces_grid = Image.new('RGB', (grid_width, grid_height), color='white')

    row_start = 0
    for cluster, size in cluster_order.items():    
        cluster_df = clusters.get_group(cluster)
        for j, path in enumerate(cluster_df['face_img_path']):
            if clip and (j > ncols):
                continue

            # Compute (x, y) position in the grid
            row = row_start + j // ncols
            col = j % ncols
            x = col * image_size[0]
            y = row * image_size[1] 

            # Paste image
            face = PillowExifOpen(cluster_df.iloc[j]['face_img_path'])
            face = face.resize(image_size)
            faces_grid.paste(face, (x, y))

        row_start += nrows[cluster]

    # Draw legend
    legend_width = 120
    legend_height = tot_rows * image_size[1]
    legend_img = Image.new('RGB', (legend_width, legend_height), color='white')
    legend_draw = ImageDraw.Draw(legend_img, mode='RGB')
    font = ImageFont.truetype(os.path.join('fonts', 'Inconsolata-Bold.ttf'), 16)

    row_start = 0
    text_offset = (10, 5)
    for cluster in cluster_order.index:
        x = text_offset[0]
        y = text_offset[1] + row_start * image_size[1]
        legend_draw.text((x, y), 'Cluster {:>3}'.format(cluster), fill='black', font=font)
        row_start += nrows[cluster]
    
    # Return image
    return Image.fromarray(np.hstack([legend_img, faces_grid]))

def plot_timeline(frames_dir, faces_dir, top_N=10):
    # Cluster faces
    cluster_faces(faces_dir)

    # Load metadata
    frames_df = pd.read_csv(os.path.join(frames_dir, 'metadata.csv'))
    faces_df = pd.read_csv(os.path.join(faces_dir, 'metadata.csv'))
    
    # Group faces by cluster
    merged_df = pd.merge(left=faces_df, right=frames_df, left_on='src_img_path', right_on='path')
    clusters = merged_df.groupby('cluster')

    # Work on the top N clusters
    top_N = len(clusters) if top_N == -1 else max(1, min(len(clusters), top_N))
    top_clusters = clusters.size().sort_values(ascending=False)[:top_N]

    # Start, end of video and order of appearance of each cluster
    start_sec, end_sec = 0, frames_df.sec.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 0.4*top_N))

    yindex_to_cluster = {}
    for y, cluster in enumerate(top_clusters.index):
        # Retrieve timestamps
        cluster_df = clusters.get_group(cluster)
        timestamps = cluster_df['sec'].values

        # Plot
        ax.hlines(y=y, xmin=start_sec, xmax=end_sec, color='gray', alpha=1, linewidth=1, linestyle='--', zorder=1)
        ax.scatter(y=np.full_like(timestamps, y), x=timestamps, marker='_', s=100, linewidth=10, zorder=2)
    
    # Format x-axis to minutes/seconds
    sec_formatter = ticker.FuncFormatter(lambda sec, x: time.strftime('{:02d}:{:02d}'.format(int(sec//60), int(sec)%60)))
    ax.xaxis.set_major_formatter(sec_formatter)
    ax.xaxis.set_ticks(np.linspace(start_sec, end_sec, 10))

    # Format y-axis
    ax.set_yticks(range(top_N))
    ax.set_yticklabels(['cluster {}'.format(cluster) for cluster in top_clusters.index])
    ax.invert_yaxis()

    # Format figure
    ax.tick_params(axis='y', which='both',length=0, pad=-20)
    ax.set_frame_on(False)

    plt.show()