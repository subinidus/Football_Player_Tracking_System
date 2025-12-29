# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Constants (FIFA Standard) ---
FIELD_LENGTH = 105  # meters
FIELD_WIDTH = 68    # meters

# --- [Core Class] View Transformer ---
class ViewTransformer:
    """
    Class to transform video pixel coordinates to real-world field meters.
    Currently adopts 'Linear Mapping' considering broadcast footage characteristics
    (zoom/pan/occlusion) to ensure robustness.
    """
    def __init__(self, method='linear'):
        self.method = method
        # Structure prepared for future Homography expansion

    def transform_point(self, point, frame_shape):
        """
        :param point: (x, y) pixel coordinates
        :param frame_shape: (height, width) of the frame
        :return: (real_x, real_y) in meters
        """
        if point is None: return None
        
        # [Strategy] Robustness over Precision
        # Uses Linear Mapping for consistent calculation
        real_x = (point[0] / frame_shape[1]) * FIELD_LENGTH
        real_y = (point[1] / frame_shape[0]) * FIELD_WIDTH
        return (real_x, real_y)

def get_zone(field_pos):
    """
    Converts coordinates into tactical 18-Zone (6x3 Grid).
    """
    if field_pos is None: return None
    x, y = field_pos
    
    # Zone Index Calculation
    col = int(x / (FIELD_LENGTH / 6))
    col = max(0, min(col, 5))
    row = int(y / (FIELD_WIDTH / 3))
    row = max(0, min(row, 2))
    
    # Return Zone ID (1~18)
    return (row * 6) + col + 1

def stitch_tracks(df, target_id, max_frame_gap=60, max_dist_gap=150):
    """
    [Core Logic] Spatio-temporal Track Stitching.
    Recursively reconnects broken track IDs based on spatial distance and time proximity.
    """
    target_df = df[df['track_id'] == target_id]
    if target_df.empty: return df, target_id

    last_frame = target_df['frame_id'].max()
    last_row = target_df.loc[target_df['frame_id'] == last_frame].iloc[0]
    last_pos = ((last_row['x1']+last_row['x2'])/2, (last_row['y1']+last_row['y2'])/2)

    # Search for candidates in future frames
    candidates = df[
        (df['frame_id'] > last_frame) &
        (df['frame_id'] <= last_frame + max_frame_gap) &
        (df['track_id'] != target_id)
    ]

    if candidates.empty: return df, target_id

    best_match_id = None
    min_dist = float('inf')

    for cand_id in candidates['track_id'].unique():
        cand_track = candidates[candidates['track_id'] == cand_id]
        start_row = cand_track.iloc[0]
        start_pos = ((start_row['x1']+start_row['x2'])/2, (start_row['y1']+start_row['y2'])/2)
        
        # Euclidean Distance Matching
        dist = np.sqrt((start_pos[0]-last_pos[0])**2 + (start_pos[1]-last_pos[1])**2)

        if dist < max_dist_gap and dist < min_dist:
            min_dist = dist
            best_match_id = cand_id

    if best_match_id is not None:
        print(f"   [Link] ID {target_id} -> ID {best_match_id} connected (Dist: {min_dist:.1f}px)")
        # Overwrite the candidate ID with the target ID
        df.loc[df['track_id'] == best_match_id, 'track_id'] = target_id
        # Recursive call to find further connections
        return stitch_tracks(df, target_id, max_frame_gap, max_dist_gap)

    return df, target_id

def render_video(video_path, target_df, output_path, scale=1.5):
    """
    Generates the final output video based on the processed data.
    """
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
    
    frame_idx = 0
    print(f"▶ Generating Video: {output_path} ...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Upscaling
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Check if target data exists for the current frame
        if frame_idx in target_df.index:
            row = target_df.loc[frame_idx]
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            zone_id = int(row['zone_id'])
            
            # [Visual] Draw Target Box & ID
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"TARGET (Zone {zone_id})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        out.write(frame_resized)
        frame_idx += 1
        
    cap.release()
    out.release()
    print("  - Video Generation Complete.")

def plot_heatmap(df, save_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('darkgreen')
    ax.set_xlim(0, FIELD_LENGTH); ax.set_ylim(FIELD_WIDTH, 0)
    
    ax.plot([0, FIELD_LENGTH, FIELD_LENGTH, 0, 0], [0, 0, FIELD_WIDTH, FIELD_WIDTH, 0], color='white')
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color='white') # Center line
    
    sns.kdeplot(x=df['real_x'], y=df['real_y'], cmap="Reds", fill=True, alpha=0.5, ax=ax, warn_singular=False)
    ax.plot(df['real_x'], df['real_y'], color='yellow', linewidth=1, alpha=0.8, label='Movement')
    
    plt.title(f"Tactical Heatmap (Zone Analysis Included)")
    plt.legend()
    plt.savefig(save_path)
    print(f"  - Report Saved: {save_path}")

def run_analysis(video_path, target_init_pos, output_path, model_path='yolov10m.pt'):
    # --- 1. Initialize ---
    transformer = ViewTransformer(method='linear')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # [Tech] 1.5x Upscaling for Small Object Detection
    SCALE = 1.5
    new_w, new_h = int(orig_w * SCALE), int(orig_h * SCALE)
    scaled_target_pos = (target_init_pos[0]*SCALE, target_init_pos[1]*SCALE)

    print(f"▶ Start Analysis: {video_path}")
    print(f"  - Upscaling: {orig_w}x{orig_h} -> {new_w}x{new_h} (x{SCALE})")
    
    # Download model if not exists (Optional check)
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        try:
            import torch
            torch.hub.download_url_to_file(f'https://github.com/THU-MIG/yolov10/releases/download/v1.1/{model_path}', model_path)
        except:
            print("Warning: Auto-download failed. Please ensure yolov10m.pt exists.")

    model = YOLO(model_path)
    raw_data = []
    initial_target_id = None
    frame_id = 0

    # --- 2. Inference Loop (Data Collection) ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # BoT-SORT Tracking
        results = model.track(frame_resized, persist=True, tracker="botsort.yaml",
                              conf=0.1, iou=0.5, verbose=False, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # Target Locking
            if initial_target_id is None:
                min_dist = float('inf')
                best_id = None
                for box, tid in zip(boxes, track_ids):
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    dist = np.sqrt((cx - scaled_target_pos[0])**2 + (cy - scaled_target_pos[1])**2)
                    if dist < (150*SCALE) and dist < min_dist:
                        min_dist = dist
                        best_id = int(tid)
                
                if best_id is not None:
                    initial_target_id = best_id
                    print(f"  - Target Locked: ID {initial_target_id}")

            for box, tid in zip(boxes, track_ids):
                raw_data.append([frame_id, box[0], box[1], box[2], box[3], int(tid)])

        frame_id += 1
        if frame_id % 100 == 0: print(f"  ...Processing frame {frame_id}")

    cap.release()

    if not raw_data or initial_target_id is None:
        print("Target not found.")
        return

    # --- 3. Post-Processing ---
    print("\n▶ Post-Processing (Stitching & Interpolation)...")
    df = pd.DataFrame(raw_data, columns=['frame_id', 'x1', 'y1', 'x2', 'y2', 'track_id'])

    # Stitching
    df, final_id = stitch_tracks(df, initial_target_id)
    
    # Filtering & Interpolation
    target_df = df[df['track_id'] == final_id].copy()
    target_df = target_df.drop_duplicates(subset=['frame_id'], keep='first').set_index('frame_id')
    
    full_idx = pd.RangeIndex(start=0, stop=frame_id)
    target_df = target_df.reindex(full_idx)
    target_df[['x1','y1','x2','y2']] = target_df[['x1','y1','x2','y2']].interpolate(method='linear')
    target_df = target_df.ffill().bfill()

    # --- 4. Coordinate Mapping & Zone Analysis ---
    print("▶ Calculating Spatial Data (Coordinates & Zones)...")
    
    target_df['center_x'] = (target_df['x1'] + target_df['x2']) / 2
    target_df['center_y'] = (target_df['y1'] + target_df['y2']) / 2
    
    real_coords = target_df.apply(
        lambda row: transformer.transform_point((row['center_x'], row['center_y']), (new_h, new_w)), 
        axis=1
    )
    
    target_df['real_x'] = [c[0] for c in real_coords]
    target_df['real_y'] = [c[1] for c in real_coords]
    
    # 18-Zone Calculation
    target_df['zone_id'] = target_df.apply(
        lambda row: get_zone((row['real_x'], row['real_y'])), 
        axis=1
    )

    # --- 5. Export ---
    csv_path = output_path.replace('.mp4', '.csv')
    target_df.to_csv(csv_path)
    print(f"  - Data Saved: {csv_path}")
    
    plot_path = output_path.replace('.mp4', '_heatmap.png')
    plot_heatmap(target_df, plot_path)
    
    render_video(video_path, target_df, output_path, scale=SCALE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Player Tracking System")
    parser.add_argument('--video', type=str, required=True, help="Path to input video file")
    parser.add_argument('--tx', type=int, required=True, help="Target Player Initial X")
    parser.add_argument('--ty', type=int, required=True, help="Target Player Initial Y")
    parser.add_argument('--output', type=str, default="output.mp4", help="Path to save result video")
    
    args = parser.parse_args()
    
    run_analysis(args.video, (args.tx, args.ty), args.output)
