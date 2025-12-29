import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from collections import defaultdict

# =================================================================================
# [Configuration] Constants & Environment Variables
# =================================================================================
FIELD_LENGTH = 105  # Meters (Length)
FIELD_WIDTH = 68    # Meters (Width)

# =================================================================================
# [Classes & Functions] Coordinate Transformation & Analysis Tools
# =================================================================================
class ViewTransformer:
    """
    Class to transform video pixel coordinates to real-world field meters.
    """
    def __init__(self, method='linear'):
        self.method = method

    def transform_point(self, point, frame_shape):
        if point is None: return None
        # Simple Linear Mapping (Assuming Top-Down view)
        real_x = (point[0] / frame_shape[1]) * FIELD_LENGTH
        real_y = (point[1] / frame_shape[0]) * FIELD_WIDTH
        return (real_x, real_y)

def get_zone(field_pos):
    """
    Calculates the 18-Zone (6x3 grid) number from field coordinates.
    """
    if field_pos is None: return None
    x, y = field_pos
    
    # Horizontal: 6 divisions (0~5)
    col = int(x / (FIELD_LENGTH / 6))
    col = max(0, min(col, 5))
    
    # Vertical: 3 divisions (0~2)
    row = int(y / (FIELD_WIDTH / 3))
    row = max(0, min(row, 2))
    
    # Zone Number: 1 ~ 18
    return (row * 6) + col + 1

def stitch_tracks(df, target_id, max_frame_gap=60, max_dist_gap=150):
    """
    [Core Logic] Track Stitching: Reconnects broken IDs (Recursive).
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

    # Compare distance with candidates
    for cand_id in candidates['track_id'].unique():
        cand_track = candidates[candidates['track_id'] == cand_id]
        start_row = cand_track.iloc[0]
        start_pos = ((start_row['x1']+start_row['x2'])/2, (start_row['y1']+start_row['y2'])/2)
        
        dist = np.sqrt((start_pos[0]-last_pos[0])**2 + (start_pos[1]-last_pos[1])**2)

        if dist < max_dist_gap and dist < min_dist:
            min_dist = dist
            best_match_id = cand_id

    # Merge and recursive call
    if best_match_id is not None:
        print(f"   [LINK] ID {target_id} -> ID {best_match_id} connected (Dist: {min_dist:.1f}px)")
        df.loc[df['track_id'] == best_match_id, 'track_id'] = target_id
        return stitch_tracks(df, target_id, max_frame_gap, max_dist_gap)

    return df, target_id

# =================================================================================
# [Visualization] Report Generation Tools
# =================================================================================
def draw_full_pitch(ax, pitch_color='darkgreen', line_color='white'):
    ax.set_facecolor(pitch_color)
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(FIELD_WIDTH, 0) # Invert Y-axis

    # Outlines & Center line
    ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, edgecolor=line_color, facecolor='none', linewidth=2))
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color=line_color, linewidth=2)
    # Center circle
    ax.add_patch(patches.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15, edgecolor=line_color, facecolor='none', linewidth=2))
    # Penalty box
    ax.add_patch(patches.Rectangle((0, (FIELD_WIDTH-40.3)/2), 16.5, 40.3, edgecolor=line_color, facecolor='none', linewidth=2))
    ax.add_patch(patches.Rectangle((FIELD_LENGTH-16.5, (FIELD_WIDTH-40.3)/2), 16.5, 40.3, edgecolor=line_color, facecolor='none', linewidth=2))

def plot_results(target_df, report_title="Player Analysis Report", save_path="result_plot.png"):
    if target_df.empty: return
    
    # Aggregate data
    final_player_zones = defaultdict(int)
    for z in target_df['zone_id']:
        if z > 0: final_player_zones[z] += 1
        
    final_player_positions = list(zip(target_df.index, target_df['real_x'], target_df['real_y']))

    # Set style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(report_title, fontsize=20, color='white')

    # 1. Left: 18-Zone Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    zone_grid = np.zeros((3, 6))
    total_frames = len(target_df)

    if total_frames > 0:
        for zone, count in final_player_zones.items():
            idx = int(zone) - 1
            r, c = divmod(idx, 6)
            if 0 <= r < 3 and 0 <= c < 6:
                zone_grid[r, c] = count / total_frames

    sns.heatmap(zone_grid, annot=True, fmt=".1%", cmap="YlGn", ax=ax1, cbar=False,
                xticklabels=[1,2,3,4,5,6], yticklabels=[])
    ax1.set_title("18-Zone Occupation Rate")
    ax1.set_xlabel("Horizontal Zones")

    # 2. Right: Full Pitch Trajectory
    ax2 = fig.add_subplot(1, 2, 2)
    draw_full_pitch(ax2)

    xs = [p[1] for p in final_player_positions]
    ys = [p[2] for p in final_player_positions]

    # KDE Plot (Density)
    if len(xs) > 10:
        sns.kdeplot(x=xs, y=ys, cmap="Reds", fill=True, alpha=0.4, thresh=0.05, ax=ax2, warn_singular=False)
    
    ax2.plot(xs, ys, color='yellow', linewidth=2, label='Movement')
    ax2.plot(xs[0], ys[0], 'wo', markersize=8, label='Start')
    ax2.plot(xs[-1], ys[-1], 'rx', markersize=8, label='End')
    ax2.set_title("Full Pitch Heatmap & Trajectory")
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"  - Plot Saved: {save_path}")
    plt.close()

# =================================================================================
# [Main Logic] Analysis Execution Function
# =================================================================================
def run_analysis(video_path, target_init_pos, output_path, model_path='yolov10m.pt'):
    # 1. Initialization
    transformer = ViewTransformer(method='linear')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 1.5x upscaling for small object detection
    SCALE = 1.5
    new_w, new_h = int(orig_w * SCALE), int(orig_h * SCALE)
    scaled_target_pos = (target_init_pos[0]*SCALE, target_init_pos[1]*SCALE)

    print(f"▶ Start Analysis: {video_path}")
    
    # Load model (download if not exists)
    if not os.path.exists(model_path):
        try:
            import torch
            torch.hub.download_url_to_file(f'https://github.com/THU-MIG/yolov10/releases/download/v1.1/{model_path}', model_path)
        except: pass
    
    model = YOLO(model_path)
    raw_data = []
    initial_target_id = None
    frame_id = 0

    # 2. Inference Loop (Tracking)
    print("▶ Step 1: Scanning Video & Tracking...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results = model.track(frame_resized, persist=True, tracker="botsort.yaml",
                              conf=0.1, iou=0.5, verbose=False, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # Initial Target Identification
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
        if frame_id % 100 == 0: print(f"   ...Scanning frame {frame_id}")

    cap.release()

    if not raw_data or initial_target_id is None:
        print("Target not found.")
        return

    # 3. Post-Processing (Stitching & Coordinates)
    print("\n▶ Step 2: Post-Processing (Stitching & Interpolation)...")
    df = pd.DataFrame(raw_data, columns=['frame_id', 'x1', 'y1', 'x2', 'y2', 'track_id'])
    df, final_id = stitch_tracks(df, initial_target_id)
    
    target_df = df[df['track_id'] == final_id].copy()
    target_df = target_df.drop_duplicates(subset=['frame_id'], keep='first').set_index('frame_id')
    
    full_idx = pd.RangeIndex(start=0, stop=frame_id)
    target_df = target_df.reindex(full_idx)
    target_df[['x1','y1','x2','y2']] = target_df[['x1','y1','x2','y2']].interpolate(method='linear')
    target_df = target_df.ffill().bfill()

    # Coordinate Transformation & Zone Calculation
    target_df['center_x'] = (target_df['x1'] + target_df['x2']) / 2
    target_df['center_y'] = (target_df['y1'] + target_df['y2']) / 2
    
    real_coords = target_df.apply(
        lambda row: transformer.transform_point((row['center_x'], row['center_y']), (new_h, new_w)), 
        axis=1
    )
    target_df['real_x'] = [c[0] for c in real_coords]
    target_df['real_y'] = [c[1] for c in real_coords]
    target_df['zone_id'] = target_df.apply(lambda row: get_zone((row['real_x'], row['real_y'])), axis=1)

    # 4. Save Results (CSV & Plot)
    csv_path = output_path.replace('.mp4', '.csv')
    target_df.to_csv(csv_path)
    print(f"  - CSV Saved: {csv_path}")

    plot_path = output_path.replace('.mp4', '_analysis.png')
    plot_results(target_df, save_path=plot_path)

    # 5. Video Rendering (Integrated)
    print(f"\n▶ Step 3: Rendering Video: {output_path} ...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
    
    curr_f = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if curr_f in target_df.index:
            row = target_df.loc[curr_f]
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            zone_id = int(row['zone_id'])
            
            # Draw Box & Info
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"TARGET (Zone {zone_id})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        out.write(frame_resized)
        curr_f += 1
        
    cap.release()
    out.release()
    print("✅ All Analysis Complete.")

# =================================================================================
# [Execution] User Configuration
# =================================================================================
# Update the paths and coordinates below to match your files before running.

video_file_path = 'video1.mp4'         # Path to the video file
target_initial_pos = (300, 360)        # Initial pixel coordinates of the target (x, y)
output_file_path = 'final_output.mp4'  # Path for saving results

# Run Analysis
if os.path.exists(video_file_path):
    run_analysis(video_file_path, target_initial_pos, output_file_path)
else:
    print(f"⚠️ File not found: {video_file_path}")
