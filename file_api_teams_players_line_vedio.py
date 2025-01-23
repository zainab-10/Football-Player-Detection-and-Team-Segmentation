import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
# Define classes
classes = ['ball', 'goalkeeper', 'nonimpact', 'player']

# Load the video
cap = cv2.VideoCapture('0a2d9b_0.mp4')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)

    # Initialize lists to store player colors
    player_colors = []

    # Process each detection
    for det in results.pred[0]:
        class_idx = int(det[5])
        class_name = classes[class_idx]
        if class_name == 'player':
            bbox = det[:4].int().cpu().numpy()  # Convert bounding box to numpy array
            player_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Extract player region
            # Convert player region to LAB color space
            lab_player_region = rgb2lab(player_region)
            avg_color = np.mean(lab_player_region, axis=(0, 1))  # Calculate average color of player region in LAB space
            player_colors.append(avg_color)

    # Convert player colors to numpy array
    player_colors = np.array(player_colors)

    # Perform color clustering to group players by similar colors
    num_teams = 2  # Number of teams
    kmeans = KMeans(n_clusters=num_teams, random_state=0, n_init=50).fit(player_colors)  # Adjust n_init
    labels = kmeans.labels_

    # Initialize dictionaries to store players for each team
    team1_players = []
    team2_players = []

    # Assign players to teams based on color clusters
    for i, label in enumerate(labels):
        if label == 0:
            team1_players.append(results.pred[0][i].tolist())  # Convert tensor to list
        elif label == 1:
            team2_players.append(results.pred[0][i].tolist())  # Convert tensor to list

    # Write predicted classes and bounding boxes on the frame
    for det in results.pred[0]:
        class_idx = int(det[5])
        class_name = classes[class_idx]
        bbox = det[:4].int().cpu().numpy()  # Convert bounding box to numpy array
        if class_name == 'ball':
            bbox = det[:4].int().cpu().numpy()  # Convert bounding box to numpy array
            ball_x = (bbox[0] + bbox[2]) // 2  # Calculate the x-coordinate of the center of the ball
            # Calculate the x-coordinate of the position slightly in front of the ball
            line_x = ball_x + 10  # Adjust the value to move the line further in front of the ball
            # Draw the line
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

        if class_name == 'player' and class_name != 'goalkeeper' and class_name != 'nonimpact':
            if det.tolist() in team1_players:  # Convert tensor to list before comparison
                class_name = 'player1'
            elif det.tolist() in team2_players:  # Convert tensor to list before comparison
                class_name = 'player2'
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame, class_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Write class name

    # Find leftmost and rightmost players for both teams
    team1_leftmost = min(team1_players, key=lambda x: x[0])
    team1_rightmost = max(team1_players, key=lambda x: x[0])
    team2_leftmost = min(team2_players, key=lambda x: x[0])
    team2_rightmost = max(team2_players, key=lambda x: x[0])

    # Get positions of leftmost and rightmost players
    team1_leftmost_x = int((team1_leftmost[0] + team1_leftmost[2]) / 2)
    team1_leftmost_x = team1_leftmost_x + 10
    team1_rightmost_x = int((team1_rightmost[0] + team1_rightmost[2]) / 2)
    team1_rightmost_x = team1_rightmost_x + 10
    team2_leftmost_x = int((team2_leftmost[0] + team2_leftmost[2]) / 2)
    team2_leftmost_x = team2_leftmost_x + 80
    team2_rightmost_x = int((team2_rightmost[0] + team2_rightmost[2]) / 2)
    team2_rightmost_x = team2_rightmost_x - 20

    # Draw lines in front of leftmost and rightmost players of both teams
    cv2.line(frame, (team1_leftmost_x, 0), (team1_leftmost_x, frame.shape[0]), (255, 0, 0), 2)  # Blue line for team 1
    cv2.line(frame, (team1_rightmost_x, 0), (team1_rightmost_x, frame.shape[0]), (255, 0, 0), 2)  # Blue line for team 1
    cv2.line(frame, (team2_leftmost_x, 0), (team2_leftmost_x, frame.shape[0]), (0, 0, 255), 2)  # Red line for team 2
    cv2.line(frame, (team2_rightmost_x, 0), (team2_rightmost_x, frame.shape[0]), (0, 0, 255), 2)  # Red line for team 2

    # Write the processed frame to the output video
    out.write(frame)

# Release video capture and writer
cap.release()
out.release()
