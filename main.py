from utils import (read_video, save_video, get_foot_position)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
import numpy as np
import math

def main():
    input_video_path = r"input\cv_video_input.mp4"
    video_frames = read_video(input_video_path)

    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_best.pt')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    court_line_detector = CourtLineDetector('models/keypoints_model.pth')
    court_keypoints = court_line_detector.predict(video_frames[0])
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    #Draw Bounding Boxes
    output_video_frames = player_tracker.draw_boundingboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_boundingboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    player_paths = {}
    for player_dict in player_detections:
        for track_id, boundingbox in player_dict.items():
            center = get_foot_position(boundingbox)
            if track_id not in player_paths:
                player_paths[track_id] = []
            player_paths[track_id].append(center)

    trajectory_image = output_video_frames[0].copy()
    for track_id, path in player_paths.items():
        color = (0, 255, 0)  
        for i in range(len(path) - 1):
            cv2.line(trajectory_image, path[i], path[i+1], color, 2)

    cv2.imwrite("output_videos/player_trajectories.png", trajectory_image)


    #Draw frame number in the video for our reference
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, r"output_videos/output_video3.mp4")

    src_points = np.float32([
        [578, 297],   # top-left
        [1334, 297],   # top-right
        [358, 852],   # bottom-right
        [1560, 852]    # bottom-left
    ])
    
    dst_points = np.float32([
        [200, 170],       # top-left
        [470, 170],       # top-right
        [200, 955],       # bottom-right
        [470, 955]        # bottom-left
    ])
    
    H, _ = cv2.findHomography(src_points, dst_points)
    # Compute the perspective transform matrix
    warped = cv2.warpPerspective(trajectory_image, H, (600, 1080))

    cv2.imwrite("output_videos/player_trajectories_top_down.png", warped)

    for idx, (track_id, path) in enumerate(player_paths.items()):
        # Transform path points into warped coordinates
        path_array = np.array(path, dtype=np.float32).reshape(-1,1,2)
        warped_path = cv2.perspectiveTransform(path_array, H).reshape(-1,2)

        dist = 0.0
        for i in range(len(warped_path)-1):
            p1 = warped_path[i]
            p2 = warped_path[i+1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            pixel_distance = math.sqrt(dx*dx + dy*dy)
            dist += pixel_distance * 0.033

        # Print the distance and put text on the warped image
        print(f"Player {track_id} covered {dist:.2f} meters.")
        y_offset = 50 + (idx * 40)
        cv2.putText(warped, f"Player {track_id}: {dist:.2f}m", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imwrite("output_videos/player_trajectories_top_down_annotated.png", warped)

    

if __name__ == "__main__":
    main()