#!/usr/bin/env python3
"""
Simple Football Player Detection Script using YOLOv8

This script can be used to run inference on images or videos using the trained
football player detection model.

Usage:
    python simple_detection.py --input path/to/image_or_video --output path/to/output
    python simple_detection.py --input image.jpg
    python simple_detection.py --input video.mp4 --conf 0.5
"""

import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def detect_on_image(model, image_path, output_path=None, conf_threshold=0.25):
    """
    Run detection on a single image
    
    Args:
        model: YOLO model instance
        image_path: Path to input image
        output_path: Path to save output image (optional)
        conf_threshold: Confidence threshold for detections
    """
    print(f"Running detection on image: {image_path}")
    
    # Run inference
    results = model.predict(source=image_path, conf=conf_threshold, save=False)
    
    # Get the result for the first (and only) image
    result = results[0]
    
    # Draw bounding boxes on the image
    annotated_image = result.plot()
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Output saved to: {output_path}")
    else:
        # Display the image
        cv2.imshow('Football Player Detection', annotated_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Print detection summary
    if len(result.boxes) > 0:
        print(f"Detected {len(result.boxes)} players with confidence >= {conf_threshold}")
        for i, box in enumerate(result.boxes):
            conf = box.conf.item()
            print(f"  Player {i+1}: Confidence = {conf:.3f}")
    else:
        print("No players detected")

def detect_on_video(model, video_path, output_path=None, conf_threshold=0.25):
    """
    Run detection on a video
    
    Args:
        model: YOLO model instance
        video_path: Path to input video
        output_path: Path to save output video (optional)
        conf_threshold: Confidence threshold for detections
    """
    print(f"Running detection on video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Run inference on frame
            results = model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()
            
            if writer:
                writer.write(annotated_frame)
            else:
                # Display frame (press 'q' to quit)
                cv2.imshow('Football Player Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\nOutput video saved to: {output_path}")
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")

def main():
    parser = argparse.ArgumentParser(description='Football Player Detection using YOLOv8')
    parser.add_argument('--input', '-i', required=True, help='Path to input image or video')
    parser.add_argument('--output', '-o', help='Path to output file (optional)')
    parser.add_argument('--model', '-m', default='runs/detect/train2/weights/best.pt', 
                       help='Path to model weights (default: runs/detect/train2/weights/best.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, 
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Make sure you have trained the model or use the correct path.")
        return
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded successfully!")
    
    # Determine input type
    input_path = Path(args.input)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if input_path.suffix.lower() in video_extensions:
        detect_on_video(model, args.input, args.output, args.conf)
    elif input_path.suffix.lower() in image_extensions:
        detect_on_image(model, args.input, args.output, args.conf)
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        print(f"Supported formats: {image_extensions + video_extensions}")

if __name__ == "__main__":
    main() 