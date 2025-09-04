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
import sys
import time
import psutil
import gc
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    
    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def print_detection_stats(frame_count, total_frames, detection_count, max_detections):
    """
    Print detection statistics for current frame
    
    Args:
        frame_count: Current frame number
        total_frames: Total frames in video
        detection_count: Players detected in current frame
        max_detections: Maximum detections seen so far
    """
    stats = f"Frame {frame_count}/{total_frames} | Players: {detection_count} | Max: {max_detections}"
    sys.stdout.write(f'\r{stats}')
    sys.stdout.flush()

def get_system_stats():
    """
    Get current system resource usage
    
    Returns:
        dict: System resource information
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    return {
        'memory_mb': memory_info.rss / 1024 / 1024,
        'cpu_percent': cpu_percent,
        'memory_percent': process.memory_percent()
    }

def save_detection_report(input_file, frame_count, processing_time, total_detections, 
                         avg_confidence, detection_rate, fps_processed, detection_distribution,
                         min_confidence, max_confidence, detection_std, detection_median,
                         memory_used, final_memory, output_dir="reports"):
    """
    Save detection statistics to a JSON report file
    
    Args:
        input_file: Path to input file
        frame_count: Total frames processed
        processing_time: Total processing time
        total_detections: Total player detections
        avg_confidence: Average detection confidence
        detection_rate: Percentage of frames with detections
        fps_processed: Processing speed in FPS
        detection_distribution: Scene complexity distribution
        min_confidence: Minimum confidence score
        max_confidence: Maximum confidence score
        detection_std: Standard deviation of detections
        detection_median: Median detections per frame
        memory_used: Memory consumed during processing
        final_memory: Final memory usage
        output_dir: Directory to save reports
    """
    # Create reports directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report data
    report_data = {
        "metadata": {
            "input_file": str(input_file),
            "timestamp": datetime.now().isoformat(),
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "processing_stats": {
            "total_frames": frame_count,
            "processing_time_seconds": round(processing_time, 2),
            "processing_speed_fps": round(fps_processed, 2),
            "memory_used_mb": round(memory_used, 1),
            "final_memory_mb": round(final_memory, 1)
        },
        "detection_stats": {
            "total_detections": total_detections,
            "detection_rate_percent": round(detection_rate, 1),
            "avg_players_per_frame": round(total_detections / frame_count if frame_count > 0 else 0, 2),
            "detections_per_second": round(total_detections / processing_time if processing_time > 0 else 0, 1)
        },
        "confidence_stats": {
            "average_confidence": round(avg_confidence, 3) if total_detections > 0 else 0,
            "min_confidence": round(min_confidence, 3) if min_confidence != float('inf') else 0,
            "max_confidence": round(max_confidence, 3),
            "confidence_range": round(max_confidence - min_confidence, 3) if min_confidence != float('inf') else 0
        },
        "consistency_stats": {
            "detection_std_dev": round(detection_std, 2) if detection_std else 0,
            "median_players_per_frame": round(detection_median, 1) if detection_median else 0
        },
        "scene_complexity": {
            "low_complexity_frames": detection_distribution['low'],
            "medium_complexity_frames": detection_distribution['medium'],
            "high_complexity_frames": detection_distribution['high']
        },
        "performance_scores": {
            "efficiency_score": round(total_detections / processing_time if processing_time > 0 else 0, 2),
            "quality_score": round(avg_confidence * (detection_rate / 100) if total_detections > 0 else 0, 3)
        }
    }
    
    # Generate filename
    input_name = Path(input_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"detection_report_{input_name}_{timestamp}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Detection report saved to: {report_path}")
    return report_path

def generate_performance_report(frame_count, processing_time, total_detections, 
                              avg_confidence, detection_rate, fps_processed):
    """
    Generate a comprehensive performance report
    
    Args:
        frame_count: Total frames processed
        processing_time: Total processing time
        total_detections: Total player detections
        avg_confidence: Average detection confidence
        detection_rate: Percentage of frames with detections
        fps_processed: Processing speed in FPS
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    # Efficiency metrics
    efficiency_score = (total_detections / processing_time) if processing_time > 0 else 0
    quality_score = avg_confidence * (detection_rate / 100) if total_detections > 0 else 0
    
    print(f"Overall Efficiency Score: {efficiency_score:.2f} detections/second")
    print(f"Quality Score: {quality_score:.3f} (confidence × detection rate)")
    
    # Performance recommendations
    print("\nPERFORMANCE RECOMMENDATIONS:")
    if fps_processed < 10:
        print("  ⚠️  Processing speed is low. Consider:")
        print("     - Using GPU acceleration if available")
        print("     - Reducing input resolution")
        print("     - Lowering confidence threshold")
    
    if avg_confidence < 0.7:
        print("  ⚠️  Detection confidence is low. Consider:")
        print("     - Retraining the model with more data")
        print("     - Adjusting confidence threshold")
        print("     - Checking input video quality")
    
    if detection_rate < 50:
        print("  ⚠️  Low detection rate. Consider:")
        print("     - Checking if video contains players")
        print("     - Adjusting model parameters")
        print("     - Verifying model training data")
    
    print("="*60)

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
    start_time = time.time()
    max_detections = 0
    total_confidence = 0.0
    total_detections = 0
    frames_with_detections = 0
    min_confidence = float('inf')
    max_confidence = 0.0
    detection_counts = []
    detection_distribution = {'low': 0, 'medium': 0, 'high': 0}
    
    # Get initial system stats
    initial_stats = get_system_stats()
    print(f"Initial memory usage: {initial_stats['memory_mb']:.1f} MB")
    print(f"Initial CPU usage: {initial_stats['cpu_percent']:.1f}%")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference on frame
            results = model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)
            
            # Get detection count for this frame
            detection_count = len(results[0].boxes)
            max_detections = max(max_detections, detection_count)
            detection_counts.append(detection_count)
            
            # Categorize frame complexity
            if detection_count == 0:
                pass  # No detections
            elif detection_count <= 3:
                detection_distribution['low'] += 1
            elif detection_count <= 6:
                detection_distribution['medium'] += 1
            else:
                detection_distribution['high'] += 1
            
            # Accumulate confidence scores
            if detection_count > 0:
                frames_with_detections += 1
                for box in results[0].boxes:
                    confidence = box.conf.item()
                    total_confidence += confidence
                    total_detections += 1
                    min_confidence = min(min_confidence, confidence)
                    max_confidence = max(max_confidence, confidence)
            
            # Show detection stats
            print_detection_stats(frame_count, total_frames, detection_count, max_detections)
            
            # Monitor system resources every 50 frames
            if frame_count % 50 == 0:
                current_stats = get_system_stats()
                print(f"\n[Frame {frame_count}] Memory: {current_stats['memory_mb']:.1f} MB, CPU: {current_stats['cpu_percent']:.1f}%")
            
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
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nProcessed {frame_count} frames")
        print(f"Total player detections: {total_detections}")
        print(f"Maximum players detected in a single frame: {max_detections}")
        if total_detections > 0:
            avg_confidence = total_confidence / total_detections
            print(f"Average detection confidence: {avg_confidence:.3f}")
            print(f"Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Calculate and display detection rate
        detection_rate = (frames_with_detections / frame_count * 100) if frame_count > 0 else 0
        print(f"Detection rate: {detection_rate:.1f}% of frames had players")
        
        # Calculate and display detection density
        avg_players_per_frame = total_detections / frame_count if frame_count > 0 else 0
        print(f"Average players per frame: {avg_players_per_frame:.2f}")
        
        # Calculate and display detection consistency
        if len(detection_counts) > 1:
            import statistics
            detection_std = statistics.stdev(detection_counts)
            detection_median = statistics.median(detection_counts)
            print(f"Detection consistency (std dev): {detection_std:.2f} players")
            print(f"Median players per frame: {detection_median:.1f}")
        
        # Display detection distribution
        print(f"Scene complexity distribution:")
        print(f"  Low (1-3 players): {detection_distribution['low']} frames")
        print(f"  Medium (4-6 players): {detection_distribution['medium']} frames")
        print(f"  High (7+ players): {detection_distribution['high']} frames")
        
        # Calculate and display detection efficiency
        detections_per_second = total_detections / processing_time if processing_time > 0 else 0
        print(f"Detection rate: {detections_per_second:.1f} players/second")
        
        # Calculate and display processing speed
        fps_processed = frame_count / processing_time if processing_time > 0 else 0
        print(f"Processing speed: {fps_processed:.2f} FPS")
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        # System resource summary
        final_stats = get_system_stats()
        memory_used = final_stats['memory_mb'] - initial_stats['memory_mb']
        print(f"\nSYSTEM RESOURCE SUMMARY:")
        print(f"Memory used: {memory_used:.1f} MB")
        print(f"Final memory: {final_stats['memory_mb']:.1f} MB")
        print(f"Peak CPU usage: {final_stats['cpu_percent']:.1f}%")
        
        # Generate performance report
        generate_performance_report(frame_count, processing_time, total_detections, 
                                 avg_confidence, detection_rate, fps_processed)
        
        # Save detection report to JSON
        save_detection_report(video_path, frame_count, processing_time, total_detections,
                            avg_confidence, detection_rate, fps_processed, detection_distribution,
                            min_confidence, max_confidence, detection_std, detection_median,
                            memory_used, final_stats['memory_mb'])
        
        # Clean up memory
        gc.collect()
        print(f"Detection completed successfully!")

def process_batch_files(model, input_dir, output_dir, conf_threshold):
    """
    Process multiple files in a directory
    
    Args:
        model: YOLO model instance
        input_dir: Directory containing input files
        output_dir: Directory to save outputs
        conf_threshold: Confidence threshold
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    all_extensions = video_extensions + image_extensions
    
    # Find all supported files
    files_to_process = []
    for ext in all_extensions:
        files_to_process.extend(input_path.glob(f"*{ext}"))
        files_to_process.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not files_to_process:
        print(f"No supported files found in {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(files_to_process)}: {file_path.name}")
        print(f"{'='*60}")
        
        # Generate output path if specified
        output_file = None
        if output_path:
            output_path.mkdir(exist_ok=True)
            if file_path.suffix.lower() in video_extensions:
                output_file = output_path / f"detected_{file_path.stem}.mp4"
            else:
                output_file = output_path / f"detected_{file_path.name}"
        
        # Process file
        try:
            if file_path.suffix.lower() in video_extensions:
                detect_on_video(model, str(file_path), str(output_file) if output_file else None, conf_threshold)
            else:
                detect_on_image(model, str(file_path), str(output_file) if output_file else None, conf_threshold)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    print(f"\nBatch processing completed! Processed {len(files_to_process)} files.")

def main():
    parser = argparse.ArgumentParser(description='Football Player Detection using YOLOv8')
    parser.add_argument('--input', '-i', required=True, help='Path to input image, video, or directory')
    parser.add_argument('--output', '-o', help='Path to output file or directory (optional)')
    parser.add_argument('--model', '-m', default='runs/detect/train2/weights/best.pt', 
                       help='Path to model weights (default: runs/detect/train2/weights/best.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, 
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all supported files in the input directory')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
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
    
    # Check if batch processing is requested
    if args.batch or os.path.isdir(args.input):
        process_batch_files(model, args.input, args.output, args.conf)
        return
    
    # Single file processing
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