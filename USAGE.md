# Simple Football Player Detection Script

This script provides an easy way to use the trained YOLOv8 football player detection model for inference on images and videos.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

#### For Images:
```bash
# Run detection on an image and display result
python simple_detection.py --input image.jpg

# Run detection and save output image
python simple_detection.py --input image.jpg --output result.jpg

# Run with custom confidence threshold
python simple_detection.py --input image.jpg --conf 0.5
```

#### For Videos:
```bash
# Run detection on video and display real-time
python simple_detection.py --input video.mp4

# Run detection and save output video
python simple_detection.py --input video.mp4 --output result.mp4

# Process video with higher confidence threshold
python simple_detection.py --input video.mp4 --output result.mp4 --conf 0.4
```

### Advanced Usage

#### Custom Model Path:
```bash
python simple_detection.py --input image.jpg --model path/to/your/model.pt
```

#### All Options:
```bash
python simple_detection.py \
    --input input_file.jpg \
    --output output_file.jpg \
    --model runs/detect/train2/weights/best.pt \
    --conf 0.25
```

## Command Line Arguments

- `--input` / `-i`: Path to input image or video (required)
- `--output` / `-o`: Path to output file (optional, if not provided, results will be displayed)
- `--model` / `-m`: Path to model weights (default: `runs/detect/train2/weights/best.pt`)
- `--conf` / `-c`: Confidence threshold for detections (default: 0.25)

## Supported Formats

### Images:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Videos:
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

## Examples

### Example 1: Quick Image Detection
```bash
python simple_detection.py -i sample_image.jpg
```

### Example 2: Batch Processing with High Confidence
```bash
# For multiple files, you can use a simple bash loop
for file in *.jpg; do
    python simple_detection.py -i "$file" -o "detected_$file" -c 0.6
done
```

### Example 3: Video Processing
```bash
python simple_detection.py -i football_match.mp4 -o detected_match.mp4 -c 0.3
```

## Output

- **Images**: The script will either display the image with bounding boxes or save it to the specified output path
- **Videos**: For videos, you can either watch real-time detection (press 'q' to quit) or save the processed video
- **Console**: The script prints detection statistics including the number of players detected and their confidence scores

## Model Requirements

Make sure you have the trained model file at `runs/detect/train2/weights/best.pt` or specify a different path using the `--model` argument.

## Troubleshooting

1. **Model not found**: Ensure the model path is correct and the model file exists
2. **Input file not found**: Check the input file path
3. **OpenCV display issues**: If you're running on a headless server, always use the `--output` option to save results instead of displaying them
4. **Memory issues with large videos**: Consider processing shorter clips or reducing the input resolution 