# ⚽ Advanced Football Player Detection & Analytics Platform

![Football Player Detection](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-brightgreen)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Analytics](https://img.shields.io/badge/Analytics-Professional-purple)
![Batch Processing](https://img.shields.io/badge/Processing-Batch%20Support-red)

## 🌟 Overview

A comprehensive football player detection and analytics platform built with YOLOv8. This professional-grade tool goes beyond simple detection to provide detailed performance analytics, visual insights, batch processing capabilities, and comprehensive reporting - perfect for sports analysts, researchers, and video processing workflows.

## 🎬 Demo

[![Football Player Detection Demo](https://img.youtube.com/vi/AXp5uXWnYQA/0.jpg)](https://youtu.be/AXp5uXWnYQA)
*Click on the image above to watch the demo video*

## ✨ Key Features

### 🔍 **Advanced Detection Capabilities**
- **High-Precision Player Detection**: State-of-the-art YOLOv8 model for accurate player identification
- **Real-time Processing**: Optimized for efficient video processing with FPS monitoring
- **Confidence Scoring**: Detailed confidence analysis with min/max/average metrics
- **Scene Complexity Analysis**: Automatic categorization of low/medium/high complexity scenes

### 📊 **Professional Analytics Dashboard**
- **Visual Charts Generation**: 4-panel analytics dashboard with matplotlib
- **Detection Timeline**: Player count variations across video frames
- **Confidence Distribution**: Statistical analysis of detection quality
- **Scene Complexity Breakdown**: Pie charts and histograms for comprehensive insights

### 🚀 **Performance Monitoring**
- **System Resource Tracking**: Real-time memory and CPU usage monitoring
- **Processing Speed Analysis**: FPS calculations and efficiency metrics
- **Performance Recommendations**: Intelligent suggestions for optimization
- **Quality Scoring**: Combined efficiency and quality assessment

### 📁 **Batch Processing & Automation**
- **Directory Processing**: Process entire folders of videos/images automatically
- **Multi-format Support**: Handles MP4, AVI, MOV, JPG, PNG, and more
- **Progress Tracking**: Visual progress bars and status updates
- **Error Resilience**: Continues processing despite individual file failures

### 📋 **Comprehensive Reporting**
- **JSON Export**: Structured data export for further analysis
- **Timestamped Reports**: Automatic report generation with metadata
- **Statistical Summaries**: Complete detection and performance statistics
- **Chart Generation**: High-resolution visualizations for presentations

### ⚙️ **Configuration Management**
- **YAML Configuration**: Persistent settings with `config.yaml`
- **Command-line Overrides**: Flexible parameter control
- **Feature Toggles**: Enable/disable charts, reports, monitoring
- **Default Generation**: Auto-creates configuration files

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Football-Player-Detection-YOLOv8.git
cd Football-Player-Detection-YOLOv8

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Dependencies
```
ultralytics>=8.0.0    # YOLOv8 framework
opencv-python>=4.6.0  # Computer vision
torch>=1.8.0          # Deep learning
matplotlib>=3.5.0     # Visualization
psutil>=5.8.0         # System monitoring
pyyaml>=6.0           # Configuration
numpy>=1.23.0         # Numerical computing
```

## 🚀 Usage

### Quick Start

```bash
# Single video processing
python simple_detection.py --input video.mp4

# Process with custom confidence threshold
python simple_detection.py --input video.mp4 --conf 0.5

# Batch process entire directory
python simple_detection.py --input /path/to/videos/ --output /path/to/results/

# Use custom model and configuration
python simple_detection.py --input video.mp4 --model custom_model.pt --config my_config.yaml
```

### Advanced Usage

```bash
# Batch processing with custom settings
python simple_detection.py --batch --input ./videos/ --output ./results/ --conf 0.3

# Skip chart generation for faster processing
python simple_detection.py --input video.mp4 --no-charts

# Process without JSON reports
python simple_detection.py --input video.mp4 --no-reports

# Use custom configuration file
python simple_detection.py --input video.mp4 --config production_config.yaml
```

### Configuration File (`config.yaml`)

```yaml
# Model settings
model_path: runs/detect/train2/weights/best.pt
confidence_threshold: 0.25

# Output directories
output_dir: outputs
reports_dir: reports

# Feature toggles
generate_charts: true
save_reports: true
show_progress: true
monitor_resources: true
```

## 📊 Output Examples

### Console Output
```
Loading model: runs/detect/train2/weights/best.pt
Model loaded successfully!

Current Configuration:
  Model: runs/detect/train2/weights/best.pt
  Confidence threshold: 0.25
  Generate charts: True
  Save reports: True
  Monitor resources: True

Video info: 1920x1080, 30 FPS, 900 frames
Initial memory usage: 156.8 MB
Initial CPU usage: 12.3%

Frame 450/900 | Players: 8 | Max: 12
[Frame 450] Memory: 201.2 MB, CPU: 45.7%

Processed 900 frames
Total player detections: 7,234
Maximum players detected in a single frame: 12
Average detection confidence: 0.847
Confidence range: 0.312 - 0.998
Detection rate: 89.7% of frames had players
Average players per frame: 8.04
Detection consistency (std dev): 2.31 players
Median players per frame: 8.0
Scene complexity distribution:
  Low (1-3 players): 67 frames
  Medium (4-6 players): 234 frames
  High (7+ players): 599 frames
Detection rate: 156.3 players/second
Processing speed: 18.45 FPS
Total processing time: 48.78 seconds

SYSTEM RESOURCE SUMMARY:
Memory used: 44.4 MB
Final memory: 201.2 MB
Peak CPU usage: 67.8%

============================================================
PERFORMANCE ANALYSIS REPORT
============================================================
Overall Efficiency Score: 148.32 detections/second
Quality Score: 0.759 (confidence × detection rate)

PERFORMANCE RECOMMENDATIONS:
  ✅ Processing speed is good (18.45 FPS)
  ✅ Detection confidence is excellent (0.847)
  ✅ Detection rate is high (89.7%)
============================================================

Detection charts saved to: charts/detection_charts_video_20241215_143045.png
Detection report saved to: reports/detection_report_video_20241215_143045.json
Detection completed successfully!
```

### Generated Files

#### 1. **Visual Analytics Dashboard**
- `charts/detection_charts_[filename]_[timestamp].png`
- 4-panel dashboard with timeline, distribution, and confidence analysis
- High-resolution (300 DPI) suitable for presentations

#### 2. **JSON Reports**
```json
{
  "metadata": {
    "input_file": "football_match.mp4",
    "timestamp": "2024-01-15T14:30:45",
    "processing_date": "2024-01-15 14:30:45"
  },
  "processing_stats": {
    "total_frames": 900,
    "processing_time_seconds": 48.78,
    "processing_speed_fps": 18.45,
    "memory_used_mb": 44.4,
    "final_memory_mb": 201.2
  },
  "detection_stats": {
    "total_detections": 7234,
    "detection_rate_percent": 89.7,
    "avg_players_per_frame": 8.04,
    "detections_per_second": 156.3
  },
  "confidence_stats": {
    "average_confidence": 0.847,
    "min_confidence": 0.312,
    "max_confidence": 0.998,
    "confidence_range": 0.686
  }
}
```

## 🎯 Use Cases

### 🏟️ **Sports Analytics**
- Player counting and density analysis
- Game tempo and intensity measurement
- Performance benchmarking across matches

### 🎥 **Video Production**
- Automated player detection for highlight reels
- Scene complexity analysis for editing decisions
- Quality assessment of footage

### 🔬 **Research & Development**
- Computer vision model evaluation
- Dataset analysis and validation
- Performance benchmarking studies

### 🏢 **Enterprise Applications**
- Batch processing of video archives
- Automated content analysis pipelines
- Quality control and monitoring

## 📈 Performance Metrics

The platform provides comprehensive performance insights:

- **Detection Quality**: Confidence scores, accuracy metrics
- **Processing Efficiency**: FPS, memory usage, CPU utilization
- **Scene Analysis**: Complexity distribution, player density
- **System Health**: Resource monitoring, optimization recommendations

## 🔧 Advanced Features

### Jupyter Notebook Integration
- Interactive analysis with `football-player-detection-yolov8.ipynb`
- Step-by-step processing with visualizations
- Experimental features and model comparisons

### Custom Model Support
- Load your own trained YOLOv8 models
- Fine-tuned models for specific scenarios
- Model performance comparison tools

### Multi-format Support
**Videos**: MP4, AVI, MOV, MKV, FLV, WMV
**Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP

## 🚀 Future Roadmap

- [ ] **Real-time Streaming**: Live video processing capabilities
- [ ] **Team Classification**: Distinguish between different teams
- [ ] **Player Tracking**: Maintain identity across frames
- [ ] **Action Recognition**: Detect specific player actions
- [ ] **Web Interface**: Browser-based processing dashboard
- [ ] **API Integration**: RESTful API for external applications
- [ ] **Cloud Processing**: Scalable cloud-based analysis
- [ ] **Mobile App**: Smartphone processing capabilities

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat(): add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Git Commit Format
```
feat(): add new feature or enhancement
fix(): fix bug or issue
refactor(): improve code structure without changing functionality
chore(): update dependencies, configs, or maintenance tasks
docs(): update documentation
test(): add or update tests
style(): formatting, missing semicolons, etc
perf(): performance improvements
ci(): CI/CD changes
build(): build system changes
```

## 📚 Documentation

- [Usage Guide](USAGE.md) - Detailed usage instructions
- [API Reference](docs/api.md) - Function and class documentation
- [Configuration Guide](docs/config.md) - Advanced configuration options
- [Performance Tuning](docs/performance.md) - Optimization techniques

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for the excellent YOLOv8 framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Matplotlib](https://matplotlib.org/) for visualization tools
- Open-source computer vision community for inspiration and support

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/Football-Player-Detection-YOLOv8/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/Football-Player-Detection-YOLOv8/discussions)
- 📧 **Email**: your.email@example.com
- 💬 **Discord**: [Join our community](https://discord.gg/your-invite)

---

⭐ **Star this repository if you find it useful!** ⭐

Made with ❤️ for the computer vision and sports analytics community.
