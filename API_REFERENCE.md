# Football Player Detection API Reference

## Overview

The Football Player Detection system provides both a web dashboard and REST API for processing football videos and images. This document covers all available endpoints and their usage.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, no authentication is required. In production, implement proper authentication and authorization.

## Web Dashboard Routes

### GET /
**Main Dashboard**
- Returns the main dashboard page with system overview
- Shows real-time processing status and recent jobs

### GET /upload
**File Upload Page**
- Drag-and-drop interface for uploading files
- Supports batch uploads and progress tracking

### GET /history
**Processing History**
- Complete history of all processing jobs
- Filterable and searchable results

### GET /analytics
**Analytics Dashboard**
- Charts and statistics about processing performance
- System metrics and trends

## REST API Endpoints

### File Management

#### POST /api/upload
**Upload a file for processing**

**Request:**
```http
POST /api/upload
Content-Type: multipart/form-data

file: [binary file data]
```

**Response:**
```json
{
  "success": true,
  "filename": "20241215_143045_football_match.mp4",
  "job_id": 123,
  "file_info": {
    "size": 52428800,
    "type": "video",
    "modified": "2024-12-15T14:30:45"
  }
}
```

**Supported Formats:**
- **Videos**: MP4, AVI, MOV, MKV, FLV, WMV
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Max Size**: 500MB per file

### Processing Control

#### GET /api/process/{job_id}
**Start processing a specific file**

**Parameters:**
- `job_id` (integer): ID of the uploaded file

**Response:**
```json
{
  "success": true,
  "message": "Processing started"
}
```

**Error Response:**
```json
{
  "error": "Another file is currently being processed"
}
```

#### GET /api/batch_process
**Start batch processing of all uploaded files**

**Response:**
```json
{
  "success": true,
  "message": "Batch processing started"
}
```

### Status and Monitoring

#### GET /api/status
**Get current processing status**

**Response:**
```json
{
  "is_processing": true,
  "current_file": "football_match.mp4",
  "progress": 45,
  "total_files": 5,
  "processed_files": 2,
  "current_frame": 450,
  "total_frames": 1000,
  "start_time": "2024-12-15T14:30:45",
  "estimated_completion": "2024-12-15T14:35:30"
}
```

#### GET /api/jobs
**Get list of processing jobs**

**Response:**
```json
[
  {
    "id": 123,
    "filename": "football_match.mp4",
    "file_type": "video",
    "status": "completed",
    "upload_time": "2024-12-15T14:30:45",
    "processing_start": "2024-12-15T14:31:00",
    "processing_end": "2024-12-15T14:33:15",
    "total_detections": 2847,
    "avg_confidence": 0.847,
    "detection_rate": 89.7,
    "processing_speed": 18.45
  }
]
```

**Job Status Values:**
- `uploaded`: File uploaded, ready for processing
- `processing`: Currently being processed
- `completed`: Processing completed successfully
- `failed`: Processing failed with error

### Analytics and Reports

#### GET /api/analytics
**Get system analytics and statistics**

**Response:**
```json
{
  "summary": {
    "total_jobs": 150,
    "completed_jobs": 142,
    "failed_jobs": 3,
    "avg_processing_speed": 18.45,
    "avg_confidence": 0.847,
    "total_detections": 425000
  },
  "daily_stats": [
    {
      "date": "2024-12-15",
      "jobs": 25
    }
  ]
}
```

#### GET /api/download/{job_id}
**Download processing results**

**Parameters:**
- `job_id` (integer): ID of the completed job

**Response:**
- Returns the processed video/image file as download
- Filename format: `processed_{original_filename}`

## WebSocket Events

The system uses WebSocket connections for real-time updates.

### Client Events

#### connect
**Establish WebSocket connection**
- Automatically sends current status upon connection

#### request_status
**Request current processing status**
- Server responds with `status_update` event

### Server Events

#### status_update
**Real-time processing status updates**

**Payload:**
```json
{
  "is_processing": true,
  "current_file": "football_match.mp4",
  "progress": 45,
  "total_files": 5,
  "processed_files": 2
}
```

#### job_completed
**Notification when a job completes**

**Payload:**
```json
{
  "job_id": 123,
  "status": "completed"
}
```

#### batch_completed
**Notification when batch processing completes**

**Payload:**
```json
{
  "total_processed": 5
}
```

## Database Schema

### processing_jobs Table

```sql
CREATE TABLE processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_start TIMESTAMP,
    processing_end TIMESTAMP,
    status TEXT DEFAULT 'pending',
    total_frames INTEGER,
    total_detections INTEGER,
    avg_confidence REAL,
    detection_rate REAL,
    processing_speed REAL,
    file_size INTEGER,
    output_path TEXT,
    error_message TEXT
);
```

### system_metrics Table

```sql
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage REAL,
    memory_usage REAL,
    gpu_usage REAL,
    processing_fps REAL,
    active_jobs INTEGER
);
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Processing already in progress
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": "Error message description",
  "code": "ERROR_CODE",
  "details": {
    "additional": "error details"
  }
}
```

## Rate Limiting

Currently no rate limiting is implemented. In production:
- Implement rate limiting per IP
- Limit concurrent uploads
- Queue management for processing jobs

## Usage Examples

### Python Client Example

```python
import requests
import json

# Upload a file
with open('football_video.mp4', 'rb') as f:
    response = requests.post('http://localhost:5000/api/upload', 
                           files={'file': f})
    job_data = response.json()
    job_id = job_data['job_id']

# Start processing
response = requests.get(f'http://localhost:5000/api/process/{job_id}')

# Check status
response = requests.get('http://localhost:5000/api/status')
status = response.json()

# Download results (when completed)
response = requests.get(f'http://localhost:5000/api/download/{job_id}')
with open('processed_video.mp4', 'wb') as f:
    f.write(response.content)
```

### JavaScript Client Example

```javascript
// Upload file with progress tracking
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const xhr = new XMLHttpRequest();
xhr.upload.addEventListener('progress', (e) => {
    const progress = (e.loaded / e.total) * 100;
    console.log(`Upload progress: ${progress}%`);
});

xhr.onload = () => {
    const response = JSON.parse(xhr.responseText);
    console.log('Upload complete:', response);
};

xhr.open('POST', '/api/upload');
xhr.send(formData);

// WebSocket connection for real-time updates
const socket = io();
socket.on('status_update', (status) => {
    console.log('Processing status:', status);
});
```

## Configuration

### Environment Variables

```bash
# Flask configuration
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-secret-key-here

# File upload settings
MAX_CONTENT_LENGTH=524288000  # 500MB
UPLOAD_FOLDER=uploads

# Database settings
DATABASE_URL=sqlite:///detection_history.db

# Model settings
MODEL_PATH=runs/detect/train2/weights/best.pt
CONFIDENCE_THRESHOLD=0.25
```

### Configuration File (config.yaml)

```yaml
# Web dashboard settings
web_dashboard:
  host: 0.0.0.0
  port: 5000
  debug: false

# Processing settings
processing:
  model_path: runs/detect/train2/weights/best.pt
  confidence_threshold: 0.25
  max_concurrent_jobs: 1
  
# Storage settings
storage:
  upload_dir: uploads
  output_dir: outputs
  reports_dir: reports
  charts_dir: charts
  
# Features
features:
  generate_charts: true
  save_reports: true
  export_csv: true
  email_notifications: false
```

## Security Considerations

### Production Deployment

1. **Authentication**: Implement user authentication and authorization
2. **File Validation**: Strict file type and size validation
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **HTTPS**: Use HTTPS in production
5. **Input Sanitization**: Sanitize all user inputs
6. **File Scanning**: Scan uploaded files for malware
7. **Access Control**: Implement proper access controls

### Recommended Security Headers

```python
@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response
```

## Performance Optimization

### Recommendations

1. **Caching**: Implement Redis for session and result caching
2. **Queue System**: Use Celery for background job processing
3. **Load Balancing**: Use nginx for load balancing multiple instances
4. **Database**: Use PostgreSQL for better performance at scale
5. **CDN**: Use CDN for static assets
6. **Monitoring**: Implement comprehensive monitoring and logging

### Scaling Considerations

- Horizontal scaling with multiple worker processes
- Separate processing workers from web interface
- Distributed file storage for large deployments
- Database connection pooling
- Asynchronous processing with job queues

---

For more information, see the main [README.md](README.md) file. 