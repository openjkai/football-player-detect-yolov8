#!/usr/bin/env python3
"""
Football Player Detection Web Dashboard

A comprehensive web-based interface for the football player detection system.
Provides real-time monitoring, file upload, batch processing, and results visualization.

Usage:
    python web_dashboard.py
    
Then navigate to http://localhost:5000 in your browser.
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import sqlite3
from simple_detection import YOLO, detect_on_video, detect_on_image, load_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'football_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for real-time monitoring
processing_status = {
    'is_processing': False,
    'current_file': None,
    'progress': 0,
    'total_files': 0,
    'processed_files': 0,
    'current_frame': 0,
    'total_frames': 0,
    'start_time': None,
    'estimated_completion': None
}

# Database setup
def init_database():
    """Initialize SQLite database for storing processing history"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_jobs (
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
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cpu_usage REAL,
            memory_usage REAL,
            gpu_usage REAL,
            processing_fps REAL,
            active_jobs INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    """Get file information"""
    file_stats = os.stat(filepath)
    return {
        'size': file_stats.st_size,
        'modified': datetime.fromtimestamp(file_stats.st_mtime),
        'type': 'video' if filepath.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'] else 'image'
    }

def update_processing_status(status_update):
    """Update processing status and emit to connected clients"""
    global processing_status
    processing_status.update(status_update)
    socketio.emit('status_update', processing_status)

def save_job_to_db(job_data):
    """Save processing job to database"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO processing_jobs 
        (filename, file_type, file_size, status, processing_start, processing_end,
         total_frames, total_detections, avg_confidence, detection_rate, processing_speed, output_path, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        job_data.get('filename'),
        job_data.get('file_type'),
        job_data.get('file_size'),
        job_data.get('status'),
        job_data.get('processing_start'),
        job_data.get('processing_end'),
        job_data.get('total_frames'),
        job_data.get('total_detections'),
        job_data.get('avg_confidence'),
        job_data.get('detection_rate'),
        job_data.get('processing_speed'),
        job_data.get('output_path'),
        job_data.get('error_message')
    ))
    
    conn.commit()
    job_id = cursor.lastrowid
    conn.close()
    return job_id

# Routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')

@app.route('/history')
def history_page():
    """Processing history page"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM processing_jobs 
        ORDER BY upload_time DESC 
        LIMIT 100
    ''')
    
    jobs = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', jobs=jobs)

@app.route('/analytics')
def analytics_page():
    """Analytics and statistics page"""
    return render_template('analytics.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file info and save to database
        file_info = get_file_info(Path(filepath))
        job_data = {
            'filename': filename,
            'file_type': file_info['type'],
            'file_size': file_info['size'],
            'status': 'uploaded'
        }
        job_id = save_job_to_db(job_data)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'job_id': job_id,
            'file_info': file_info
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/process/<int:job_id>')
def process_file(job_id):
    """Start processing a specific file"""
    if processing_status['is_processing']:
        return jsonify({'error': 'Another file is currently being processed'}), 409
    
    # Start processing in background thread
    thread = threading.Thread(target=background_process_file, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})

@app.route('/api/batch_process')
def batch_process():
    """Start batch processing of all uploaded files"""
    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress'}), 409
    
    # Start batch processing in background thread
    thread = threading.Thread(target=background_batch_process)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Batch processing started'})

@app.route('/api/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_status)

@app.route('/api/jobs')
def get_jobs():
    """Get recent processing jobs"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, file_type, status, upload_time, processing_start, processing_end,
               total_detections, avg_confidence, detection_rate, processing_speed
        FROM processing_jobs 
        ORDER BY upload_time DESC 
        LIMIT 50
    ''')
    
    jobs = []
    for row in cursor.fetchall():
        jobs.append({
            'id': row[0],
            'filename': row[1],
            'file_type': row[2],
            'status': row[3],
            'upload_time': row[4],
            'processing_start': row[5],
            'processing_end': row[6],
            'total_detections': row[7],
            'avg_confidence': row[8],
            'detection_rate': row[9],
            'processing_speed': row[10]
        })
    
    conn.close()
    return jsonify(jobs)

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    # Get summary statistics
    cursor.execute('''
        SELECT 
            COUNT(*) as total_jobs,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
            AVG(CASE WHEN status = 'completed' THEN processing_speed ELSE NULL END) as avg_processing_speed,
            AVG(CASE WHEN status = 'completed' THEN avg_confidence ELSE NULL END) as avg_confidence,
            SUM(CASE WHEN status = 'completed' THEN total_detections ELSE 0 END) as total_detections
        FROM processing_jobs
    ''')
    
    summary = cursor.fetchone()
    
    # Get processing history for charts
    cursor.execute('''
        SELECT DATE(upload_time) as date, COUNT(*) as jobs_count
        FROM processing_jobs 
        WHERE upload_time >= date('now', '-30 days')
        GROUP BY DATE(upload_time)
        ORDER BY date
    ''')
    
    daily_stats = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'summary': {
            'total_jobs': summary[0] or 0,
            'completed_jobs': summary[1] or 0,
            'failed_jobs': summary[2] or 0,
            'avg_processing_speed': round(summary[3] or 0, 2),
            'avg_confidence': round(summary[4] or 0, 3),
            'total_detections': summary[5] or 0
        },
        'daily_stats': [{'date': row[0], 'jobs': row[1]} for row in daily_stats]
    })

@app.route('/api/download/<int:job_id>')
def download_results(job_id):
    """Download processing results"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT output_path, filename FROM processing_jobs WHERE id = ?', (job_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0] and os.path.exists(result[0]):
        return send_file(result[0], as_attachment=True, download_name=f"processed_{result[1]}")
    
    return jsonify({'error': 'File not found'}), 404

# Background processing functions
def background_process_file(job_id):
    """Process a single file in background"""
    global processing_status
    
    try:
        # Get job details from database
        conn = sqlite3.connect('detection_history.db')
        cursor = conn.cursor()
        cursor.execute('SELECT filename, file_type FROM processing_jobs WHERE id = ?', (job_id,))
        job = cursor.fetchone()
        conn.close()
        
        if not job:
            return
        
        filename, file_type = job
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Update status
        update_processing_status({
            'is_processing': True,
            'current_file': filename,
            'progress': 0,
            'start_time': datetime.now().isoformat()
        })
        
        # Load model and config
        config = load_config()
        model = YOLO(config['model_path'])
        
        # Process file
        processing_start = datetime.now()
        
        if file_type == 'video':
            # For video processing, we'll need to modify detect_on_video to provide progress updates
            # This is a simplified version - in practice, you'd need to modify the detection function
            result = detect_on_video(model, filepath, None, config['confidence_threshold'])
        else:
            result = detect_on_image(model, filepath, None, config['confidence_threshold'])
        
        processing_end = datetime.now()
        
        # Update database with results
        job_data = {
            'status': 'completed',
            'processing_start': processing_start,
            'processing_end': processing_end,
            'total_frames': getattr(result, 'total_frames', 1),
            'total_detections': getattr(result, 'total_detections', 0),
            'avg_confidence': getattr(result, 'avg_confidence', 0),
            'detection_rate': getattr(result, 'detection_rate', 0),
            'processing_speed': getattr(result, 'processing_speed', 0)
        }
        
        conn = sqlite3.connect('detection_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE processing_jobs 
            SET status = ?, processing_start = ?, processing_end = ?,
                total_frames = ?, total_detections = ?, avg_confidence = ?,
                detection_rate = ?, processing_speed = ?
            WHERE id = ?
        ''', (
            job_data['status'], job_data['processing_start'], job_data['processing_end'],
            job_data['total_frames'], job_data['total_detections'], job_data['avg_confidence'],
            job_data['detection_rate'], job_data['processing_speed'], job_id
        ))
        conn.commit()
        conn.close()
        
        # Update final status
        update_processing_status({
            'is_processing': False,
            'current_file': None,
            'progress': 100
        })
        
        socketio.emit('job_completed', {'job_id': job_id, 'status': 'completed'})
        
    except Exception as e:
        # Handle errors
        conn = sqlite3.connect('detection_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE processing_jobs 
            SET status = ?, error_message = ?
            WHERE id = ?
        ''', ('failed', str(e), job_id))
        conn.commit()
        conn.close()
        
        update_processing_status({
            'is_processing': False,
            'current_file': None,
            'progress': 0
        })
        
        socketio.emit('job_completed', {'job_id': job_id, 'status': 'failed', 'error': str(e)})

def background_batch_process():
    """Process all pending files in background"""
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM processing_jobs WHERE status = "uploaded"')
    pending_jobs = cursor.fetchall()
    conn.close()
    
    total_jobs = len(pending_jobs)
    
    update_processing_status({
        'is_processing': True,
        'total_files': total_jobs,
        'processed_files': 0,
        'start_time': datetime.now().isoformat()
    })
    
    for i, (job_id,) in enumerate(pending_jobs):
        update_processing_status({
            'processed_files': i,
            'progress': int((i / total_jobs) * 100) if total_jobs > 0 else 0
        })
        
        background_process_file(job_id)
        
        # Small delay between files
        time.sleep(1)
    
    update_processing_status({
        'is_processing': False,
        'processed_files': total_jobs,
        'progress': 100
    })
    
    socketio.emit('batch_completed', {'total_processed': total_jobs})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', processing_status)

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('status_update', processing_status)

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting Football Player Detection Web Dashboard...")
    print("Navigate to http://localhost:5000 in your browser")
    
    # Run the Flask app with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 