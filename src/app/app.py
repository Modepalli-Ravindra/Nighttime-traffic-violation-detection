from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import os
import cv2
import time
from werkzeug.utils import secure_filename
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.detector import TrafficDetector

def create_app(template_folder=None, static_folder=None):
    app = Flask(__name__, 
                template_folder=template_folder,
                static_folder=static_folder)

    # Config
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'input')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'data', 'output')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

    # Ensure dirs exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize Detector (Removed global instance to prevent state issues)
    # detector = TrafficDetector()

    # Global Stats Store
    current_stats = {
        'signal': 0,
        'helmet': 0,
        'triple': 0,
        'traffic_helmet': 0,  # Renamed from 'combined'
        'multiple': 0  # New counter for multiple violations
    }

    # Dictionary to track violations per vehicle
    vehicle_violations = {}

    @app.route('/')
    def index():
        return render_template('index.html')

    def generate_frames(path):
        global current_stats, vehicle_violations
        # Reset stats for new video
        current_stats = {'signal': 0, 'helmet': 0, 'triple': 0, 'traffic_helmet': 0, 'multiple': 0}
        vehicle_violations = {}
        
        # Instantiate detector for this specific session to ensure clean state
        # This might add a small delay on start, but ensures tracking is fresh and robust
        local_detector = TrafficDetector()
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + os.path.basename(path))
        
        try:
            # Process
            for frame, violations in local_detector.process_video(path, output_path):
                
                # Update Stats from Violations (with enhanced tracking)
                for v in violations:
                    v_type = v.get('type', '')
                    track_id = v.get('track_id')
                
                    # Track violations per vehicle
                    if track_id is not None:
                        if track_id not in vehicle_violations:
                            vehicle_violations[track_id] = set()
                        
                        # Add this violation type to the vehicle's violations
                        vehicle_violations[track_id].add(v_type)
                        
                        # Create a unique key for this specific violation instance
                        violation_key = f"{track_id}_{v_type}"
                        
                        # Only count if we haven't seen this ID + Violation combination yet
                        if violation_key not in [f"{tid}_{vt}" for tid in vehicle_violations.keys() for vt in vehicle_violations[tid] if tid == track_id and vt == v_type]:
                            # Count individual violations
                            if 'Signal' in v_type: 
                                current_stats['signal'] += 1
                            elif 'Helmet' in v_type: 
                                current_stats['helmet'] += 1
                            elif 'Triple' in v_type: 
                                current_stats['triple'] += 1
                
                # Update multiple violations counter
                for track_id, violations_set in vehicle_violations.items():
                    if len(violations_set) >= 2:  # Vehicle has multiple violations
                        # Create unique key for multiple violations tracking
                        multiple_key = f"{track_id}_multiple"
                        if multiple_key not in [f"{tid}_multiple" for tid in vehicle_violations.keys()]:
                            current_stats['multiple'] += 1
                
                # Update traffic_helmet counter (vehicles with both signal and helmet violations)
                for track_id, violations_set in vehicle_violations.items():
                    if 'Signal' in str(violations_set) and 'Helmet' in str(violations_set):
                        traffic_helmet_key = f"{track_id}_traffic_helmet"
                        if traffic_helmet_key not in [f"{tid}_traffic_helmet" for tid in vehicle_violations.keys()]:
                            current_stats['traffic_helmet'] += 1
            
                # Encode frame for web
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        except Exception as e:
            print(f"Error in video processing: {e}")
        finally:
            print("Finished processing video request.")

    @app.route('/upload', methods=['POST'])
    def upload_video():
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            # Ensure filename is not empty after secure_filename
            if not filename:
                filename = f"video_{int(time.time())}.mp4"
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"DEBUG: Saving file to {filepath}")
            file.save(filepath)
            return jsonify({'message': 'File uploaded successfully', 'filepath': filename})

    @app.route('/video_feed')
    def video_feed():
        filename = request.args.get('path')
        if not filename:
            return "Error: No path provided", 400
        
        # Reconstruct absolute path
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"DEBUG: Processing video path: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found at {video_path}")
            return "Error: File not found", 404
            
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/stats')
    def get_stats():
        return jsonify(current_stats)

    return app

if __name__ == '__main__':
    # For backward compatibility when running directly
    app_instance = create_app('../../ui/templates', '../../ui/static')
    app_instance.run(debug=True, port=5000)