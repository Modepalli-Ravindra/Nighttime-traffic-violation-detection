import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict

class TrafficDetector:
    def __init__(self, model_path=None):
        # --- Model 1: Base Model for Vehicles (Context & Signal Jump via Line Cross) ---
        self.base_model_path = '../../yolov8n.pt'
        if not os.path.exists(self.base_model_path):
             # Try downloading or find in weights? usually it downloads automatically
             print("⚠️ yolov8n.pt not found locally, YOLO will attempt download.")
        
        print(f"🔄 Loading Base Model (Vehicles): {self.base_model_path}")
        self.base_model = YOLO(self.base_model_path)
        
        # --- Model 2: Custom Model for Violations (No Helmet, etc) ---
        # Priority 1: Check for latest trained model in runs/
        latest_model = r'../../runs/detect/traffic_night_model5/weights/best.pt'
        fallback_model = r'../../models/weights/custom_traffic.pt'

        if model_path is None:
            if os.path.exists(latest_model):
                print(f"✅ Found latest trained model at: {latest_model}")
                model_path = latest_model
            elif os.path.exists(fallback_model):
                print(f"⚠️ Latest model not found, falling back to: {fallback_model}")
                model_path = fallback_model
            else:
                 print("⚠️ No custom model found! Violation detection might utilize base model only.")
                 model_path = None
        
        self.violation_model = None
        if model_path:
            print(f"🔄 Loading Custom Model (Violations): {os.path.abspath(model_path)}")
            try:
                self.violation_model = YOLO(model_path)
                print(f"📋 Custom Model Classes: {self.violation_model.names}")
            except Exception as e:
                print(f"❌ Failed to load custom model: {e}")

        # Tracking logic
        self.track_history = defaultdict(lambda: [])
        
    def enhance_night_frame(self, frame):
        """Enhance low-light frames using Gamma Correction and CLAHE"""
        # 1. Gamma Correction (Brighten)
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        bright_frame = cv2.LUT(frame, table)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_frame

    def detect_violations(self, frame, enhanced_frame, frame_count):
        """
        Dual-Model Logic with Association:
        1. Base Model -> Detect & Track Vehicles (Get IDs) & People
        2. Signal Jump -> Only enabled when simulated Traffic Light is RED
        3. Triple Riding -> Heuristic (Person count on bike) + Custom Model
        """
        violations = []
        annotated_frame = frame.copy()
        
        height, width = frame.shape[:2]
        stop_line_y = int(height * 0.75)
        
        # --- TRAFFIC LIGHT SIMULATION (Internal Logic Only) ---
        # Toggle every 150 frames (approx 5 seconds at 30fps)
        is_red_light = (frame_count % 300) < 150
        
        # Draw Stop Line (Always Red for visibility)
        cv2.line(annotated_frame, (0, stop_line_y), (width, stop_line_y), (0, 0, 255), 3)
        
        # VISUALS REMOVED AS REQUESTED
        # light_color = (0, 0, 255) if is_red_light else (0, 255, 0)
        # status_text = "RED LIGHT" if is_red_light else "GREEN LIGHT"
        # cv2.circle(annotated_frame, (50, 50), 30, (0,0,0), -1)
        # cv2.circle(annotated_frame, (50, 50), 25, light_color, -1)
        # cv2.putText(annotated_frame, status_text, (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
        
        # Storage
        tracked_vehicles = {}
        persons = [] # [x1, y1, x2, y2]
        motorcycles = [] # {'id': id, 'box': [x1,y1,x2,y2]}

        # --- 1. Run Base Model (Vehicles & People) ---
        # Conf 0.25 to catch more people
        base_results = self.base_model.track(frame, persist=True, verbose=False, conf=0.25)
        
        for result in base_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.base_model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Collect People (Visual Debugging)
                if label == 'person':
                    persons.append([x1, y1, x2, y2])
                    # Draw Person in Yellow to debug (Optional: Remove if too cluttered, but keeping for now)
                    # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                
                # Check for vehicles
                if label in ['car', 'motorcycle', 'bus', 'truck', 'auto']:
                    if track_id is not None:
                        tracked_vehicles[track_id] = [x1, y1, x2, y2]
                    
                    if label == 'motorcycle':
                         motorcycles.append({'id': track_id, 'box': [x1, y1, x2, y2]})

                    center_y = (y1 + y2) / 2
                    color = (0, 255, 0) # Green default
                    
                    # --- SIGNAL JUMP LOGIC ---
                    # Only check if Light is RED (Internal Simulation)
                    if is_red_light:
                        if center_y > stop_line_y:
                             if track_id is not None:
                                violations.append({
                                    "type": "Signal Jump",
                                    "object": label,
                                    "bbox": [x1, y1, x2, y2],
                                    "track_id": track_id
                                })
                                color = (0, 0, 255)
                                cv2.putText(annotated_frame, "SIGNAL JUMP", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} {track_id}" if track_id else label
                    cv2.putText(annotated_frame, label_text, (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- TRIPLE RIDING HEURISTIC ---
        for bike in motorcycles:
            bx1, by1, bx2, by2 = bike['box']
            bike_id = bike['id']
            
            # Count people overlapping with the bike
            rider_count = 0
            for p in persons:
                px1, py1, px2, py2 = p
                
                # Check Intersection
                ix1 = max(bx1, px1)
                iy1 = max(by1, py1)
                ix2 = min(bx2, px2)
                iy2 = min(by2, py2)
                
                if ix2 > ix1 and iy2 > iy1:
                    # If there is ANY intersection, we count it as a rider candidate
                    # (Simple overlap check is often better for riders who sit ON top of the bike box)
                    rider_count += 1
            
            # --- DEBUG VISUALIZATION REMOVED ---
            # cv2.putText(annotated_frame, f"Riders:{rider_count}", (bx1, by1-15), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Heuristic Trigger: > 2 riders
            if rider_count > 2:
                 violations.append({
                    "type": "Triple Riding",
                    "object": "motorcycle",
                    "bbox": bike['box'],
                    "track_id": bike_id
                 })
                 cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 3)
                 cv2.putText(annotated_frame, f"TRIPLE RIDING", (bx1, by1-60), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- 2. Run Custom Model (Violations) ---
        if self.violation_model:
            # Custom Model Logic
            # LOWER CONFIDENCE significantly to catch missed detections
            custom_results = self.violation_model.predict(enhanced_frame, conf=0.10, verbose=False) 
            
            for result in custom_results:
                boxes = result.boxes
                for box in boxes:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = self.violation_model.names[cls]
                    
                    # Find ID
                    v_center_x = (vx1 + vx2) / 2
                    v_center_y = (vy1 + vy2) / 2
                    assigned_id = None
                    for vid, vbox in tracked_vehicles.items():
                        if vbox[0] < v_center_x < vbox[2] and vbox[1] < v_center_y < vbox[3]:
                            assigned_id = vid
                            break
                    
                    violation_obj = {
                        "object": label,
                        "bbox": [vx1, vy1, vx2, vy2],
                        "track_id": assigned_id if assigned_id else f"loc_{int(v_center_x)}_{int(v_center_y)}"
                    }

                    is_violation = False
                    label_lower = label.lower()
                    
                    if 'helmet' in label_lower: 
                        violation_obj["type"] = "No Helmet"
                        is_violation = True
                    elif 'triple' in label_lower:
                        violation_obj["type"] = "Triple Riding" 
                        is_violation = True
                    elif 'jump' in label_lower or 'signal' in label_lower:
                        # Custom model signal jump usually better than logic
                        violation_obj["type"] = "Signal Jump"
                        is_violation = True

                    if is_violation:
                        cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                        t_id_str = f"ID:{assigned_id}" if assigned_id else ""
                        cv2.putText(annotated_frame, f"{violation_obj['type']} {t_id_str}", (vx1, vy1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        violations.append(violation_obj)

        return annotated_frame, violations

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
             print(f"ERROR: Could not open video file: {input_path}")
             return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Guard
        if width == 0 or height == 0:
             cap.release()
             return

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            enhanced = self.enhance_night_frame(frame)
            processed_frame, violations = self.detect_violations(frame, enhanced, frame_count)
            
            out.write(processed_frame)
            yield processed_frame, violations
            
        cap.release()
        out.release()