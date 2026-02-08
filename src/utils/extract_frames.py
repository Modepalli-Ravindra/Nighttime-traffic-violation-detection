import cv2
import os

def extract_frames(video_dir="input_videos", output_dir="training_data/images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    total_frames = 0
    
    for video_file in video_files:
        path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(path)
        
        frame_count = 0
        saved_count = 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30 # Fallback
        
        print(f"Processing {video_file} (FPS: {fps})...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save 1 frame every second (frame_count % fps == 0)
            if frame_count % fps == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame_{saved_count}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                saved_count += 1
                total_frames += 1
            
            frame_count += 1
            
        cap.release()
        print(f"  -> Extracted {saved_count} images from {video_file}")

    print(f"\nDone! Total extracted images: {total_frames}")
    print(f"You can find them in: {output_dir}")

if __name__ == "__main__":
    extract_frames()
