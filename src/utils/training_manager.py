import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
from pathlib import Path
import yaml

class TrainingManager:
    def __init__(self, base_dir=None):
        """Initialize the training manager with proper directory structure."""
        if base_dir is None:
            # Use the src directory structure
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
        
        self.base_dir = os.path.abspath(base_dir)
        self.violation_types = [
            'signal_jump',
            'no_helmet', 
            'triple_riding',
            'traffic_helmet',
            'multiple_violations'
        ]
        
        # Create directory structure if it doesn't exist
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create the required directory structure for training data."""
        for violation_type in self.violation_types:
            images_dir = os.path.join(self.base_dir, violation_type, 'images')
            videos_dir = os.path.join(self.base_dir, violation_type, 'videos')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            
        print(f"Training directory structure created at: {self.base_dir}")
        
    def organize_training_data(self, source_path, violation_type, file_type='images'):
        """
        Organize training data by moving files to appropriate directories.
        
        Args:
            source_path (str): Path to source files
            violation_type (str): Type of violation (must be one of self.violation_types)
            file_type (str): 'images' or 'videos'
        """
        if violation_type not in self.violation_types:
            raise ValueError(f"Invalid violation type. Must be one of: {self.violation_types}")
            
        if file_type not in ['images', 'videos']:
            raise ValueError("file_type must be 'images' or 'videos'")
            
        source_path = os.path.abspath(source_path)
        target_dir = os.path.join(self.base_dir, violation_type, file_type)
        
        if os.path.isfile(source_path):
            # Single file
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
            print(f"Copied {filename} to {target_dir}")
            
        elif os.path.isdir(source_path):
            # Directory of files
            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                if os.path.isfile(source_file):
                    target_path = os.path.join(target_dir, filename)
                    shutil.copy2(source_file, target_path)
                    print(f"Copied {filename} to {target_dir}")
        else:
            print(f"Source path {source_path} not found")
            
    def extract_frames_from_videos(self, violation_type=None):
        """
        Extract frames from videos for training data.
        
        Args:
            violation_type (str, optional): Specific violation type to process
        """
        types_to_process = [violation_type] if violation_type else self.violation_types
        
        for v_type in types_to_process:
            videos_dir = os.path.join(self.base_dir, v_type, 'videos')
            images_dir = os.path.join(self.base_dir, v_type, 'images')
            
            if not os.path.exists(videos_dir):
                continue
                
            video_files = [f for f in os.listdir(videos_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            for video_file in video_files:
                video_path = os.path.join(videos_dir, video_file)
                self._extract_frames_from_video(video_path, images_dir, v_type)
                
    def _extract_frames_from_video(self, video_path, output_dir, violation_type, frame_interval=30):
        """
        Extract frames from a single video file.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames
            violation_type (str): Violation type for naming
            frame_interval (int): Extract every nth frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
            
        frame_count = 0
        extracted_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"{violation_type}_{video_name}_frame_{extracted_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {extracted_count} frames from {video_name} for {violation_type}")
        
    def create_dataset_yaml(self, output_path=None):
        """
        Create YAML configuration file for YOLO training.
        
        Args:
            output_path (str, optional): Path to save the YAML file
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.yaml')
            
        # Count images in each category
        class_counts = {}
        for i, violation_type in enumerate(self.violation_types):
            images_dir = os.path.join(self.base_dir, violation_type, 'images')
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[violation_type] = image_count
            else:
                class_counts[violation_type] = 0
                
        # Create YAML content
        yaml_content = {
            'path': self.base_dir,
            'train': '.',  # Will be set to training directory
            'val': '.',    # Will be set to validation directory
            'nc': len(self.violation_types),
            'names': self.violation_types
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f"Dataset YAML created at: {output_path}")
        print("Class counts:")
        for violation_type, count in class_counts.items():
            print(f"  {violation_type}: {count} images")
            
        return output_path
        
    def prepare_training_data(self, train_ratio=0.8):
        """
        Split data into training and validation sets.
        
        Args:
            train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
        """
        train_dir = os.path.join(self.base_dir, 'train')
        val_dir = os.path.join(self.base_dir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for violation_type in self.violation_types:
            images_dir = os.path.join(self.base_dir, violation_type, 'images')
            if not os.path.exists(images_dir):
                continue
                
            # Create violation type directories in train/val
            train_violation_dir = os.path.join(train_dir, violation_type)
            val_violation_dir = os.path.join(val_dir, violation_type)
            os.makedirs(train_violation_dir, exist_ok=True)
            os.makedirs(val_violation_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle and split
            np.random.shuffle(image_files)
            split_index = int(len(image_files) * train_ratio)
            
            train_files = image_files[:split_index]
            val_files = image_files[split_index:]
            
            # Copy files
            for file in train_files:
                src = os.path.join(images_dir, file)
                dst = os.path.join(train_violation_dir, file)
                shutil.copy2(src, dst)
                
            for file in val_files:
                src = os.path.join(images_dir, file)
                dst = os.path.join(val_violation_dir, file)
                shutil.copy2(src, dst)
                
            print(f"{violation_type}: {len(train_files)} train, {len(val_files)} val")
            
        # Update YAML file paths
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                
            yaml_content['train'] = 'train'
            yaml_content['val'] = 'val'
            
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
                
    def train_model(self, model_path='yolov8n.pt', epochs=100, imgsz=640, batch_size=16):
        """
        Train the YOLO model with the prepared dataset.
        
        Args:
            model_path (str): Path to pre-trained model
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size for training
        """
        # Create dataset YAML
        yaml_path = self.create_dataset_yaml()
        
        # Initialize model
        model = YOLO(model_path)
        
        # Train model
        print("Starting model training...")
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='traffic_violation_model',
            exist_ok=True
        )
        
        print("Training completed!")
        print(f"Model saved to: {results.save_dir}")
        
        return results
        
    def get_training_stats(self):
        """Get statistics about the training data."""
        stats = {}
        
        for violation_type in self.violation_types:
            images_dir = os.path.join(self.base_dir, violation_type, 'images')
            videos_dir = os.path.join(self.base_dir, violation_type, 'videos')
            
            image_count = 0
            video_count = 0
            
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if os.path.exists(videos_dir):
                video_count = len([f for f in os.listdir(videos_dir) 
                                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
            
            stats[violation_type] = {
                'images': image_count,
                'videos': video_count,
                'total': image_count + video_count
            }
            
        return stats

# Example usage function
def setup_training_example():
    """Example of how to use the TrainingManager."""
    # Initialize training manager
    trainer = TrainingManager()
    
    # Example: Organize some sample data
    # trainer.organize_training_data('/path/to/signal_jump_images', 'signal_jump', 'images')
    # trainer.organize_training_data('/path/to/helmet_videos', 'no_helmet', 'videos')
    
    # Extract frames from videos
    # trainer.extract_frames_from_videos()
    
    # Prepare training data
    # trainer.prepare_training_data(train_ratio=0.8)
    
    # Train model
    # trainer.train_model(epochs=50)
    
    # Get statistics
    stats = trainer.get_training_stats()
    print("\nTraining Data Statistics:")
    for violation_type, data in stats.items():
        print(f"{violation_type}: {data['images']} images, {data['videos']} videos")

if __name__ == "__main__":
    setup_training_example()