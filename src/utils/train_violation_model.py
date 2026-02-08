#!/usr/bin/env python3
"""
Traffic Violation Detection Model Training Script

This script provides a complete workflow for training a YOLO model 
to detect traffic violations using organized training data.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import training_manager
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

from utils.training_manager import TrainingManager

def main():
    """Main training workflow."""
    print("=== Traffic Violation Detection Model Training ===\n")
    
    # Initialize training manager
    trainer = TrainingManager()
    
    # Show current data statistics
    print("Current Training Data Status:")
    stats = trainer.get_training_stats()
    total_images = 0
    total_videos = 0
    
    for violation_type, data in stats.items():
        print(f"  {violation_type}: {data['images']} images, {data['videos']} videos")
        total_images += data['images']
        total_videos += data['videos']
    
    print(f"\nTotal: {total_images} images, {total_videos} videos\n")
    
    # Check if we have enough data to train
    if total_images < 100:
        print("⚠️  Warning: Less than 100 images available for training.")
        print("Please add more training data to the directories before training.\n")
        return
    
    # Ask user what they want to do
    print("Available actions:")
    print("1. Extract frames from videos")
    print("2. Prepare training/validation split")
    print("3. Train model")
    print("4. Show statistics")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\nExtracting frames from videos...")
                trainer.extract_frames_from_videos()
                print("Frame extraction completed!")
                
            elif choice == '2':
                train_ratio = input("Enter train ratio (default 0.8): ").strip()
                train_ratio = float(train_ratio) if train_ratio else 0.8
                print(f"\nPreparing training data with {train_ratio*100}% train split...")
                trainer.prepare_training_data(train_ratio)
                print("Data preparation completed!")
                
            elif choice == '3':
                epochs = input("Enter number of epochs (default 100): ").strip()
                epochs = int(epochs) if epochs else 100
                
                batch_size = input("Enter batch size (default 16): ").strip()
                batch_size = int(batch_size) if batch_size else 16
                
                print(f"\nStarting training with {epochs} epochs and batch size {batch_size}...")
                results = trainer.train_model(epochs=epochs, batch_size=batch_size)
                print("Training completed successfully!")
                
            elif choice == '4':
                print("\nCurrent Training Data Statistics:")
                stats = trainer.get_training_stats()
                for violation_type, data in stats.items():
                    print(f"  {violation_type}: {data['images']} images, {data['videos']} videos")
                    
            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def add_sample_data_example():
    """Example of how to add sample data to training directories."""
    print("\n=== Adding Sample Training Data ===")
    
    trainer = TrainingManager()
    
    # Example paths - you would replace these with your actual data paths
    sample_data = {
        'signal_jump': {
            'images': r'C:\path\to\your\signal_jump_images',
            'videos': r'C:\path\to\your\signal_jump_videos'
        },
        'no_helmet': {
            'images': r'C:\path\to\your\helmet_violation_images',
            'videos': r'C:\path\to\your\helmet_violation_videos'
        },
        'triple_riding': {
            'images': r'C:\path\to\your\triple_riding_images',
            'videos': r'C:\path\to\your\triple_riding_videos'
        },
        'traffic_helmet': {
            'images': r'C:\path\to\your\traffic_helmet_images',
            'videos': r'C:\path\to\your\traffic_helmet_videos'
        },
        'multiple_violations': {
            'images': r'C:\path\to\your\multiple_violations_images',
            'videos': r'C:\path\to\your\multiple_violations_videos'
        }
    }
    
    print("To add your training data, use the organize_training_data method:")
    print("Example:")
    print("trainer.organize_training_data('/path/to/your/images', 'signal_jump', 'images')")
    print("trainer.organize_training_data('/path/to/your/videos', 'no_helmet', 'videos')")
    
    # Uncomment and modify the following lines to add your actual data
    """
    for violation_type, sources in sample_data.items():
        if os.path.exists(sources['images']):
            print(f"Adding images for {violation_type}...")
            trainer.organize_training_data(sources['images'], violation_type, 'images')
            
        if os.path.exists(sources['videos']):
            print(f"Adding videos for {violation_type}...")
            trainer.organize_training_data(sources['videos'], violation_type, 'videos')
    """

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add_data":
        add_sample_data_example()
    else:
        main()