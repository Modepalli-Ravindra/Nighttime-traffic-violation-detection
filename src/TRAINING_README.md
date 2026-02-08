# Traffic Violation Detection Model Training

This document explains how to train a custom YOLO model for traffic violation detection using the organized training data structure.

## Directory Structure

The training data is organized in the following structure:

```
src/
└── data/
    └── training/
        ├── signal_jump/
        │   ├── images/
        │   └── videos/
        ├── no_helmet/
        │   ├── images/
        │   └── videos/
        ├── triple_riding/
        │   ├── images/
        │   └── videos/
        ├── traffic_helmet/
        │   ├── images/
        │   └── videos/
        └── multiple_violations/
            ├── images/
            └── videos/
```

## Training Workflow

### 1. Prepare Your Training Data

Place your images and videos in the appropriate folders:
- Images should go in the `images/` subdirectory
- Videos should go in the `videos/` subdirectory

Supported formats:
- Images: `.jpg`, `.jpeg`, `.png`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`

### 2. Extract Frames from Videos (Optional)

If you have videos, you can extract frames for training:

```python
from utils.training_manager import TrainingManager

trainer = TrainingManager()
trainer.extract_frames_from_videos()
```

### 3. Organize Training Data Programmatically

You can also organize data programmatically:

```python
from utils.training_manager import TrainingManager

trainer = TrainingManager()

# Add images for signal jump violations
trainer.organize_training_data('/path/to/signal_jump_images', 'signal_jump', 'images')

# Add videos for helmet violations
trainer.organize_training_data('/path/to/helmet_videos', 'no_helmet', 'videos')
```

### 4. Check Training Data Statistics

```python
stats = trainer.get_training_stats()
for violation_type, data in stats.items():
    print(f"{violation_type}: {data['images']} images, {data['videos']} videos")
```

### 5. Prepare Training/Validation Split

```python
# Split data with 80% for training, 20% for validation
trainer.prepare_training_data(train_ratio=0.8)
```

### 6. Train the Model

```python
# Train with default parameters
results = trainer.train_model(epochs=100, batch_size=16)

# Or with custom parameters
results = trainer.train_model(
    model_path='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch_size=8
)
```

## Using the Training Script

Run the interactive training script:

```bash
# Navigate to the src directory
cd src

# Run the main training workflow
python utils/train_violation_model.py

# See example of how to add data
python utils/train_violation_model.py add_data
```

The interactive script will guide you through:
1. Checking current data statistics
2. Extracting frames from videos
3. Preparing training/validation splits
4. Training the model
5. Viewing statistics

## Training Requirements

### Minimum Data Requirements
- At least 100 images total for basic training
- Recommended: 500+ images for good performance
- Balanced distribution across violation types

### System Requirements
- Python 3.8+
- Sufficient RAM (8GB+ recommended)
- GPU recommended for faster training
- At least 10GB free disk space

## Model Output

After training, the model will be saved in:
```
runs/detect/traffic_violation_model/weights/best.pt
```

You can use this model in your detection system by updating the model path in `core/detector.py`.

## Tips for Better Training Results

1. **Data Quality**: Use high-quality, well-labeled images
2. **Data Diversity**: Include various lighting conditions, angles, and scenarios
3. **Balanced Dataset**: Try to have similar numbers of examples for each violation type
4. **Data Augmentation**: The training process includes automatic augmentation
5. **Validation**: Always validate your model on unseen data
6. **Hyperparameter Tuning**: Experiment with different epochs, batch sizes, and learning rates

## Troubleshooting

### Common Issues

1. **Insufficient Data**: Add more training images
2. **Memory Issues**: Reduce batch size or image size
3. **Poor Performance**: 
   - Check data quality
   - Increase training epochs
   - Adjust learning rate
   - Add more diverse training data

### Getting Help

If you encounter issues:
1. Check the training logs in the console output
2. Verify your data organization follows the structure above
3. Ensure all required dependencies are installed
4. Check that you have sufficient system resources

## Example Usage

```python
# Complete example workflow
from utils.training_manager import TrainingManager

# Initialize
trainer = TrainingManager()

# Add your data (replace with actual paths)
trainer.organize_training_data('/my/signal_jump_photos', 'signal_jump', 'images')
trainer.organize_training_data('/my/helmet_videos', 'no_helmet', 'videos')

# Extract frames from videos
trainer.extract_frames_from_videos()

# Prepare data split
trainer.prepare_training_data(train_ratio=0.8)

# Train model
results = trainer.train_model(epochs=100, batch_size=16)

# Use the trained model
print(f"Model saved to: {results.save_dir}")
```

The trained model will automatically be used by the detection system when placed in the correct location.