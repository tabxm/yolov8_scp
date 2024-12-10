# you can create your own using the following steps:

# Collect Images

Use a camera or smartphone to capture images of each PSL gesture.
Ensure varied backgrounds, lighting conditions, and angles to improve robustness.
# Organize images into folders for each gesture:
```
raw_data/
├── gesture_1/
├── gesture_2/
├── gesture_3/
```
# Annotate Images

Use a tool like LabelImg to annotate the hand gestures in each image.
Save the annotations in YOLO format (.txt files). Each label file should have the same name as its corresponding image and be placed in a labels/ folder.
Prepare Folder Structure
# Organize the dataset into the required format:

```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
├── labels/
│   ├── train/
│   ├── val/
│   ├── test/
```
# Split the Dataset
Divide the dataset into training, validation, and testing sets. For example:

Training set: 70% of the images
Validation set: 20% of the images
Testing set: 10% of the images
# You can automate this process using Python:

```python

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(image_dir, label_dir, output_dir, train_size=0.7, val_size=0.2):
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))
    
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=(1 - train_size), random_state=42)
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=(1 - val_size / (1 - train_size)), random_state=42)

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', folder), exist_ok=True)

    for dataset, folder in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
        for img in dataset:
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, 'images', folder))
            shutil.copy(os.path.join(label_dir, img.replace('.jpg', '.txt')), os.path.join(output_dir, 'labels', folder))
```
# Example usage:
split_dataset('raw_data/images', 'raw_data/labels', 'dataset', train_size=0.7, val_size=0.2)

# Dataset Configuration
Create a data.yaml file to configure the dataset for YOLOv8:

```yaml

train: dataset/images/train
val: dataset/images/val
nc: 10                       # Number of gesture classes
names: ['gesture_1', 'gesture_2', ..., 'gesture_10']
```
