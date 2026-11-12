import os
import shutil
import random

input_dir = "Dataset2"
output_dir = "dataset_split_2"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

splits = ['train', 'val', 'test']

for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

classes = [
    c for c in os.listdir(input_dir)
    if os.path.isdir(os.path.join(input_dir, c))
]

print("Classes found:", classes)

