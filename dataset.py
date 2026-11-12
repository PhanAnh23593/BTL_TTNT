import os
import shutil
import random

input_dir = "Dataset2"
output_dir = "dataset_split_2"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

