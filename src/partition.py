import random
import shutil
import os

repo_root = os.path.abspath(os.path.join(__file__, "../.."))

def partition(
        source_folder=repo_root + "/data/500_mp3/mp3/",
        train_folder=repo_root + "/data/500_mp3/mp3/train/",
        test_folder=repo_root + "/data/500_mp3/mp3/test/",
        split_ratio=0.8, seed=42):
    
    random.seed(seed)

    file_list = []
    contents = os.listdir(source_folder)
    for el in contents:
        if os.path.isfile(os.path.join(source_folder, el)):
            file_list.append(el)

    random.shuffle(file_list)

    split_index = int(split_ratio * len(file_list))

    train_set = file_list[:split_index]
    test_set = file_list[split_index:]

    for file_name in train_set:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(train_folder, file_name)
        shutil.copy(source_path, dest_path)

    for file_name in test_set:
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(test_folder, file_name)
        shutil.copy(source_path, dest_path)
