import os
import shutil
import random


def create_grouped_structure(base_path, new_base_path, group_size):
    # Create new directory structure
    for split in ['train', 'valid', 'test']:
        for subfolder in ['images', 'labels']:
            os.makedirs(os.path.join(new_base_path, split, subfolder), exist_ok=True)


def group_and_rename_files(base_path, new_base_path, group_size):
    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')

        new_image_dir = os.path.join(new_base_path, split, 'images')
        new_label_dir = os.path.join(new_base_path, split, 'labels')

        images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        # Ensure the images and labels match
        assert len(images) == len(labels)

        # Shuffle the list to ensure random grouping
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)

        num_groups = len(images) // group_size
        rest_count = len(images) % group_size

        # Process each group
        for i in range(num_groups):
            group_name = f'group_{i + 1}'
            os.makedirs(os.path.join(new_image_dir, group_name), exist_ok=True)
            os.makedirs(os.path.join(new_label_dir, group_name), exist_ok=True)

            for j in range(group_size):
                img_index = i * group_size + j
                new_img_name = f'image{j + 1:02d}.jpg'
                new_label_name = f'image{j + 1:02d}.txt'

                shutil.copy(os.path.join(image_dir, images[img_index]),
                            os.path.join(new_image_dir, group_name, new_img_name))
                shutil.copy(os.path.join(label_dir, labels[img_index]),
                            os.path.join(new_label_dir, group_name, new_label_name))

        # Handle remaining files
        if rest_count > 0:
            rest_image_dir = os.path.join(new_image_dir, 'rest_images')
            rest_label_dir = os.path.join(new_label_dir, 'rest_labels')
            os.makedirs(rest_image_dir, exist_ok=True)
            os.makedirs(rest_label_dir, exist_ok=True)

            for i in range(rest_count):
                img_index = num_groups * group_size + i
                new_img_name = f'image{i + 1:02d}.jpg'
                new_label_name = f'image{i + 1:02d}.txt'

                shutil.copy(os.path.join(image_dir, images[img_index]), os.path.join(rest_image_dir, new_img_name))
                shutil.copy(os.path.join(label_dir, labels[img_index]), os.path.join(rest_label_dir, new_label_name))


if __name__ == "__main__":
    base_path = "../image_data/geok"
    new_base_path = "../image_data/geok_grouped"
    group_size = 10

    create_grouped_structure(base_path, new_base_path, group_size)
    group_and_rename_files(base_path, new_base_path, group_size)
