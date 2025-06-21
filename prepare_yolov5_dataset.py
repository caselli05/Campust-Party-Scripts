import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import random_split
from tqdm import tqdm # For progress bar

def prepare_yolov5_dataset(root_dataset_path, output_base_path='yolov5_dataset', train_ratio=0.8):
    print(f"Starting dataset preparation from '{root_dataset_path}'...")

    # 1. Define a basic transform (we just need to load images, no complex augmentation yet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print(f"Current working directory: {os.getcwd()}")
    full_path_to_check = os.path.join(os.getcwd(), root_dataset_path)
    print(f"Attempting to find dataset at: {full_path_to_check}")
    if not os.path.exists(full_path_to_check):
        print(f"ERROR: Path does not exist: {full_path_to_check}")
        print("Please ensure your 'dataset' folder is in the same directory as this script,")
        print("OR provide the full absolute path to your dataset folder in original_dataset_root.")
        import sys
        sys.exit(1) # Exit the script if the path isn't found
    # 2. Load the original dataset
    full_dataset = datasets.ImageFolder(root=root_dataset_path, transform=transform)
    
    # Get class names and map them to integer IDs
    class_names = full_dataset.classes
    # FIX: Define num_classes here
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")


    # 3. Split the dataset into training and validation sets
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Splitting dataset: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    # 4. Create the YOLOv5 directory structure
    yolo_images_train_dir = os.path.join(output_base_path, 'images', 'train')
    yolo_images_val_dir = os.path.join(output_base_path, 'images', 'val')
    yolo_labels_train_dir = os.path.join(output_base_path, 'labels', 'train')
    yolo_labels_val_dir = os.path.join(output_base_path, 'labels', 'val')

    # Create directories, clean up if they exist
    for directory in [yolo_images_train_dir, yolo_images_val_dir,
                       yolo_labels_train_dir, yolo_labels_val_dir]:
        if os.path.exists(directory):
            print(f"Removing existing directory: {directory}")
            shutil.rmtree(directory) # Remove existing content
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # 5. Populate the new directories
    print("Copying images and creating dummy label files...")

    # Helper function to process a split
    def process_split(dataset_split, image_dest_dir, label_dest_dir, data_type):
        for i, (img_tensor, class_idx) in tqdm(enumerate(dataset_split), total=len(dataset_split), desc=f"Processing {data_type} split"):
            # Get the original image path from the dataset
            original_idx = dataset_split.indices[i]
            original_img_path, _ = full_dataset.imgs[original_idx]

            # Construct new image filename (e.g., original_name_00001.jpg)
            img_original_filename = os.path.basename(original_img_path)
            img_filename = f"{os.path.splitext(img_original_filename)[0]}_{i:05d}.jpg"
            img_dest_path = os.path.join(image_dest_dir, img_filename)

            # Copy the image file
            shutil.copyfile(original_img_path, img_dest_path)

            # Create dummy label file
            label_filename = f"{os.path.splitext(img_filename)[0]}.txt" # Label name matches image name
            label_dest_path = os.path.join(label_dest_dir, label_filename)

            # YOLO format: class_id x_center y_center width height (all normalized 0-1)
            # Dummy label: class_idx 0.5 0.5 1.0 1.0 (single object of specific class, filling whole image)
            dummy_label_content = f"{class_idx} 0.5 0.5 1.0 1.0\n"
            with open(label_dest_path, 'w') as f:
                f.write(dummy_label_content)

    process_split(train_dataset, yolo_images_train_dir, yolo_labels_train_dir, "train")
    process_split(val_dataset, yolo_images_val_dir, yolo_labels_val_dir, "val")

# 6. Create a data.yaml file for YOLOv5 training
    # CHANGE IS HERE: path should be the ABSOLUTE path to your dataset root
    absolute_dataset_root = os.path.abspath(output_base_path)
    # Ensure forward slashes for YAML portability, especially on Windows
    absolute_dataset_root = absolute_dataset_root.replace(os.sep, '/')

    data_yaml_content = f"""
# YOLOv5 dataset for {class_names}
path: {absolute_dataset_root}  # Explicitly set absolute path
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# number of classes
nc: {num_classes}

# class names
names: {class_names}
"""
    data_yaml_path = os.path.join(output_base_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    print(f"\nGenerated data.yaml at: {data_yaml_path}")
    print("When running YOLOv5's train.py from *any* directory, use:")
    print(f"python C:\\Users\\pc-de-caselli\\Desktop\\Campust-Party-Scripts\\yolov5\\train.py --data {os.path.relpath(data_yaml_path, os.getcwd())} ...") # This will still print a relative path, but the important part is data.yaml's content
    print("\nEnsure your command uses the ABSOLUTE path to train.py if you are not in its directory.")
    print(f"The --data argument should be: --data {os.path.abspath(data_yaml_path)}")

    print("\nDataset preparation complete!")
    print(f"Your YOLOv5 ready dataset is in: {os.path.abspath(output_base_path)}")
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! IMPORTANT: These labels are DUMMY. For real object detection, !!!")
    print("!!!            you MUST annotate bounding boxes using a tool     !!!")
    print("!!!            like LabelImg, Roboflow, or CVAT!                 !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\nExample dummy label content (class_id x_center y_center width height):\n '0 0.5 0.5 1.0 1.0'") # Using '0' as an example class_id
    print("This implies one object of its class fills the entire image, which is rarely true for detection tasks.")


if __name__ == "__main__":
    # Define your original dataset root
    original_dataset_root = 'dataset' # This should be the folder containing 'area_degradada', 'construcao_irregular', etc.

    # Define where the new YOLOv5 formatted dataset will be created
    yolov5_output_dir = 'yolov5_ready_dataset' # This folder will be created in the same directory as this script

    prepare_yolov5_dataset(original_dataset_root, yolov5_output_dir)