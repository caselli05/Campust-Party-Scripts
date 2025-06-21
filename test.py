import torch
import os
from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import matplotlib.pyplot as plt

def run_inference_on_image(model_path, image_path, local_yolov5_repo_path, output_inference_base_dir):
    """
    Loads a trained YOLOv5 model and performs inference on a single image,
    then displays the results and saves them to a specified output directory.

    Args:
        model_path (str): Path to your trained YOLOv5 weights (e.g., 'runs/train/yolov5s_custom/weights/best.pt').
        image_path (str): Path to the image file you want to test.
        local_yolov5_repo_path (str): ABSOLUTE path to your locally cloned YOLOv5 repository.
        output_inference_base_dir (str): Base directory where inference results will be saved.
                                        YOLOv5 will create subfolders like 'exp', 'exp1' etc. within this.
    """
    print(f"Loading YOLOv5 model from: {model_path}")
    print(f"Using local YOLOv5 repo at: {local_yolov5_repo_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = torch.hub.load(local_yolov5_repo_path, "custom", path=model_path, source='local', force_reload=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'local_yolov5_repo_path' points directly to your cloned YOLOv5 repository.")
        print("Also, confirm that the 'model_path' is correct and accessible.")
        return
    
    print(f'My model is {model}')

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    print(f"Running inference on image: {image_path}")

    # Perform inference
    results = model(image_path)

    # --- CHANGE: Use the specified output_inference_base_dir ---
    # The .save() method saves to a directory like 'output_inference_base_dir/expX/'
    results.save(save_dir=output_inference_base_dir)

    # Find the actual saved image path within the new 'exp' folder
    latest_exp_dir = sorted([d for d in os.listdir(output_inference_base_dir) if d.startswith('exp')],
                            key=lambda x: os.path.getmtime(os.path.join(output_inference_base_dir, x)),
                            reverse=True)

    if latest_exp_dir:
        actual_saved_img_path = os.path.join(output_inference_base_dir, latest_exp_dir[0], os.path.basename(image_path))
        print(f"Annotated image saved to: {actual_saved_img_path}")

        try:
            img_display = Image.open(actual_saved_img_path)
            plt.imshow(img_display)
            plt.axis('off')
            plt.title("Detected Objects")
            plt.show()
        except Exception as e:
            print("Could not display image using matplotlib. Ensure you have a GUI environment.")
            print(f"Error: {e}")
            print("Results are still saved to:", actual_saved_img_path)
    else:
        print("Could not find the saved results image directory.")


if __name__ == "__main__":
    # --- Configuration for your PC ---

    # 1. Path to your best.pt model weights
    BEST_PT_PATH = 'C:/Users/pc-de-caselli/Desktop/Campust-Party-Scripts/yolov5/runs/train/yolov5s_custom2/weights/best.pt' # <--- CONFIRM THIS PATH

    # --- NEW: Folder where you will put your photos for testing ---
    # Create this folder manually and put your phone photos inside it.
    SOURCE_PHOTOS_DIR = 'C:/Users/pc-de-caselli/Desktop/imagens' # <--- DEFINE THIS PATH

    # --- NEW: Folder where the annotated (inferred) images will be saved ---
    # This folder will be created by the script if it doesn't exist.
    OUTPUT_INFERENCE_DIR = 'C:/Users/pc-de-caselli/Desktop/Campust-Party-Scripts/inference_output' # <--- DEFINE THIS PATH

    # 3. Path to your local YOLOv5 repository clone
    LOCAL_YOLOV5_REPO_PATH = 'C:/Users/pc-de-caselli/Desktop/Campust-Party-Scripts/yolov5' # <--- CONFIRM THIS PATH

    # --- Example of how to get a specific photo from SOURCE_PHOTOS_DIR ---
    # You can list all files and loop through them, or pick one.
    # For now, let's just pick a specific image name.
    # Make sure 'my_phone_pic.jpg' exists in your SOURCE_PHOTOS_DIR
    TEST_IMAGE_FILENAME = 'imagem_arvore.jpg' # <--- CHANGE THIS to your photo's filename
    TEST_IMAGE_PATH = os.path.join(SOURCE_PHOTOS_DIR, TEST_IMAGE_FILENAME)


    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_INFERENCE_DIR, exist_ok=True)
    print(f"Ensuring output directory exists: {OUTPUT_INFERENCE_DIR}")


    run_inference_on_image(BEST_PT_PATH, TEST_IMAGE_PATH, LOCAL_YOLOV5_REPO_PATH, OUTPUT_INFERENCE_DIR)

    print("\nInference script finished. Check the displayed image window and the specified output folder for annotated images.")
    print(f"Output images are saved in: {os.path.abspath(OUTPUT_INFERENCE_DIR)}")