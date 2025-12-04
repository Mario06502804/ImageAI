from pathlib import Path
from PIL import Image
import predict as pred

def loadSingleImage():
    root = Path("img_to_Torch")
    imgDir = Path("img")
    print("Available images:")
    for image_path in root.iterdir():
        if image_path.suffix == ".pt":
            print(image_path.name)

    selected_image = input("Enter the filename of the image to load (with .pt extension): ")

    selected_image = selected_image.replace(".pt", ".jpg")

    # Name ohne Extension (z.B. Love01)
    base_name = selected_image.replace(".jpg", "")

    # Automatisch richtigen Ordner finden
    found_path = None
    for folder in imgDir.iterdir():
        if folder.is_dir():
            candidate = folder / selected_image
            if candidate.exists():
                found_path = candidate
                break

    if not found_path:
        print(f"ERROR: Could not find {selected_image} in any folder inside {imgDir}")
        return

    img = Image.open(found_path)
    img.show()

    pred.main(found_path) 
    # execute prediction.py from here with the selected image as param.


