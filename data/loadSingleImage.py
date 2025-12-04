from pathlib import Path
from PIL import Image

def loadSingleImage():
    root = Path("img_to_Torch")
    imgDir = Path("img")
    print("Available images:")
    for image_path in root.iterdir():
        if image_path.suffix == ".pt":
            print(image_path.name)

    selected_image = input("Enter the filename of the image to load (with .pt extension): ")

    selected_image = selected_image.replace(".pt", ".jpg") 
    # Gotta strip the numbers from the filename to open the Dir
    selectedImageStripped = selected_image
    jpgDir = selectedImageStripped.split("0")[0]
    imgDir = imgDir/jpgDir/selected_image # Apparently have to add .jpg here?? probably cuz we strip it above?
    # jfc this is stupid code. FIX ME LATER PLS

    img = Image.open(imgDir) 
    img.show()

    #continue with loading the .pt file to send it to the model afterwards
