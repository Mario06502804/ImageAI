import data.torchvision as vision
import data.prepare_images as prep
import models.plant_model as plantModel
import data.loadSingleImage as loadImg

def main():
    print("-----MAIN MENU-----")
    print("--------CLI--------")
    print("1. Load Image to Analyze by AI")
    print("2. Train Plantmodel")
    print("3. Validate Plantmodel")
    choice = input()    
    if choice == "1":
        loadImg.loadSingleImage()
    elif choice == "2":
        print("Not implemented yet") # Train model
    elif choice == "3":
        print("Not implemented yet") # Validate model
main()
