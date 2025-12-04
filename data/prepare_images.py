from pathlib import Path
import torch
from pillow import load_image, make_eval_pipeline, to_numpy, iter_images

# 1. Pipeline erzeugen (wie torchvision transforms)
pipeline = make_eval_pipeline(
    size=224,          # Modell-Input
    shorter=256,
    strategy="center"  # oder "letterbox"
)

# 2. Jeder Bildpfad im img-Ordner durchgehen
root = Path("img")
output = Path("processed")
output.mkdir(exist_ok=True)

for image_path in iter_images(root):
    print("Bearbeite:", image_path)

    # 3. Bild laden
    img = load_image(image_path)

    # 4. Transformation anwenden
    img_transformed = pipeline(img)

    # 5. In numpy → Tensor
    arr = to_numpy(img_transformed)
    tensor = torch.tensor(arr)  # shape: (3, 224, 224)

    # 6. Optional: speichern als .pt Datei
    save_path = output / (image_path.stem + ".pt")
    torch.save(tensor, save_path)

print("Fertig!")
