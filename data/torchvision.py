import torch
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'  # Ensure tight bounding boxes for saved figures

from torchvision.transforms import v2
from torchvision.io import decode_image