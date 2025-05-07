import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as F

from config import IMAGES_PATH, OUTPUT_DIR

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# Get list of image files (excluding readme.txt)
images = [img for img in os.listdir(IMAGES_PATH) if img != "readme.txt"]

# Select a random image (with seed fixed, it will be the same image each time)
image_name = np.random.choice(images)
image_path = os.path.join(IMAGES_PATH, image_name)

original_img = Image.open(image_path).convert('L')  # Convert to grayscale

# Create a figure with subplots for all transformations
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Transformations on fingerprint image: {image_name}', fontsize=20, y=0.98)


shear_angle = 20  
scale_factor = 1.5
combined_scale = 0.8
combined_shear = 15

# Original image
axs[0, 0].imshow(original_img, cmap='gray')
axs[0, 0].set_title('Original', fontsize=16, pad=10)
axs[0, 0].axis('off')

# 1. Shear transformation
sheared_img = F.affine(
    original_img,
    angle=0,  # No rotation
    translate=[0, 0],  # No translation
    scale=1.0,  # No scaling
    shear=[shear_angle, 0]  # Shear in the x-direction
)
axs[0, 1].imshow(sheared_img, cmap='gray')
axs[0, 1].set_title(f'Shear (angle={shear_angle}°)', fontsize=16, pad=10)
axs[0, 1].axis('off')

# 2. Arbitrary scaling
scaled_img = F.affine(
    original_img,
    angle=0,  # No rotation
    translate=[0, 0],  # No translation
    scale=scale_factor,  # Scale by factor
    shear=[0, 0]  # No shear
)
axs[0, 2].imshow(scaled_img, cmap='gray')
axs[0, 2].set_title(f'Scale (factor={scale_factor})', fontsize=16, pad=10)
axs[0, 2].axis('off')

# 3. Horizontal flip
hflip_img = F.hflip(original_img)
axs[1, 0].imshow(hflip_img, cmap='gray')
axs[1, 0].set_title('Horizontal Flip', fontsize=16, pad=10)
axs[1, 0].axis('off')

# 4. Vertical flip
vflip_img = F.vflip(original_img)
axs[1, 1].imshow(vflip_img, cmap='gray')
axs[1, 1].set_title('Vertical Flip', fontsize=16, pad=10)
axs[1, 1].axis('off')

# 5. Combined transformation: shear + scale + flip with fixed parameters
combined_img = F.hflip(  # First apply horizontal flip
    F.affine(
        original_img,
        angle=0,  # No rotation
        translate=[0, 0],  # No translation
        scale=combined_scale,  # Scale down
        shear=[combined_shear, 0]  # Apply shear
    )
)
axs[1, 2].imshow(combined_img, cmap='gray')
axs[1, 2].set_title(f'Combined Transform\nScale: {combined_scale}x | Shear: {combined_shear}° | H-Flip', 
                   fontsize=16, pad=10)
axs[1, 2].axis('off')

# Add colored borders to each subplot for better separation
for i in range(2):
    for j in range(3):
        for spine in axs[i, j].spines.values():
            spine.set_visible(True)
            spine.set_color('darkblue')
            spine.set_linewidth(2)

# Create output directory for saving images if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get filename without extension for saving
base_name = os.path.splitext(image_name)[0]

# Save individual transformed images
original_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_original.png"))
sheared_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_sheared.png"))
scaled_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_scaled.png"))
hflip_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_hflip.png"))
vflip_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_vflip.png"))
combined_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_combined.png"))


plt.figtext(0.5, 0.01, 
           f"Seed: {SEED} | Original image: {image_name}\n"
           f"Transformations: Shear({shear_angle}°), Scale({scale_factor}x), H-Flip, V-Flip, "
           f"Combined(Scale={combined_scale}x, Shear={combined_shear}°, H-Flip)",
           ha="center", fontsize=14, 
           bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})


plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.08)  
plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_all_transforms.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"Applied transformations to image: {image_name}")
print(f"Individual transformations:")
print(f"  - Shear: {shear_angle}°")
print(f"  - Scale: {scale_factor}x")
print(f"  - Horizontal flip")
print(f"  - Vertical flip")
print(f"Combined transformation:")
print(f"  - Scale: {combined_scale}x")
print(f"  - Shear: {combined_shear}°")
print(f"  - Horizontal flip")
print(f"Random seed: {SEED}")
print(f"All images saved to directory: {OUTPUT_DIR}")