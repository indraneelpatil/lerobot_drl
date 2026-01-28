# !/usr/bin/env python
"""
Docstring for lerobot_drl.neel.scripts.dataset_view_image
"""
from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("indraneelpatil/isaac_sim_pick_lift_sim4", split="train")

sample = ds[0]

print(sample)

front = sample["observation.images.front"]
wrist = sample["observation.images.wrist"]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(front)
plt.title("Front camera")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(wrist)
plt.title("Wrist camera")
plt.axis("off")

plt.show()