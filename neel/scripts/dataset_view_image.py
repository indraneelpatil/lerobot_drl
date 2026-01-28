# !/usr/bin/env python
"""
Docstring for lerobot_drl.neel.scripts.dataset_view_image
"""
from datasets import load_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

#ds = LeRobotDataset(repo_id="indraneelpatil/isaac_sim_pick_lift_sim4")
ds = LeRobotDataset("/home/neel/.cache/huggingface/lerobot/indraneelpatil/isaac_sim_pick_lift_sim5")

sample = ds[120]

print(sample)

front = sample["observation.images.front"]
wrist = sample["observation.images.wrist"]

front = front.permute(1, 2, 0).cpu().numpy()
wrist = wrist.permute(1, 2, 0).cpu().numpy()

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