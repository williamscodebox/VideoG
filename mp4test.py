import imageio
import numpy as np

frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(8)]
imageio.mimwrite("test.mp4", frames, fps=8, quality=8)
