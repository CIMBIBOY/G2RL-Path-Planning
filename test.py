
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_data_file(file_path):
    image = Image.open(file_path)
    return np.array(image)

data_file = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
data = read_data_file(data_file)

# Modify the data
data[0, 0, 0] = 255
data[0, 0, 1] = 165
data[0, 0, 2] = 0


# Display the modified image
plt.imshow(data)
plt.show()
