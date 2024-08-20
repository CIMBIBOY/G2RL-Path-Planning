import cv2
import os
import re

def create_video_from_images(image_folder, train_number, video_name='train_x_video.avi', fps=5, codec='MJPG', scale=50):
    # Get list of all image filenames in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and f'{train_number}_' in img]
    
    # Sort images by the second part of their filename
    images.sort(key=lambda x: int(re.search(r'\w+_\w+_\d+_\d+_(\d+).png', x).group(1)))
    
    if not images:
        print(f"No images found for train number {train_number}.")
        return

    # Get the dimensions of the images
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # Rescale dimensions
    height, width = height * scale, width * scale
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is not None:
            # Resize the frame with improved interpolation
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            video.write(frame)
        else:
            print(f"Warning: Could not read image {image_path}. Skipping.")
    
    video.release()
    cv2.destroyAllWindows()

# Generate video
train_name = 'czm1_mppo_42_114811'

create_video_from_images(f'eval/training_images/{train_name}', train_name, f'{train_name}.avi', fps=7, codec='MJPG', scale=10)