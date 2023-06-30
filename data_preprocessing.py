import tensorflow as tf
import cv2
import os
import imghdr
from tensorflow.keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt

def plot_images(images, labels):
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(images[:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(labels[idx])
    plt.show()

def visualize_data(data_dir):
    data = image_dataset_from_directory(data_dir)
    data_iterator = data.as_numpy_iterator()
    images, labels = data_iterator.next()
    plot_images(images, labels)
    
def preprocess_data(data_dir):
    image_exts = ['jpeg','jpg', 'bmp', 'png']

    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))

    data = image_dataset_from_directory(data_dir)
    data = data.map(lambda x,y: (x/255, y))
    
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    
    return train, val, test
