import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def csv_reader(path):
    headers = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    return pd.read_csv(path, names=headers, skiprows=1)
def prepare_data():
    train = csv_reader('data/driving_log.csv')
    train_nonzero = train[train.steering != 0]
    train_zero = train[train.steering == 0]
    valid = pd.concat([train_nonzero, train_zero.sample(frac=0.1)], ignore_index=True)
    return train, valid

def crop(image, top, bottom):
    return image[top:bottom,:]

def resize(image, dim):
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def preprocess_image(image, top=60, bottom=140, dim = (64,64)):
    return resize(crop(image, top, bottom), dim)

def camera_data(row, camera, angle_correction = 0.23):
    """

     :param row: each row of the csv represented as pandas data frame (iloc function)
     :param camera: location for each of the three positions of camera: left, center and right.
     :param angle: angle correction.
     :return: the image read by mpimg and its corresponding corrected steering angle.
     """
    if camera == 0:
        image_path = row.left.strip()
        steering = row.steering + angle_correction
    elif camera == 1:
        image_path = row.center.strip()
        steering = row.steering
    else:
        image_path = row.right.strip()
        steering = row.steering - angle_correction
    image = mpimg.imread('./data/'+image_path[image_path.find('IMG'):])

    return image, steering

#function to randomly choose a camera image and its corresponding steering angle
def camera_image_choose(row):
    """
    :param row: row (pandas dataframe)
    :return: randomly selected image and steering angle
    """
    camera = np.random.randint(0,3)
    image, steering = camera_data(row, camera)
    return image, steering

def flip_random(image, steering):
    # randomly choose
    #if np.random.binomial(1, 0.5):
    return cv2.flip(image, 1), -steering
    #else:
    #    return image, steering

def gamma_corrector(image):
    """
    description:
    Gamma correction and the Power Law Transform

Gamma correction is also known as the Power Law Transform. First, our image pixel intensities must be scaled from the range [0, 255] to [0, 1.0]. From there, we obtain our output gamma corrected image by applying the following equation:

O = I ^ (1 / G)

Where I is our input image and G is our gamma value. The output image O is then scaled back to the range [0, 255].

Gamma values < 1 will shift the image towards the darker end of the spectrum while gamma values > 1 will make the image appear lighter. A gamma value of G=1 will have no affect on the input image:
    """

    """
    :link: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    :param image:
    :return:
    """
    gamma = np.random.uniform(0.4, 1.5)
    gamma_inv = 1.0/ gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def random_shear(image, steering, shear_range=200):
  rows, cols, _ = image.shape
  dx = np.random.randint(-shear_range, shear_range + 1)
  random_point = [cols / 2 + dx, rows / 2]
  pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
  pts2 = np.float32([[0, rows], [cols, rows], random_point])
  dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
  M = cv2.getAffineTransform(pts1, pts2)
  image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
  return image, steering + dsteering

def random_bumpy(image, y_range=20):
  rows, cols, _ = image.shape
  dy = (y_range * np.random.uniform()) - (y_range / 2)
  M = np.float32([[1, 0, 0], [0, 1, dy]])
  return cv2.warpAffine(image, M, (cols, rows))

def get_augmented_data(image, steering):
  if np.random.binomial(1, 0.9):
    image, steering = random_shear(image, steering)
  image, steering = flip_random(image, steering)
  image = gamma_corrector(image)
  image = preprocess_image(image)

  return image, steering


def next_train_batch(data, batch_size):
  total = len(data)
  while True:
    images = []
    steerings = []
    random_indices = np.random.randint(0, total, batch_size)
    for idx in random_indices:
      row = data.iloc[idx]
      image, steering = camera_image_choose(row)
      image, steering = get_augmented_data(image, steering)
      images.append(image)
      steerings.append(steering)

    yield np.array(images), np.array(steerings)

def next_valid_batch(data, batch_size):
  total = len(data)
  current = 0
  while True:
    images = []
    steerings = []
    for i in range(batch_size):
      row = data.iloc[current]
      image, steering = camera_data(row, 1)
      images.append(preprocess_image(image))
      steerings.append(steering)
      current = (current + 1) % total

    yield np.array(images), np.array(steerings)
def test_image_shear(data):
    r = np.random.randint(1,100)
    original_image, steering  = camera_image_choose(data.iloc[r])
    image , steering = random_shear(original_image, steering,shear_range=100)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img = original_image

    imgplot = plt.imshow(img)
    a.set_title('original image')

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(image)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('sheared image')
    plt.show()

def test_image_gamma(data):
    r = np.random.randint(1,100)
    original_image, steering_angle = camera_image_choose(data.iloc[r])
    gamma_corrected = gamma_corrector(original_image)
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img = original_image

    imgplot = plt.imshow(img)
    a.set_title('original image')

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(gamma_corrected)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('gamma corrected image')
    plt.show()

def test_image_flip(data):
    r = np.random.randint(1, 100)
    original_image, steering_angle = camera_image_choose(data.iloc[r])
    flipped_image, flipped_steering = flip_random(original_image,steering_angle)
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img = original_image

    imgplot = plt.imshow(img)
    a.set_title('original image')

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(flipped_image)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('flipped image')
    plt.show()


def test_augmented_image(data):
    r = np.random.randint(1, 100)
    original_image, steering_angle = camera_image_choose(data.iloc[r])
    augmented_image, augmented_steering = get_augmented_data(original_image,steering_angle)
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img = original_image

    imgplot = plt.imshow(img)
    a.set_title('original image')

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(augmented_image)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('augmented image')
    plt.show()

def test_left_center_right(data):
    r = np.random.randint(1,1500)
    row = data.iloc[r]
    left,center, right = 0,1,2
    left_image,l_angle = camera_data(row,left)
    center_image,c_angle = camera_data(row, center)
    right_image, r_angle = camera_data(row, right)

    #plotting the images
    fig = plt.figure()
    a = fig.add_subplot(1,3,1)
    img_plot = plt.imshow(left_image)
    a.set_title('left camera')

    a = fig.add_subplot(1,3,2)
    img_plot = plt.imshow(center_image)
    a.set_title('center camera')

    a = fig.add_subplot(1, 3, 3)
    img_plot = plt.imshow(right_image)
    a.set_title('right camera')
    plt.show()
