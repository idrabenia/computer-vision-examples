import cv2
import os


def split_image(path, dest_dir):
    img = cv2.imread(path)

    x_max, y_max = img.shape[1], img.shape[0]

    items = []
    for x in range(0, x_max, 300):
        for y in range(0, y_max, 300):
            items.append(img[y:y + 300, x:x + 300])

    global j
    for i in items:
        cv2.imwrite(dest_dir + '/img_%s.jpg' % j, i)
        j += 1


def segment_images(images_dir, dest_dir):
    for img in os.listdir(images_dir):
        if img.endswith('.JPG'):
            split_image(images_dir + '/' + img, dest_dir)

j = 0
# segment_images('./../dataset_minsk_freezer/open', './items')
split_image('./IMG_1051.JPG', './items')