# -*- coding: utf-8 -*-

import cv2
import os

(width, height) = (224, 224)

if __name__ == '__main__':
    for (path, dir_list, file_list) in os.walk(r'.\YouTubeFaces\frame_images_DB'):
        # print(path)
        resized_dir = path.replace('frame_images_DB', 'frame_images_DB_' + str(width) + 'x' + str(height))
        if not os.path.exists(resized_dir):
            os.mkdir(resized_dir)
        for file_name in file_list:
            if file_name.endswith('.jpg'):
                image_path = os.path.join(path, file_name)
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, (width, height))
                resized_image_path = os.path.join(resized_dir, file_name)
                cv2.imwrite(resized_image_path, resized_image)
