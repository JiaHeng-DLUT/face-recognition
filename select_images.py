import os
from shutil import copy

if __name__ == '__main__':
    for (path, dir_list, file_list) in os.walk('./YouTubeFaces/frame_images_DB_224x224'):
        for file_name in file_list:
            if file_name.endswith('.csv'):
                with open(os.path.join(path, file_name), "r") as file:
                    first_line = file.readline()
                    for last_line in file:
                        pass
                print(first_line)
                image_path = first_line.split(',')[0]
                new_path = path.replace("frame_images_DB_224x224", "frame_images_DB_224x224_highest")
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                copy(os.path.abspath(image_path), os.path.abspath(new_path))
                print(last_line)
                image_path = last_line.split(',')[0]
                new_path = path.replace("frame_images_DB_224x224", "frame_images_DB_224x224_lowest")
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                copy(image_path, new_path)