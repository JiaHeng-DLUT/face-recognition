YouTube Faces DB
================

Created by: Lior Wolf, Tal Hassner, Itay Maoz
==========

Contents:
========

1. WolfHassnerMaoz_CVPR11.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper explaining this work, how the database was created, the benchmarks, MBGS, etc.
Read it to get more details about the following.


2. frame_images_DB
~~~~~~~~~~~~~~~~~~

Contains the videos downloaded from youtube, broken to frames.
The directory structure is:
subject_name\video_number\video_number.frame.jpg

For each person in the database there is a file called <subject_name>.labeled_faces.txt
The data in this file is in the following format:
filename,[ignore],x,y,width,height,[ignore],[ignore]
where:
x,y are the center of the face and the width and height are of the rectangle that the face is in.
For example:
$ head -3 Richard_Gere.labeled_faces.txt
Richard_Gere\3\3.618.jpg,0,262,165,132,132,0.0,1
Richard_Gere\3\3.619.jpg,0,260,164,131,131,0.0,1
Richard_Gere\3\3.620.jpg,0,261,165,129,129,0.0,1


3. aligned_images_DB
~~~~~~~~~~~~~~~~~~~

Similar to frame_images_DB, contains the videos downloaded from youtube broken to frames, but after some manipulation:
a. face detection, expanding the bounding box by 2.2 and cropping from the frame.
b. alignment.
The directory structure is:
subject_name\video_number\aligned_detect_video_number.frame.jpg


4. descriptors_DB
~~~~~~~~~~~~~~~~~

Contains mat files with the descriptors of the frames.
The directory structure is:
subject_name\mat files

For each video there are two files:
aligned_video_1.mat
video_1.mat

The files contain descriptors per frame, several descriptors type per frame.
One contains the aligned version of the faces in the frame and the other contains the not aligned version.
Each of the above file has a struct with the following:

VID_DESCS_FPLBP: [560x80 double]
VID_DESCS_LBP: [1770x80 double]
VID_DESCS_CSLBP: [480x80 double]
VID_DESCS_FILENAMES: {1x80 cell}

These are the different descriptors and the file names.


5. meta_data
~~~~~~~~~~~~

Contains the meta_and_splits.mat file, which provides an easy way for accessing the mat files in the descriptors DB.
The Splits is a data structure dividing the data set to 10 independent splits.
Each triplet in the Splits is in the format of (index1, index2, is_same_person), where index1 and index2 are the indices in the mat_names structure.
All together 5000 pairs divided equaly to 10 independent splits, with 2500 same pairs and 2500 not-same pairs.

    video_labels: [1x3425 double]
     video_names: {3425x1 cell}
       mat_names: {3425x1 cell}
          Splits: [500x3x10 double]


6. headpose_DB
~~~~~~~~~~~~~~

Contains mat files with the three rotation angles of the head for each frame in the data set.
The directory structure is:
headorient_apirun_subject_name_video_number.mat

Each mat file contains a struct with the following:
headpose: [3x60 double]

