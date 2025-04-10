# aruco-pose-estimation
This repository contains all the code you need to generate an ArucoTag, detect ArucoTags in images and videos, and then use the detected tags to estimate the pose of the object.


This software is partially base on [this repository](https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python).


```commandline
usage: pi_pose_estimation.py [-h] -i CAMERA [-s MARKER_SIZE] [-k K_MATRIX] [-d D_COEFF] [-m MARKER] [-c DICT] [-t DURATION] [-n SAMPLE] [-w WIDTH] [-y HEIGHT] [-v] [-r RES] [-g] [-o] [-sm] [-b] [-mr] [-e NOTE] [-l LENSPOS]

options:
  -h, --help            show this help message and exit
  -i CAMERA, --camera CAMERA
                        One of arducam, aideck, pi3, pi3w, or pihq6mm
  -s MARKER_SIZE, --marker_size MARKER_SIZE
                        Dimension of marker (meter)
  -k K_MATRIX, --K_Matrix K_MATRIX
                        Path to calibration matrix (numpy file)
  -d D_COEFF, --D_Coeff D_COEFF
                        Path to distortion coefficients (numpy file)
  -m MARKER, --marker MARKER
                        Type of tag to detect. One of ARUCO, APRILTAG, STAG, or P4
  -c DICT, --dict DICT  Type of dictionary of tag to detect
  -t DURATION, --duration DURATION
                        Duration of sampling (second)
  -n SAMPLE, --sample SAMPLE
                        Number of samples per second
  -w WIDTH, --width WIDTH
                        Width of image
  -y HEIGHT, --height HEIGHT
                        Height of image
  -v, --live            Show live camera image
  -r RES, --res RES     Image resolution, one of 244p, 480p, 720p, 1080p, or 1440p, overwrites width and height
  -g, --debug           Print logs
  -o, --save            Save data
  -sm, --sensor_modes   Print sensor modes of the camera and terminate
  -b, --broadcast       Broadcast measurements modify worker socket with proper ip and port
  -mr, --max_fps        Use maximum fps
  -e NOTE, --note NOTE  Notes
  -l LENSPOS, --lenspos LENSPOS
                        Lens position for manual focus
```