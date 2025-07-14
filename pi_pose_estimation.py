'''
Sample Usage:
source env/bin/activate
python pi_pose_estimation.py -i pi3 -r 720p -t 10 --marker_size 0.02 --live -e 200mm.
Indicate the actual distance you are measuring by passing -e input. Format it as [distance in millimeters]mm.
Pass --save flag to save results in the results directory.
To process, make a directory inside the data directory and copy the experiment results you want to process to it.
Then, run python process.py -i [dir name] to generate summary tables.

References:
[1] Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition, 47(6), 2280-2292.
[2] Benligiray, B., Topal, C., & Akinlar, C. (2019). STag: A stable fiducial marker system. Image and Vision Computing, 89, 158-169.
[3] Olson, E. (2011, May). AprilTag: A robust and flexible visual fiducial system. In 2011 IEEE international conference on robotics and automation (pp. 3400-3407). IEEE.
[4] Alimohammadzadeh, H., & Ghandeharizadeh, S. (2024, October). Swarical: An Integrated Hierarchical Approach to Localizing Flying Light Specks. In Proceedings of the 32nd ACM International Conference on Multimedia (pp. 6153-6161).
'''

import socket
import struct
import time
import os
import argparse
import logging
import json
from datetime import datetime

import numpy as np
import cv2
import scipy.spatial.transform as transform

from utils import ARUCO_DICT
from consts import res_map, camera_map, Marker
from worker_socket import WorkerSocket

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def yaw_pitch_roll_decomposition(R):
    """
    :param R: Rotation matrix
    :return: Euler angles (yam, pitch, roll)
    """
    rotation = transform.Rotation.from_matrix(R)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    return euler_angles


def pose_estimation(frame, matrix_coefficients, distortion_coefficients, marker_type, dict_type):
    """
    :param frame: Frame from the video stream
    :param matrix_coefficients: Intrinsic matrix of the calibrated camera
    :param distortion_coefficients: Distortion coefficients associated with your camera
    :param marker_type: Marker type, ARUCO, STAG or APRILTAG
    :param dict_type: Dictionary type of aruco marker
    :return frame: The frame with the axis drawn on it
    :return ids: Ids of the marker
    :return pos: Positions of the markers
    :return ori: Orientations of the markers
    """
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Three fiducial marker type are supported by this program: ARUCO [1], STAG[2], and APRILTAG[3].
    # Use the --marker argument to switch between different types, the default is ARUCO which is used in the Swarical paper.
    pos = []
    ori = []
    if marker_type is Marker.ARUCO:
        aruco_dict_type = ARUCO_DICT[dict_type]
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected_corners = detector.detectMarkers(gray)

    elif marker_type is Marker.STAG:
        corners, ids, rejected_corners = stag.detectMarkers(frame, int(dict_type))

    elif marker_type is Marker.APRILTAG:
        april_options = apriltag.DetectorOptions(families=dict_type)
        april_detector = apriltag.Detector(april_options)
        results = april_detector.detect(gray)
        corners = []
        ids = []
        for r in results:
            corners.append(np.array([r.corners], dtype='float32'))
            ids.append([r.tag_id])
        ids = np.array(ids)

    # After the 4 corners of a marker are detected, correspondence of 2D points in the image (corners)
    # and actual 3D marker points (marker_points) are used to solve PnP problem.
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Refine position of the corners in the image to the subpixel accuracy.
            refined_corners = cv2.cornerSubPix(gray, corners[i], (11, 11), (-1, -1), criteria)

            # Solve PnP using point correspondence and camera intrinsics
            nada, rvec, tvec = cv2.solvePnP(marker_points, refined_corners, matrix_coefficients,
                                            distortion_coefficients, False, cv2.SOLVEPNP_IPPE_SQUARE)

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            pos.append(tvec)
            ori.append(yaw_pitch_roll_decomposition(rmat))

            if args["live"]:
                gray = cv2.drawFrameAxes(gray, matrix_coefficients, distortion_coefficients, rvec, tvec, length=0.01)

        if args["live"]:
            gray = cv2.aruco.drawDetectedMarkers(gray, corners, ids)

    if ids is None:
        ids = []
    else:
        ids = ids.tolist()
    return gray, ids, pos, ori


# The list of detected marker ids will be an empty list if no markers are detected.
# The following function is used to calculate marker detection rate
def count_empty_lists(lst):
    """
    :param lst: A list containing lists
    :return: The number of empty lists in a lst
    """
    count = 0
    for item in lst:
        if not item:
            count += 1

    return count


def save_data(images, data):
    """
    :param images: List of captured frames
    :param data: Raw frame data containing id, position and orientation of detected marker
    :return: Writes images as file and a structured report as a json file, data.json.
    data.json contains eight objects, stats, args, ids, position, orientation, camera_delay, alg_delay, and timestamp.
    stats reports aggregate metrics including: the duration of experiment, number of captured frames, the frame rate,
    average camera delay for capturing an image, average marker detection algorithm to compute id, position,
    and orientation from a captured image, average of detected marker positions, average of detected marker
    orientations, average distance from the camera to the marker, and marker detection rate (frames with detected marker
    divided by total captured frames).
    args is the arguments passed to the program.
    the remaining object: ids, position, orientation, camera_delay, alg_delay, and timestamp contain the corresponding
    value for each frame.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}_{args['camera']}_{args['res']}")
    images_dir = os.path.join(results_dir, 'images')

    for i in range(len(data["ids"])):
        data["ids"][i] = [j[0] for j in data["ids"][i]]

    for i in range(len(data["position"])):
        for j in range(len(data["position"][i])):
            data["position"][i][j] = [p[0] for p in data["position"][i][j].tolist()]

    for i in range(len(data["orientation"])):
        for j in range(len(data["orientation"][i])):
            data["orientation"][i][j] = data["orientation"][i][j].tolist()

    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    total_ims = len(data["timestamp"])
    exp_dur = data["timestamp"][-1] - data["timestamp"][0]
    filtered_pos = list(filter(bool, data["position"]))
    filtered_ori = list(filter(bool, data["orientation"]))
    stats = {
        "exp_duration": exp_dur,
        "total_images": total_ims,
        "image_per_second": total_ims / exp_dur,
        "avg_camera_delay": sum(data["camera_delay"]) / total_ims,
        "avg_alg_delay": sum(data["alg_delay"]) / total_ims,
        "avg_position": np.mean(filtered_pos, axis=0).tolist(),
        "avg_distance": np.mean(np.linalg.norm(filtered_pos, axis=2), axis=0).tolist(),
        "avg_abs_orientation": np.mean(np.abs(filtered_ori), axis=0).tolist(),
        "detection_rate": 1 - count_empty_lists(data["ids"]) / total_ims,
    }

    data["stats"] = stats

    with open(os.path.join(results_dir, "data.json"), "w") as f:
        json.dump(data, f)

    for t, im in zip(data["timestamp"], images):
        cv2.imwrite(os.path.join(images_dir, f"{t}.jpg"), im)


# This function is useful for defining custom markers, other than APRIL, STAG, or APRILTAG to solve the PnP problem.
# For example the following implementation detects positions of four white circles in a black background and use these
# points in correspondence with 3D custom points defined as marker_points to compute the pose of a custom marker.
def pose_estimation_p4(im, k, d):
    """
    :param im: Captured image
    :param k: Intrinsic matrix of the calibrated camera
    :param d: Distortion coefficients associated with your camera
    :return frame: The frame with the axis drawn on it
    :return ids: Ids of the marker, default is [0]
    :return pos: Positions of the markers
    :return ori: Orientations of the markers
    """
    im = cv2.GaussianBlur(im, (9, 9), 0)
    grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    grey = cv2.threshold(grey, 255 * 0.75, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(im, contours, -1, (255, 0, 0), 5)

    image_points = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] > 100:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), -1)
            image_points.append([center_x, center_y])

    if len(image_points) == 0:
        image_points = []
    else:
        return img, [], [], []

    corners = np.array(image_points, dtype=np.float32)
    marker_points = np.array([[0, 0, 0],
                              [46.4, -2, 0],
                              [44.4, -19.7, 0],
                              [9.4, -29.1, 0]], dtype=np.float32) / 1000

    nada, rvec, tvec = cv2.solvePnP(marker_points, corners, k, d, False, cv2.SOLVEPNP_AP3P)

    rmat, jacobian = cv2.Rodrigues(rvec)

    pose = tvec
    ori = yaw_pitch_roll_decomposition(rmat)
    return img, [0], pose, ori


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--camera", required=True, type=str, help="One of arducam, aideck, pi3, pi3w, or pihq6mm")
    ap.add_argument("-s", "--marker_size", type=float, default=0.02, help="Dimension of marker (meter)")
    ap.add_argument("-k", "--K_Matrix", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-m", "--marker", type=str, default="ARUCO",
                    help="Type of tag to detect. One of ARUCO, APRILTAG, STAG, or P4")
    ap.add_argument("-c", "--dict", type=str, default="DICT_4X4_100", help="Type of dictionary of tag to detect")
    ap.add_argument("-t", "--duration", type=int, default=60, help="Duration of sampling (second)")
    ap.add_argument("-n", "--sample", type=str, default=30, help="Number of samples per second")
    ap.add_argument("-w", "--width", type=int, default=640, help="Width of image")
    ap.add_argument("-y", "--height", type=int, default=480, help="Height of image")
    ap.add_argument("-v", "--live", action="store_true", help="Show live camera image")
    ap.add_argument("-r", "--res", type=str, default="480p",
                    help="Image resolution, one of 244p, 480p, 720p, 1080p, or 1440p, overwrites width and height")
    ap.add_argument("-g", "--debug", action="store_true", help="Print logs")
    ap.add_argument("-o", "--save", action="store_true", help="Save data")
    ap.add_argument("-sm", "--sensor_modes", action="store_true", help="Print sensor modes of the camera and terminate")
    ap.add_argument("-b", "--broadcast", action="store_true",
                    help="Broadcast measurements modify worker socket with proper ip and port")
    ap.add_argument("-mr", "--max_fps", action="store_true", help="Use maximum fps")
    ap.add_argument("-e", "--note", type=str, help="Notes")
    ap.add_argument("-l", "--lenspos", type=float, help="Lens position for manual focus")
    args = vars(ap.parse_args())

    data = {
        "stats": {},
        "args": args,
        "timestamp": [],
        "ids": [],
        "alg_delay": [],
        "camera_delay": [],
        "position": [],
        "orientation": [],
    }

    images = []

    if args["broadcast"]:
        sock = WorkerSocket()

    if args["debug"]:
        logging.getLogger().setLevel(logging.INFO)
    image_size = (args["width"], args["height"])
    if args["res"] is not None:
        image_size = res_map[args["res"]]
    marker_size = args["marker_size"]
    dict_type = args["dict"]
    marker_type = Marker[args["marker"]]
    if args["K_Matrix"] is None:
        calibration_matrix_path = os.path.join("calibration", args["camera"], args["res"], "mtx.npy")
    else:
        calibration_matrix_path = args["K_Matrix"]
    if args["D_Coeff"] is None:
        distortion_coefficients_path = os.path.join("calibration", args["camera"], args["res"], "dist.npy")
    else:
        distortion_coefficients_path = args["D_Coeff"]
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # Make sure to install the libraries before setting the marker_type to STAG or APRILTAG
    if marker_type is Marker.STAG:
        import stag
    elif marker_type is Marker.APRILTAG:
        import apriltag

    if args["live"]:
        cv2.startWindowThread()

    # This block initialize a UDP socket to connect to AiDeck of a Crazyflie to receive video stream from its camera.
    # The AiDeck should be flashed with the video stream program, see AiDeck documents for more details.
    if args["camera"] == "aideck":
        deck_port = 5000
        deck_ip = "192.168.4.1"

        print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((deck_ip, deck_port))
        print("Socket connected")

        imgdata = None
        data_buffer = bytearray()


        def rx_bytes(size):
            data = bytearray()
            while len(data) < size:
                data.extend(client_socket.recv(size - len(data)))
            return data

    # Import libcamera to initialize camera on raspberry pi
    # Image resolution, frame duration limit, and autofocus (if available) is set
    else:
        from picamera2 import Picamera2
        from libcamera import controls

        picam2 = Picamera2(camera_map[args["camera"]])
        modes = picam2.sensor_modes

        if args["sensor_modes"]:
            print(modes)
            exit()

        cam_conf = picam2.create_video_configuration(main={"format": "XRGB8888", "size": image_size})
        picam2.configure(cam_conf)
        if args["max_fps"]:
            picam2.set_controls({"FrameDurationLimits": (8333, 8333)})

        if args["camera"] != "pihq6mm":
            if args["lenspos"]:
                picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": args['lenspos']})
            else:
                picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        picam2.start()
        time.sleep(2)

    # Capture frames from the camera
    exp_start = time.time()
    msg = 0
    while True:
        start = time.time()

        #  The following block applies to Crazyflie's AiDeck
        if args["camera"] == "aideck":
            packetInfoRaw = rx_bytes(4)
            [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
            imgHeader = rx_bytes(length - 2)
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

            if magic == 0xBC:
                imgStream = bytearray()

                while len(imgStream) < size:
                    packetInfoRaw = rx_bytes(4)
                    [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                    chunk = rx_bytes(length - 2)
                    imgStream.extend(chunk)

                if format == 0:
                    bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                    bayer_img.shape = (244, 324)
                    color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                    im = color_img
        else:
            im = picam2.capture_array()

        #  The following code block applies to 4 white circular markers on a black background
        if marker_type == 'P4':
            mid = time.time()
            output, ids, pos, ori = pose_estimation_p4(im, k, d)
            end = time.time()

        else:
            mid = time.time()
            output, ids, pos, ori = pose_estimation(im, k, d, marker_type, dict_type)
            end = time.time()

        if args["broadcast"]:
            if len(pos) and len(ori):
                p = pos[0].tolist()
                msg = f"{p[0][0]},{p[1][0]},{p[2][0]},{ori[0][0]},{ori[0][1]},{ori[0][2]}"
                # print(f"x={p[0][0]:.3f},y={p[1][0]:.3f},z={p[2][0]:.3f},norm={np.linalg.norm(np.array([p[0][0], p[1][0], p[2][0]])):.3f},{ori[0][0]:.2f},{ori[0][1]:.2f},{ori[0][2]:.2f}")

            if msg:
                sock.broadcast(msg)

        data["timestamp"].append(start)
        data["alg_delay"].append(end - mid)
        data["camera_delay"].append(mid - start)
        data["position"].append(pos)
        data["orientation"].append(ori)
        data["ids"].append(ids)

        if args["save"]:
            images.append(im)

        logging.info(f"ids:\n{ids}\n\nposisions:\n{pos}\n\norientations:\n{ori}\n\n")

        if args["live"]:
            cv2.rectangle(output, (image_size[0] // 2 - 50, image_size[1] // 2 - 50),
                          (image_size[0] // 2 + 50, image_size[1] // 2 + 50), (0, 255, 0), 1)
            cv2.imshow('Estimated Pose', output)

        if end - exp_start >= args["duration"]:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("i"):
            marker_size = float(input("input marker size in mm:")) / 1000
        elif key == ord("1"):
            marker_size = 0.02
        elif key == ord("2"):
            marker_size = 0.015
        elif key == ord("3"):
            marker_size = 0.01
        elif key == ord("4"):
            marker_size = 0.005
        elif key == ord("5"):
            marker_size = 0.0025

    if args["live"]:
        cv2.destroyAllWindows()

    if args["save"]:
        save_data(images, data)

    lens_position = picam2.capture_metadata()["LensPosition"]
    logging.info(f"lens position: {lens_position}")
