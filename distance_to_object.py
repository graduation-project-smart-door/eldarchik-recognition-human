import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import pyvirtualcam


def check_gpu() -> bool:
    gpu_count = len(tf.config.list_physical_devices("GPU"))

    print("Num GPUs Available: ", gpu_count)

    return gpu_count > 0


def set_gpus() -> None:
    if check_gpu():
        gpus = tf.config.experimental.list_physical_devices("GPU")

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[INFO]: skip installing gpu...")

set_gpus()

width, height = 640, 480

fmt = pyvirtualcam.PixelFormat.BGR

# cap = cv2.VideoCapture(0)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

print("[INFO] Starting streaming...")
pipeline.start(config)
print("[INFO] Camera ready.")
# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
print("[INFO] Loading model...")
PATH_TO_CKPT = "model/frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
print("[INFO] Model loaded.")
colors_hash = {}

with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=fmt) as cam:
    print(f'Virtual camera created: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        scaled_size = (color_frame.width, color_frame.height)
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(color_image, axis=0)
        # Perform the actual detection by running the model with the image as input
        distance_to_object = (int(depth_image[120,160]/4/10))
        if(distance_to_object > 50 and distance_to_object < 110): 
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                        feed_dict={image_tensor: image_expanded})
            
            # print(boxes)

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            for idx in range(int(num)):
                class_ = classes[idx]
                score = scores[idx]
                box = boxes[idx]
                
                if class_ not in colors_hash:
                    colors_hash[class_] = tuple(np.random.choice(range(256), size=3))
                
                if score > 0.6:
                    left = int(box[1] * color_frame.width)
                    top = int(box[0] * color_frame.height)
                    right = int(box[3] * color_frame.width)
                    bottom = int(box[2] * color_frame.height)
                    
                    p1 = (left, top)
                    p2 = (right, bottom)
                    # draw box
                    r, g, b = colors_hash[class_]
                    cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)

            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', color_image)
            
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            frame = color_image
            frame = cv2.resize(frame, (width, height))
            cam.send(frame)

            # cv2.imshow('frame', frame)

            # cv2.imshow('RealSense', color_image)
            # print(color_image.shape)
            # print(int(depth_image[120,160]/4/10))
            cv2.waitKey(1)

print("[INFO] stop streaming ...")
pipeline.stop()