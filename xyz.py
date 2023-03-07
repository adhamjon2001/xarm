import cv2
import numpy as np
import torch
from customFunctions import *
import pyrealsense2 as rs
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/adham/Desktop/opencv/best.pt')
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

# Start streaming
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

def get_depth(pixel):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_frame = pipeline.wait_for_frames().get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth = depth_image[pixel[1], pixel[0]].astype(float) * depth_scale
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel[0], pixel[1]], depth)
    return depth_point

try:
    i = 1
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Rescale depth image to match color image resolution
        depth_image_rescaled = cv2.resize(depth_image, (640, 480))

        # Apply colormap to depth image
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_rescaled, alpha=0.03), cv2.COLORMAP_JET)
      
        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))
        if i == 1:
            results = model(color_image)
            i += 1
            # if results.xyxy[0].shape[0] == 0:
            #     cv2.imshow('YOLO', np.squeeze(results.render()))
            # else:  
            #     coordinate = findBolt(results)
            #     results2 = cv2.circle(np.squeeze(results.render()), coordinate, 2, (255,0,0), -1)
            #     cv2.imshow('YOLO', results2)
            #     depth_point = get_depth(coordinate)
            #     print(depth_point)
            #     #depth = depth_frame.get_distance(x_location, y_location)
            #     print(depth-.093)
        elif i == 4:
            i = 1
        
        #     # Show images
        cv2.imshow('RealSense D435', color_image)
        key = cv2.waitKey(1)

        # Exit on ESC
        if key == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()