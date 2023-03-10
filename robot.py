import cv2
import numpy as np
import torch
from customFunctions import *
import pyrealsense2 as rs
import time
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

#Connecting to datbase
conn = sqlite3.connect('xarm.db')
c = conn.cursor()

def getCoordinate():
    #adds the table to store coordinate
    stopDatabase()
    startDatabase()
    #Number of Seconds camera will run
    seconds = 5
    t_end = time.time() + seconds
    while time.time() < t_end:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        #Using the model to find and store coordinates
        results = model(color_image)
        if results.xyxy[0].shape[0] == 0:
            cv2.imshow('YOLO', np.squeeze(results.render()))
        else:
            x_location, y_location = findBolt(results)
            coordinate = (x_location, y_location)
            results2 = cv2.circle(np.squeeze(results.render()), coordinate, 2, (255,0,0), -1)
            cv2.imshow('YOLO', results2)
            depth_point = get_depth(coordinate)
            if depth_point[2] != 0:
                xc = coordinate[0]-320
                yc = coordinate[1]-240
                x = round(1000*depth_point[0])
                y = round(1000*depth_point[1])
                z = round(1000*depth_point[2])
                c.execute("INSERT INTO xarm VALUES(?, ?, ?)", (x, y, z))
                c.execute("INSERT INTO xarm1 VALUES(?, ?)", (xc, yc))
                conn.commit()
        #Store coordinate end of loop
        
        # Exit on ESC
        key = cv2.waitKey(1)
        if key == 27:
            print(fetchdataReal())
            stopDatabase()
            break
    print(fetchdataReal())
    print(fetchdataCamera())
    return fetchdataReal(), fetchdataCamera()
   

#Start Camera to get Coordinate
getCoordinate()

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()