import cv2
import numpy as np
import pyrealsense2 as rs
x = [300,300]
y = [150,300]
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('color image', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        mouse_click = cv2.getWindowImageRect('color image')[0:2] + (0, 0)
        mouse_click = np.array(mouse_click[::-1])
        depth_point = get_depth(x)
        depth_point1 = get_depth(y)
        print('xyz coordinate:', depth_point)
        print('xyz coordinate:', depth_point1)
    