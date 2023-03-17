import cv2
import numpy as np
import torch
from customFunctions import *
import pyrealsense2 as rs
import time
import sqlite3
import sys
import math
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI
import threading
#Initial Robot moving position
current = [388.8, -98, 206.1]
new = [450, 0, 206.1]
new1 = [400, -98, 206.1]
#Loading Pre-Trained Model
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
#Connecting to database
conn = sqlite3.connect('xarm.db')
c = conn.cursor()

# This Function Will Return Real Coordinate From Pixel
def get_depth(pixel):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_frame = pipeline.wait_for_frames().get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth = depth_image[pixel[1], pixel[0]].astype(float) * depth_scale
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel[0], pixel[1]], depth)
    return depth_point
#This Function Will Find The First Bolt In The Frame
def findBolt(results):
    bolts = results.xyxy[0]
    count = bolts.shape[0]
    i = 0
    total = []
    while i < count:
        first = bolts[i]
        midx = first[2] + first[0]
        midy = first[3] + first[1]
        x_location = midx/2
        y_location = midy/2
        x_location = x_location.item()
        x_location = round(x_location)
        y_location = y_location.item()
        y_location = round(y_location)
        total.append([x_location, y_location])
        i += 1
        total.sort()
        #print(x_location)
        coordinate = (total[0][0], total[0][1])
    return coordinate
#Call this function to create a temporary database and table
def startDatabase():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("CREATE TABLE xarm (x REAL, y REAL, z REAL)")
    c.execute("CREATE TABLE xarm1 (x REAL, y REAL)")
    #commit changes
    print("database created")
    conn.commit()
    #close connection
    conn.close()
    return
#This Function Will fetch the average real of world coordinate
def fetchdataReal():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("SELECT * FROM xarm")
    x = 0
    y = 0
    z = 0
    items = c.fetchall()
    if len(items) != 0:
        for item in items:
            x = x + item[0]
            y = y + item[1]
            z = z + item[2]
        x = x/len(items)
        x = round(x, 1)
        y = y/len(items)
        y = round(y, 1)
        z = z/len(items)
        z = round(z, 1)
    return x, y, z
#This will fetch average camera coordinates .... maybe not needed
def fetchdataCamera():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("SELECT * FROM xarm1")
    x = 0
    y = 0
    items = c.fetchall()
    if len(items) != 0:
        for item in items:
            x = x + item[0]
            y = y + item[1]
        x = x/len(items)
        x = round(x)
        y = y/len(items)
        y = round(y)
    return x, y
#This will Delete the tables create to store value in database
def stopDatabase():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("DROP TABLE xarm")
    c.execute("DROP TABLE xarm1")
    conn.commit()
    print("database deleted")
    #close connection
    conn.close()
    return
#stopDatabase()
#fetchdata()
#startDatabase()

def getAverage(average):
    #Number of Seconds camera will run
    totalx = 0
    totaly = 0
    totalz = 0
    for i in range(25):
       totalx = totalx + average[i][0]
       totaly = totaly + average[i][1]
       totalz = totalz + average[i][2]
    
    totalx = totalx/25
    totaly = totaly/25
    totalz = totalz/25
    total = [totalx, totaly, totalz]
    return total
   

#Start Camera to get Coordinate

class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
   
    def run(self):
        halfway = True
        try:
            # Movement_testting
            self._angle_speed = 2
            self._angle_acc = 50
            thread_index = -1
            move = 5
            average = []
            initial = True
            def moveRobot(current):
                code = self._arm.set_position(*[current[0], current[1], current[2], 59.3, -88.7, 121.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=-1.0, wait=True)
                if not self._check_code(code, 'set_position'):
                    return
            threads = [threading.Thread(target=moveRobot(current))]
            while True:
                if not self.is_alive:
                    break
                i = 1
                while i == 1:
                    code = self._arm.set_position(*[current[0], current[1], current[2], 59.3, -88.7, 121.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=-1.0, wait=True)
                    if not self._check_code(code, 'set_position'):
                        return
                    i += 1
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                #Using the model to find and store coordinates
                results = model(color_image)
                #Check if there is any bolt in frame
                if results.xyxy[0].shape[0] == 0:
                    cv2.imshow('YOLO', np.squeeze(results.render()))
                else:
                    pixel = []
                    coordinate = findBolt(results)
                    results2 = cv2.circle(np.squeeze(results.render()), coordinate, 2, (255,0,0), -1)
                    cv2.imshow('YOLO', results2)
                    #Moving camera to where the distance read is accurate
                    pixel = [coordinate[0], coordinate[1]]
                    pixel[0] = pixel[0]-320
                    pixel[1] = pixel[1]-240
                    # coordinate[0] = coordinate[0]-320
                    # coordinate[1] = coordinate[1]-240
                    if pixel[0] < -20 or pixel[0] > 25 or pixel[1] < -25 or pixel[1] > 20:
                        if threads[thread_index].is_alive() ==  False:
                            thread_index += 1
                            if pixel[0] < -20 or get_depth(coordinate)[2] == 0:
                                current[1] = current[1] + move
                            if pixel[0] > 25:
                                current[1] = current[1] - move
                            if pixel[1] < -25:
                                current[2] = current[2] + move
                            if pixel[1] > 20:
                                current[2] = current[2] - move
                            threads[thread_index].start()
                            threads.append(threading.Thread(target=moveRobot(current)))
                    elif pixel[0] > -20 or pixel[0] < 25 or pixel[1] > -25 or pixel[1] < 20:
                        print("elif before while: ")
                        while halfway:
                            print("in while: ")
                            depth = get_depth(coordinate)
                            current[0] = current[0] + (((1000*depth[2])-245)/2)
                            if not self.is_alive:
                                break
                            moveRobot(current)
                            halfway = False
                            move = 2
                        if pixel[0] <= -3 or pixel[0] >= 3 or pixel[1] <= -3 or pixel[1] >= 3:
                            print("if 1 :")
                            if threads[thread_index].is_alive() ==  False:
                                print("if 2")
                                thread_index += 1
                                if pixel[0] <= -3 or get_depth(coordinate)[2] == 0:
                                    current[1] = current[1] + move
                                if pixel[0] >= 3:
                                    current[1] = current[1] - move
                                if pixel[1] <= -3:
                                    current[2] = current[2] + move
                                if pixel[1] >= 3:
                                    current[2] = current[2] - move
                                threads[thread_index].start()
                                threads.append(threading.Thread(target=moveRobot(current))) 
                    print(pixel)               
                    if pixel[0] > -3 and pixel[0] < 3 and pixel[1] > -3 and pixel[1] < 3:
                        depth = get_depth(coordinate)
                        #depth[2] = depth[2]/2
                        print(depth)
                        current[2] = current[2] + (1000*depth[1]) + 97.3
                        print(current[0])
                        current[0] = current[0] + (1000*depth[2])-245
                        print(current[0])
                        current[1] = current[1] + (1000*depth[0]) + 45
                        if not self.is_alive:
                            break
                        moveRobot(current)
                        break

                key = cv2.waitKey(1)
                if key == 27:
                    break
            pipeline.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)
#robot will turn on and go to current location
if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.203', baud_checkset=False)
    robot_main = RobotMain(arm)
    robot_main.run()
