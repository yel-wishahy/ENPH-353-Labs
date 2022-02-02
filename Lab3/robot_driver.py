#!/usr/bin/env python

from dis import dis
import rospy
from geometry_msgs.msg import Twist
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

max_error = 10

def main():
    rospy.init_node('robot_driver')
    controller = PID_controller()
    rate = rospy.Rate(2)

    while not rospy.is_shutdown():
        controller.control_loop()
        rate.sleep()



class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw",Image ,self.callback,queue_size=3)
        self.latest_img = Image()
        self.empty = True
    def callback(self, img):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(img,"bgr8")   
            self.empty = False
        except CvBridgeError as e:
            print(e)            


class PID_controller():
    def __init__(self):
        self.drive_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.camera = image_converter()
        self.last_error = 0
        self.last_time = 0


    def control_loop(self):
        error = self.get_raw_error()
        move = Twist()
        move.linear.x = 0.25
        move.angular.z = error
  
        self.drive_pub.publish(move)    


    def get_raw_error(self):
        xerror = 0

        #check for image
        if not self.camera.empty:
            image = self.camera.latest_img

            #image/dimension bounds
            xcenter = image.shape[1]/2
            max_reading_error = image.shape[1]/2
            min_reading_error = 25
            h_threshold = image.shape[0] - 200

            #convert colour
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Image window", image_gray)
            cv2.waitKey(3)

            gray_clr = [128,128,128]

            #locate path pixels based on colour
            Y,X = np.where(np.all(image_gray==gray_clr,axis=2))
            displacement = 0
            sign = 0
            if X.size != 0 and Y.size != 0:
                displacement = xcenter - int(np.average(X))
                sign = displacement/np.abs(displacement) 

                xerror = sign*np.interp(np.abs(displacement),
                [min_reading_error,max_reading_error],
                [0,max_error])
            
            if np.median(Y) <= h_threshold:
                print('estimated horizontal')
                xerror = max_error
            elif xerror == 0 and self.last_error != 0:
                xerror = self.last_error

            print('error: ',xerror)

        self.last_error = xerror
        return xerror

main()