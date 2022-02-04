#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

max_error = 20
multiplier = 0.1
error_history = 15
K_P = 6
K_D = 0.2
K_I = 1


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
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw", Image, self.callback, queue_size=3)
        self.latest_img = Image()
        self.empty = True

    def callback(self, img):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            self.empty = False
        except CvBridgeError as e:
            print(e)


class PID_controller():
    def __init__(self):
        self.drive_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.camera = image_converter()
        self.last_error = self.get_error()
        self.last_time = rospy.get_time()

        self.error_array = np.ones(error_history)*0.0001
        self.time_array = np.ones(error_history)
        self.index = 0

        self.move_state = 0

    def control_loop(self):
        move = self.get_move_cmd()

        self.drive_pub.publish(move)

    def get_move_cmd(self):
        error = self.get_error()
        move = Twist()

        g = self.calculate_pid2(error)

        move.angular.z = g * multiplier
        move.linear.x = np.interp(np.abs(move.angular.z),[0, 1],[0.5, 0.25])

        print('g', g)
        print('z', move.angular.z)
        print('x', move.linear.x)

        return move


    def calculate_pid(self, error):
        curr_time = rospy.get_time()
        self.error_array[self.index] = error
        self.time_array[self.index] = curr_time

        #derivative
        derivative = np.gradient(self.error_array,self.time_array)[self.index]

        # trapezoidal integration
        integral = np.trapz(self.error_array, self.time_array)

        p = K_P * error
        i = K_I * integral
        d = K_D * derivative

        if (self.index == self.error_array.size - 1):
            self.error_array = np.roll(self.error_array, -1)
            self.time_array = np.roll(self.time_array, -1)
        else:
            self.index += 1

        print("pid ",p,i,d)
        return p + i + d

    def get_error(self):
        xerror = 0

        # check for image
        if not self.camera.empty:
            image = self.camera.latest_img

            # image/dimension bounds
            xcenter = image.shape[1] / 2
            max_reading_error = image.shape[1] / 2
            min_reading_error = 25
            h_threshold = image.shape[0] - 200

            # convert colour
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # cv2.imshow("Image window", image_gray)
            # cv2.waitKey(1)

            gray_clr = [128, 128, 128]

            # locate path pixels based on colour
            Y, X = np.where(np.all(image_gray == gray_clr, axis=2))
            if X.size != 0 and Y.size != 0:
                self.move_state = 0
                displacement = xcenter - int(np.average(X))
                sign = displacement / np.abs(displacement)

                # xerror = map(np.abs(displacement), min_reading_error, 
                # max_reading_error, 0, max)
                xerror = sign * np.interp(np.abs(displacement),
                                          [min_reading_error, max_reading_error],
                                          [0, max_error])
            else:
                xerror = self.last_error
                if self.move_state == 1:
                    self.move_state = 2
                else:
                    self.move_state = 1

        self.last_error = xerror
        return xerror

main()
