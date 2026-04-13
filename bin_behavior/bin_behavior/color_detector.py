import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

BLUE_LOWER  = np.array([100, 60, 60])
BLUE_UPPER  = np.array([130, 255, 255])
GREEN_LOWER = np.array([40, 60, 60])
GREEN_UPPER = np.array([80, 255, 255])
MIN_PIXELS  = 500       # minimum contiguous colored pixels to trigger
COOLDOWN_S  = 5.0       # seconds before re-triggering is allowed


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Bool, '/bin_trigger', 10)
        self.create_subscription(Image, '/camera/rgb/image_raw', self._cb, 10)
        self._last_trigger = self.get_clock().now().nanoseconds * 1e-9 - COOLDOWN_S
        self.get_logger().info('ColorDetector ready')

    def _cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_trigger < COOLDOWN_S:
            return
        frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue  = cv2.inRange(hsv, BLUE_LOWER,  BLUE_UPPER)
        green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        if int(blue.sum() // 255) >= MIN_PIXELS or int(green.sum() // 255) >= MIN_PIXELS:
            self._pub.publish(Bool(data=True))
            self._last_trigger = now
            self.get_logger().info('Color trigger fired')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ColorDetector())
    rclpy.shutdown()
