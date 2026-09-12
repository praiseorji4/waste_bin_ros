"""Peace-sign detector: opens the bin when someone shows a V sign to the camera.

Uses MediaPipe Hands (21 hand landmarks) and classifies the gesture geometrically,
rather than a trained detector. That means no dataset, no training, and it runs at
roughly 15 fps on a Raspberry Pi 4 where a YOLO model manages 1-2 fps.

Interface is identical to bin_behavior/color_detector, so this node is a drop-in
replacement: it publishes std_msgs/Bool on /bin_trigger and behavior_manager does
the rest (open lid, wait, close, dock).

Requires mediapipe:
    sudo apt install -y python3-pip
    python3 -m pip install --break-system-packages mediapipe
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

FINGERS = {
    'index':  (INDEX_TIP, INDEX_PIP),
    'middle': (MIDDLE_TIP, MIDDLE_PIP),
    'ring':   (RING_TIP, RING_PIP),
    'pinky':  (PINKY_TIP, PINKY_PIP),
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended(points, tip_idx, pip_idx, margin=1.15):
    """True if the finger is straight out.

    Compares tip-to-wrist against pip-to-wrist distance. This is orientation
    independent, so it still works with the hand upside down or sideways -- unlike
    the common "tip is above pip in image y" shortcut.
    """
    wrist = points[WRIST]
    return _dist(points[tip_idx], wrist) > _dist(points[pip_idx], wrist) * margin


def thumb_extended(points, margin=1.12):
    wrist = points[WRIST]
    return _dist(points[THUMB_TIP], wrist) > _dist(points[THUMB_MCP], wrist) * margin


def v_angle_deg(points):
    """Angle between the index and middle finger directions, in degrees."""
    def direction(mcp, tip):
        return (points[tip][0] - points[mcp][0], points[tip][1] - points[mcp][1])

    ax, ay = direction(INDEX_MCP, INDEX_TIP)
    bx, by = direction(MIDDLE_MCP, MIDDLE_TIP)
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))


def hand_span(points):
    """Rough hand size in pixels: wrist to middle MCP. Used to reject far-away hands."""
    return _dist(points[WRIST], points[MIDDLE_MCP])


def classify_peace(points, min_v_angle=12.0, max_v_angle=70.0, allow_thumb=True):
    """Decide whether 21 landmark points (pixel coords) form a peace sign.

    Returns (is_peace, details) so the caller can log or draw why it failed.
    """
    details = {
        'index': finger_extended(points, *FINGERS['index']),
        'middle': finger_extended(points, *FINGERS['middle']),
        'ring': finger_extended(points, *FINGERS['ring']),
        'pinky': finger_extended(points, *FINGERS['pinky']),
        'thumb': thumb_extended(points),
        'v_angle': v_angle_deg(points),
        'span': hand_span(points),
    }
    ok = (
        details['index']
        and details['middle']
        and not details['ring']
        and not details['pinky']
        and min_v_angle <= details['v_angle'] <= max_v_angle
    )
    if ok and not allow_thumb and details['thumb']:
        ok = False
    details['is_peace'] = ok
    return ok, details


class PeaceDetector(Node):
    def __init__(self):
        super().__init__('peace_detector')

        self.declare_parameter('input_topic', '/camera/rgb/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('trigger_topic', '/bin_trigger')
        self.declare_parameter('debug_topic', '/peace_detector/debug_image')
        self.declare_parameter('publish_debug', True)
        # Hold the gesture for this many accepted frames before firing. Stops a
        # single blurry frame from opening the lid.
        self.declare_parameter('consecutive_frames', 4)
        self.declare_parameter('cooldown_s', 5.0)
        # Process 1 in N frames. On a Pi, 2-3 keeps CPU sane; the gesture is held
        # for a second or more anyway.
        self.declare_parameter('process_every_n', 2)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('min_hand_span_px', 25.0)
        self.declare_parameter('allow_thumb_out', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.publish_debug = self.get_parameter('publish_debug').value
        self.need_frames = int(self.get_parameter('consecutive_frames').value)
        self.cooldown_s = float(self.get_parameter('cooldown_s').value)
        self.every_n = max(1, int(self.get_parameter('process_every_n').value))
        self.min_span = float(self.get_parameter('min_hand_span_px').value)
        self.allow_thumb = bool(self.get_parameter('allow_thumb_out').value)

        try:
            import mediapipe as mp
        except ImportError:
            self.get_logger().fatal(
                'mediapipe is not installed. Run:\n'
                '  sudo apt install -y python3-pip\n'
                '  python3 -m pip install --break-system-packages mediapipe')
            raise

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,           # lightest model: matters on the Pi
            min_detection_confidence=float(self.get_parameter('min_detection_confidence').value),
            min_tracking_confidence=float(self.get_parameter('min_tracking_confidence').value),
        )

        self._bridge = CvBridge()
        self._streak = 0
        self._frame_i = 0
        self._last_trigger = self.get_clock().now().nanoseconds * 1e-9 - self.cooldown_s

        self._trigger_pub = self.create_publisher(
            Bool, self.get_parameter('trigger_topic').value, 10)
        self._debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_topic').value, 10)

        msg_type = CompressedImage if self.use_compressed else Image
        self.create_subscription(msg_type, self.input_topic, self._image_cb, 10)
        self.get_logger().info(
            f'PeaceDetector ready: listening on {self.input_topic} '
            f'({"compressed" if self.use_compressed else "raw"}), '
            f'{self.need_frames} frames to fire, {self.cooldown_s:.1f}s cooldown')

    def _decode(self, msg):
        if self.use_compressed:
            return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        return self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _image_cb(self, msg):
        self._frame_i += 1
        if self._frame_i % self.every_n:
            return

        frame = self._decode(msg)
        if frame is None:
            self.get_logger().warn('Failed to decode image')
            return

        h, w = frame.shape[:2]
        results = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        found = False
        best = None
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                pts = [(lm.x * w, lm.y * h) for lm in hand.landmark]
                is_peace, details = classify_peace(pts, allow_thumb=self.allow_thumb)
                if details['span'] < self.min_span:
                    is_peace = False
                    details['is_peace'] = False
                    details['too_small'] = True
                if self.publish_debug:
                    self._draw(frame, hand, pts, details)
                if is_peace:
                    found = True
                    best = details

        now = self.get_clock().now().nanoseconds * 1e-9
        if found:
            self._streak += 1
        else:
            self._streak = 0

        fired = False
        if self._streak >= self.need_frames and now - self._last_trigger >= self.cooldown_s:
            self._trigger_pub.publish(Bool(data=True))
            self._last_trigger = now
            self._streak = 0
            fired = True
            self.get_logger().info(
                'Peace sign confirmed - opening bin '
                f'(V angle {best["v_angle"]:.0f} deg, span {best["span"]:.0f} px)')

        if self.publish_debug:
            banner = 'BIN OPENING' if fired else (
                f'peace {self._streak}/{self.need_frames}' if self._streak else 'no peace sign')
            color = (0, 0, 255) if fired else ((0, 200, 0) if self._streak else (140, 140, 140))
            cv2.putText(frame, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            out = self._bridge.cv2_to_imgmsg(frame, 'bgr8')
            out.header = msg.header
            self._debug_pub.publish(out)

    def _draw(self, frame, hand, pts, details):
        self._mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, self._mp.solutions.hands.HAND_CONNECTIONS)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        ok = details['is_peace']
        color = (0, 200, 0) if ok else (0, 140, 255)
        cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), color, 2)
        fingers = ''.join(k[0].upper() if details[k] else '-'
                          for k in ('index', 'middle', 'ring', 'pinky'))
        cv2.putText(frame, f'{fingers} V={details["v_angle"]:.0f}',
                    (x1 - 8, max(y1 - 14, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main(args=None):
    rclpy.init(args=args)
    node = PeaceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
