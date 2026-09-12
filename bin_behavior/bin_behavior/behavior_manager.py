"""Lid sequence: open -> dwell -> close -> (optionally dock).

Driven by the reported lid angle rather than fixed timers. The previous version
waited 3 s after commanding open and 1.5 s after commanding close, which went wrong
in both directions:

  * In simulation the lid needs ~2.5 s to travel, so it was still 13 degrees open
    when the sequence declared it closed and resumed navigation -- with the LD19
    (bolted to the lid) still pitched, so scan_gate blanked the laser while Nav2
    was driving.
  * On the real robot the servo sweeps in ~0.35 s, so the same fixed timers gave a
    real 3 s dwell there and effectively none in sim. The two behaved differently.

Now each phase waits for bin_cover_joint to actually reach its target, with a
timeout so a jammed servo cannot hang the sequence, and the dwell is the time the
lid is really open.

Parameters
    joint_name                      lid joint to watch in /joint_states
    open_position, closed_position  lid angles (rad)
    position_tolerance              how close counts as arrived (rad)
    dwell_s                         how long to hold the lid open once it IS open
    move_timeout_s                  give up waiting for the lid and carry on
    dock_after_close                run the Nav2 dock goal. Default False: Nav2 runs
                                    on the laptop, not on the robot, so on-robot the
                                    goal would fail on every cycle.
    dock_x, dock_y, dock_yaw        dock pose in the map frame
    dock_timeout_s                  abandon a dock that never returns a result
"""
import math
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray


class BehaviorManager(Node):
    IDLE, OPENING, DWELL, CLOSING, DOCKING = range(5)
    STATE_NAMES = {IDLE: 'IDLE', OPENING: 'OPENING', DWELL: 'DWELL',
                   CLOSING: 'CLOSING', DOCKING: 'DOCKING'}

    def __init__(self):
        super().__init__('behavior_manager')

        self.declare_parameter('joint_name', 'bin_cover_joint')
        self.declare_parameter('open_position', -math.pi / 2)
        self.declare_parameter('closed_position', 0.0)
        self.declare_parameter('position_tolerance', 0.05)
        self.declare_parameter('dwell_s', 3.0)
        self.declare_parameter('move_timeout_s', 6.0)
        self.declare_parameter('dock_after_close', False)
        self.declare_parameter('dock_x', 0.0)
        self.declare_parameter('dock_y', 0.0)
        self.declare_parameter('dock_yaw', 0.0)
        self.declare_parameter('dock_timeout_s', 120.0)
        self.declare_parameter('cancel_nav_on_trigger', True)

        self.joint_name = self.get_parameter('joint_name').value
        self.open_pos = float(self.get_parameter('open_position').value)
        self.closed_pos = float(self.get_parameter('closed_position').value)
        self.tol = float(self.get_parameter('position_tolerance').value)
        self.dwell_s = float(self.get_parameter('dwell_s').value)
        self.move_timeout_s = float(self.get_parameter('move_timeout_s').value)
        self.dock_after_close = bool(self.get_parameter('dock_after_close').value)
        self.dock_timeout_s = float(self.get_parameter('dock_timeout_s').value)
        self.cancel_nav = bool(self.get_parameter('cancel_nav_on_trigger').value)

        cb = ReentrantCallbackGroup()
        # One lock around every state transition. The executor is multi-threaded and
        # the callbacks are reentrant, so without it two triggers arriving together
        # can both pass the IDLE check and start the sequence twice.
        self._lock = threading.Lock()
        self._state = self.IDLE
        self._lid_pos = None
        self._phase_deadline = None
        self._nav_goal_handle = None

        self._cover_pub = self.create_publisher(
            Float64MultiArray, '/bin_cover_controller/commands', 10)
        self.create_subscription(Bool, '/bin_trigger', self._trigger_cb, 10, callback_group=cb)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10,
                                 callback_group=cb)
        self._nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose', callback_group=cb)

        # A single periodic tick drives the state machine: no per-phase timers to
        # create, destroy and leak.
        self.create_timer(0.05, self._tick, callback_group=cb)

        self.get_logger().info(
            f'BehaviorManager ready (dwell {self.dwell_s:.1f}s, '
            f'dock_after_close={self.dock_after_close})')

    # ── helpers ──────────────────────────────────────────────────────────

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_cover(self, position):
        msg = Float64MultiArray()
        msg.data = [position]
        self._cover_pub.publish(msg)

    def _at(self, target):
        return self._lid_pos is not None and abs(self._lid_pos - target) <= self.tol

    def _enter(self, state, timeout_s=None):
        self._state = state
        self._phase_deadline = None if timeout_s is None else self._now() + timeout_s
        self.get_logger().info(f'-> {self.STATE_NAMES[state]}')

    def _expired(self):
        return self._phase_deadline is not None and self._now() >= self._phase_deadline

    # ── callbacks ────────────────────────────────────────────────────────

    def _joint_cb(self, msg):
        try:
            i = msg.name.index(self.joint_name)
        except ValueError:
            return
        if i < len(msg.position):
            self._lid_pos = msg.position[i]

    def _trigger_cb(self, msg):
        if not msg.data:
            return
        with self._lock:
            if self._state != self.IDLE:
                self.get_logger().debug(
                    f'trigger ignored, busy in {self.STATE_NAMES[self._state]}')
                return
            self.get_logger().info('Trigger received - opening lid')
            if self.cancel_nav and self._nav_goal_handle is not None:
                self._nav_goal_handle.cancel_goal_async()
                self._nav_goal_handle = None
            self._set_cover(self.open_pos)
            self._enter(self.OPENING, self.move_timeout_s)

    def _tick(self):
        with self._lock:
            if self._state == self.OPENING:
                if self._at(self.open_pos):
                    self.get_logger().info('Lid open - holding')
                    self._enter(self.DWELL, self.dwell_s)
                elif self._expired():
                    self.get_logger().warn(
                        f'Lid did not reach {self.open_pos:.3f} rad within '
                        f'{self.move_timeout_s:.1f}s (at {self._lid_pos}); continuing')
                    self._enter(self.DWELL, self.dwell_s)

            elif self._state == self.DWELL:
                if self._expired():
                    self._set_cover(self.closed_pos)
                    self._enter(self.CLOSING, self.move_timeout_s)

            elif self._state == self.CLOSING:
                if self._at(self.closed_pos):
                    self.get_logger().info('Lid closed')
                    self._after_close()
                elif self._expired():
                    self.get_logger().warn(
                        f'Lid did not close within {self.move_timeout_s:.1f}s '
                        f'(at {self._lid_pos}); continuing')
                    self._after_close()

            elif self._state == self.DOCKING:
                # Only reached if the action result never arrives -- without this the
                # bin would stay in DOCKING forever and never open again.
                if self._expired():
                    self.get_logger().warn('Dock timed out - cancelling, back to IDLE')
                    if self._nav_goal_handle is not None:
                        self._nav_goal_handle.cancel_goal_async()
                        self._nav_goal_handle = None
                    self._enter(self.IDLE)

    def _after_close(self):
        if not self.dock_after_close:
            self._enter(self.IDLE)
            return
        self._enter(self.DOCKING, self.dock_timeout_s)
        # Non-blocking check: wait_for_server() would stall this callback for seconds
        # on a robot where Nav2 is not running.
        if not self._nav_client.server_is_ready():
            self.get_logger().warn('NavigateToPose server not available - skipping dock')
            self._enter(self.IDLE)
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(self.get_parameter('dock_x').value)
        goal.pose.pose.position.y = float(self.get_parameter('dock_y').value)
        yaw = float(self.get_parameter('dock_yaw').value)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self._nav_client.send_goal_async(goal).add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Dock goal rejected')
            with self._lock:
                self._enter(self.IDLE)
            return
        self._nav_goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, _future):
        self._nav_goal_handle = None
        with self._lock:
            if self._state == self.DOCKING:
                self.get_logger().info('Docking complete')
                self._enter(self.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorManager()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
