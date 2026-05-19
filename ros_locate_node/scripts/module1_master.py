#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import math
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

try:
    from ros_AIUI_node.srv import textToSpeak
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class Module1Master(object):
    STATE_IDLE = "STATE_IDLE"
    STATE_TURN_TO_SOUND = "STATE_TURN_TO_SOUND"
    STATE_VISION_LOCK = "STATE_VISION_LOCK"
    STATE_WAIT_FOR_STOP = "STATE_WAIT_FOR_STOP"

    MAX_DX = 0.1
    MAX_DY = 0.05
    MAX_THETA = 10.0

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.state = self.STATE_IDLE
        self.next_after_stop = self.STATE_VISION_LOCK
        self.walking_status = 0.0
        self.latest_frame = None
        self.latest_frame_stamp = None
        self.pending_sound_angle = None
        self.wait_seen_walking = False
        self.search_step_count = 0

        self.image_topic = rospy.get_param("~image_topic", "/chin_camera/image")
        self.wakeup_topic = rospy.get_param("~wakeup_topic", "/micarrays/wakeup")
        self.walking_status_topic = rospy.get_param(
            "~walking_status_topic", "/MediumSize/BodyHub/WalkingStatus"
        )
        self.gait_command_topic = rospy.get_param("~gait_command_topic", "/gaitCommand")
        self.capture_path = rospy.get_param("~capture_path", "/tmp/target_captured.jpg")

        self.arrival_area_ratio = float(rospy.get_param("~arrival_area_ratio", 0.35))
        self.min_area_ratio = float(rospy.get_param("~min_area_ratio", 0.002))
        self.center_deadband_px = float(rospy.get_param("~center_deadband_px", 35.0))
        self.theta_gain = float(rospy.get_param("~theta_gain", 18.0))
        self.forward_gain = float(rospy.get_param("~forward_gain", 0.25))
        self.min_forward_step = float(rospy.get_param("~min_forward_step", 0.02))
        self.search_max_steps = int(rospy.get_param("~search_max_steps", 36))

        # HSV defaults target a red/orange marker. Tune via rosparam for the actual target.
        self.hsv_lower1 = self._load_hsv_param("~hsv_lower1", [0, 80, 50])
        self.hsv_upper1 = self._load_hsv_param("~hsv_upper1", [12, 255, 255])
        self.hsv_lower2 = self._load_hsv_param("~hsv_lower2", [170, 80, 50])
        self.hsv_upper2 = self._load_hsv_param("~hsv_upper2", [180, 255, 255])

        self.arrived_topic = rospy.get_param("~arrived_topic", "/module1/arrived")
        self.tts_topic = rospy.get_param("~tts_topic", "/aiui/text_to_speak")

        self.gait_pub = rospy.Publisher(
            self.gait_command_topic, Float64MultiArray, queue_size=2
        )
        self.arrived_pub = rospy.Publisher(
            self.arrived_topic, Bool, queue_size=1, latch=True
        )
        rospy.Subscriber(self.wakeup_topic, String, self._wakeup_callback, queue_size=5)
        rospy.Subscriber(
            self.walking_status_topic, Float64, self._walking_status_callback, queue_size=5
        )
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)

        self._tts_client = None
        self._tts_available = False
        if _TTS_AVAILABLE:
            try:
                rospy.wait_for_service(self.tts_topic, timeout=3.0)
                self._tts_client = rospy.ServiceProxy(self.tts_topic, textToSpeak)
                self._tts_available = True
                rospy.loginfo("TTS service %s is available", self.tts_topic)
            except rospy.ROSException:
                rospy.logwarn(
                    "TTS service %s not available, voice prompts disabled", self.tts_topic
                )
        else:
            rospy.logwarn("ros_AIUI_node not in dependency, TTS voice prompts disabled")

        rospy.loginfo(
            "module1_master initialized: search_max_steps=%d, arrival_area_ratio=%.2f",
            self.search_max_steps,
            self.arrival_area_ratio,
        )

    def _speak(self, text):
        if not self._tts_available or self._tts_client is None:
            rospy.logwarn("TTS skip (unavailable): %s", text)
            return
        try:
            self._tts_client(text)
            rospy.loginfo("TTS: %s", text)
        except rospy.ServiceException as exc:
            rospy.logwarn("TTS service call failed: %s", exc)
        except rospy.ROSException as exc:
            rospy.logwarn("TTS ros exception: %s", exc)

    def _load_hsv_param(self, name, default):
        value = rospy.get_param(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("Invalid %s, using default %s", name, default)
            value = default
        return np.array([int(self._clamp(v, 0, 255)) for v in value], dtype=np.uint8)

    def _wakeup_callback(self, msg):
        try:
            payload = ast.literal_eval(msg.data)
            angle = float(payload.get("angle"))
        except (ValueError, SyntaxError, TypeError, AttributeError) as exc:
            rospy.logwarn("Failed to parse wakeup payload %r: %s", msg.data, exc)
            return

        with self.lock:
            if self.state != self.STATE_IDLE:
                rospy.logwarn("Wakeup ignored while state=%s", self.state)
                return
            self.pending_sound_angle = self._normalize_angle(angle)
            self.search_step_count = 0
            self.state = self.STATE_TURN_TO_SOUND
        rospy.loginfo("Wakeup angle %.2f deg accepted", angle)

    def _walking_status_callback(self, msg):
        with self.lock:
            self.walking_status = float(msg.data)
            if self.state == self.STATE_WAIT_FOR_STOP and self.walking_status > 0.5:
                self.wait_seen_walking = True

    def _image_callback(self, msg):
        if self._is_walking():
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge conversion failed: %s", exc)
            return

        with self.lock:
            self.latest_frame = frame
            self.latest_frame_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def run(self):
        rospy.loginfo("module1_master started")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as exc:
                rospy.logerr("FSM step failed: %s", exc)
                self._safe_to_idle()
            rate.sleep()

    def _step(self):
        state = self._get_state()
        if state == self.STATE_IDLE:
            return
        if state == self.STATE_TURN_TO_SOUND:
            self._handle_turn_to_sound()
            return
        if state == self.STATE_VISION_LOCK:
            self._handle_vision_lock()
            return
        if state == self.STATE_WAIT_FOR_STOP:
            self._handle_wait_for_stop()
            return
        rospy.logwarn("Unknown state %s, resetting to idle", state)
        self._safe_to_idle()

    def _handle_turn_to_sound(self):
        if not self._is_stopped():
            self._enter_wait_for_stop(self.STATE_TURN_TO_SOUND)
            return

        with self.lock:
            angle = self.pending_sound_angle

        if angle is None or abs(angle) <= 1.0:
            with self.lock:
                self.pending_sound_angle = None
                self.search_step_count = 0
                self.state = self.STATE_VISION_LOCK
            rospy.loginfo("[FSM] -> STATE_VISION_LOCK (sound angle cleared)")
            return

        step_theta = self._clamp(angle, -self.MAX_THETA, self.MAX_THETA)
        remaining = self._normalize_angle(angle - step_theta)
        with self.lock:
            self.pending_sound_angle = remaining if abs(remaining) > 1.0 else None
            self.next_after_stop = (
                self.STATE_TURN_TO_SOUND if self.pending_sound_angle is not None else self.STATE_VISION_LOCK
            )

        if self._publish_gait(0.0, 0.0, step_theta):
            self._enter_wait_for_stop()
        else:
            self._set_state(self.next_after_stop)

    def _handle_vision_lock(self):
        if not self._is_stopped():
            self._enter_wait_for_stop(self.STATE_VISION_LOCK)
            return

        frame = self._get_latest_frame()
        if frame is None:
            rospy.logwarn_throttle(2.0, "Waiting for latest image frame")
            return

        target = self._detect_target(frame)
        if target is None:
            with self.lock:
                self.search_step_count += 1
                current_count = self.search_step_count

            if current_count > self.search_max_steps:
                rospy.logwarn(
                    "Search limit reached (%d steps), giving up", current_count
                )
                self._speak("未发现目标")
                with self.lock:
                    self.search_step_count = 0
                self._safe_to_idle()
                return

            rospy.logwarn_throttle(
                1.0,
                "No target found, rotating search step %d/%d",
                current_count,
                self.search_max_steps,
            )
            if self._publish_gait(0.0, 0.0, 5.0):
                self._enter_wait_for_stop(self.STATE_VISION_LOCK)
            return

        with self.lock:
            self.search_step_count = 0

        cx, cy, area_ratio = target
        height, width = frame.shape[:2]
        pixel_dx = cx - (width / 2.0)

        rospy.loginfo(
            "target cx=%.1f cy=%.1f pixel_dx=%.1f area_ratio=%.3f",
            cx,
            cy,
            pixel_dx,
            area_ratio,
        )

        if area_ratio > self.arrival_area_ratio:
            if cv2.imwrite(self.capture_path, frame):
                rospy.loginfo("Target reached, saved frame to %s", self.capture_path)
            else:
                rospy.logerr("Target reached, but failed to save %s", self.capture_path)
            self.arrived_pub.publish(Bool(data=True))
            rospy.loginfo("Published arrival signal on %s", self.arrived_topic)
            self._speak("进行下一步吧")
            self._safe_to_idle()
            return

        theta = 0.0
        if abs(pixel_dx) > self.center_deadband_px:
            normalized_dx = pixel_dx / max(width / 2.0, 1.0)
            theta = self._clamp(-normalized_dx * self.theta_gain, -self.MAX_THETA, self.MAX_THETA)

        dx = self.forward_gain * max(self.arrival_area_ratio - area_ratio, 0.0)
        dx = self._clamp(dx, 0.0, self.MAX_DX)
        if dx < self.min_forward_step and abs(theta) <= 0.1:
            dx = self.min_forward_step

        if self._publish_gait(dx, 0.0, theta):
            self._enter_wait_for_stop(self.STATE_VISION_LOCK)

    def _handle_wait_for_stop(self):
        if self._is_walking():
            with self.lock:
                self.wait_seen_walking = True
            return

        with self.lock:
            wait_seen_walking = self.wait_seen_walking

        if not wait_seen_walking:
            rospy.logwarn_throttle(2.0, "Waiting for WalkingStatus to enter walking state")
            return

        if not self._is_stopped():
            return

        with self.lock:
            self.state = self.next_after_stop
            self.wait_seen_walking = False
        rospy.loginfo("Walking stopped; switching to %s", self._get_state())

    def _detect_target(self, frame):
        if frame.size == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        image_area = float(frame.shape[0] * frame.shape[1])
        if image_area <= 0.0:
            return None

        area_ratio = area / image_area
        if area_ratio < self.min_area_ratio:
            return None

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return cx, cy, area_ratio

    def _publish_gait(self, dx, dy, theta):
        safe_dx = self._clamp(dx, -self.MAX_DX, self.MAX_DX)
        safe_dy = self._clamp(dy, -self.MAX_DY, self.MAX_DY)
        safe_theta = self._clamp(theta, -self.MAX_THETA, self.MAX_THETA)

        if abs(dx - safe_dx) > 1e-6 or abs(dy - safe_dy) > 1e-6 or abs(theta - safe_theta) > 1e-6:
            rospy.logwarn(
                "Gait command clamped from [%.3f, %.3f, %.3f] to [%.3f, %.3f, %.3f]",
                dx,
                dy,
                theta,
                safe_dx,
                safe_dy,
                safe_theta,
            )

        if abs(safe_dx) < 1e-6 and abs(safe_dy) < 1e-6 and abs(safe_theta) < 1e-6:
            rospy.logwarn("Zero gait command suppressed")
            return False

        self.gait_pub.publish(Float64MultiArray(data=[safe_dx, safe_dy, safe_theta]))
        rospy.loginfo("Published gait [dx=%.3f, dy=%.3f, theta=%.3f]", safe_dx, safe_dy, safe_theta)
        return True

    def _is_stopped(self):
        with self.lock:
            return self.walking_status == 0.0

    def _is_walking(self):
        with self.lock:
            return self.walking_status > 0.5

    def _get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _get_state(self):
        with self.lock:
            return self.state

    def _set_state(self, state, next_after_stop=None):
        with self.lock:
            self.state = state
            if next_after_stop is not None:
                self.next_after_stop = next_after_stop
        rospy.loginfo("[FSM] -> %s", state)

    def _enter_wait_for_stop(self, next_after_stop=None):
        with self.lock:
            if next_after_stop is not None:
                self.next_after_stop = next_after_stop
            self.wait_seen_walking = self.walking_status > 0.5
            self.state = self.STATE_WAIT_FOR_STOP
        rospy.loginfo("[FSM] -> STATE_WAIT_FOR_STOP (next=%s)", self.next_after_stop)

    def _safe_to_idle(self):
        with self.lock:
            self.state = self.STATE_IDLE
            self.next_after_stop = self.STATE_VISION_LOCK
            self.pending_sound_angle = None
            self.wait_seen_walking = False
            self.search_step_count = 0
        rospy.loginfo("[FSM] -> STATE_IDLE")

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def _normalize_angle(angle):
        if not math.isfinite(angle):
            return 0.0
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle


def main():
    rospy.init_node("module1_master")
    node = Module1Master()
    node.run()


if __name__ == "__main__":
    main()