#!/usr/bin/env python3
# coding=utf-8

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rospy
from bodyhub.msg import JointControlPoint
from bodyhub.srv import SrvTLSstring
from cv_bridge import CvBridge
from ros_AIUI_node.srv import SrvWakeupMute, textToSpeakMultipleOptions
from ros_find_node.srv import StartFind, StartFindResponse
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool


class PointingCenteringNode(object):
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.output_dir = os.path.expanduser(
            rospy.get_param("~output_dir", "~/pointing_snapshots")
        )
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", "/ros_find_node/pointing_debug_image"
        )

        self.center_x = float(rospy.get_param("~center_x", 320.0))
        self.center_y = float(rospy.get_param("~center_y", 240.0))
        self.tolerance_x = float(rospy.get_param("~tolerance_x", 30.0))
        self.tolerance_y = float(rospy.get_param("~tolerance_y", 30.0))
        self.stable_frames = int(rospy.get_param("~stable_frames", 5))
        self.max_lost_frames = int(rospy.get_param("~max_lost_frames", 30))

        self.min_hand_area = float(rospy.get_param("~min_hand_area", 2500.0))
        self.min_finger_distance = float(rospy.get_param("~min_finger_distance", 45.0))
        self.pointing_extension = float(rospy.get_param("~pointing_extension", 2.5))
        self.target_edge_margin = float(rospy.get_param("~target_edge_margin", 35.0))
        self.target_confirm_frames = int(rospy.get_param("~target_confirm_frames", 3))
        self.max_target_jump = float(rospy.get_param("~max_target_jump", 90.0))
        self.target_smoothing = float(rospy.get_param("~target_smoothing", 0.35))

        self.enable_search = bool(rospy.get_param("~enable_search", True))
        self.search_start_frames = int(rospy.get_param("~search_start_frames", 5))
        self.search_step_frames = int(rospy.get_param("~search_step_frames", 4))
        self.search_pan_min = float(rospy.get_param("~search_pan_min", -60.0))
        self.search_pan_max = float(rospy.get_param("~search_pan_max", 60.0))
        self.search_pan_step = abs(float(rospy.get_param("~search_pan_step", 8.0)))
        self.search_tilt_levels = rospy.get_param("~search_tilt_levels", [-10.0, 0.0, 10.0])
        if not self.search_tilt_levels:
            self.search_tilt_levels = [0.0]
        self.search_tilt_levels = [float(level) for level in self.search_tilt_levels]
        self.search_direction = 1.0
        self.search_tilt_index = 0

        self.pan_gain = float(rospy.get_param("~pan_gain", 0.01))
        self.tilt_gain = float(rospy.get_param("~tilt_gain", 0.01))
        self.pan_min = float(rospy.get_param("~pan_min", -90.0))
        self.pan_max = float(rospy.get_param("~pan_max", 90.0))
        self.tilt_min = float(rospy.get_param("~tilt_min", -25.0))
        self.tilt_max = float(rospy.get_param("~tilt_max", 25.0))
        self.max_head_step = float(rospy.get_param("~max_head_step", 2.0))
        self.pan = float(rospy.get_param("~initial_pan", 0.0))
        self.tilt = float(rospy.get_param("~initial_tilt", 0.0))

        self.control_id = int(rospy.get_param("~control_id", 2))
        self.use_master_id_service = bool(rospy.get_param("~use_master_id_service", True))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.capture_on_centered = bool(rospy.get_param("~capture_on_centered", True))
        self.exit_after_capture = bool(rospy.get_param("~exit_after_capture", False))

        self.wait_for_start_service = bool(rospy.get_param("~wait_for_start_service", True))
        self.start_service = rospy.get_param("~start_service", "/ros_find_node/start_pointing")

        self.enable_voice_commands = bool(rospy.get_param("~enable_voice_commands", True))
        self.tts_service = rospy.get_param("~tts_service", "/aiui/text_to_speak_multiple_options")
        self.speech_request_service = rospy.get_param("~speech_request_service", "/aiui/wakeup_mute")
        self.speech_result_topic = rospy.get_param("~speech_result_topic", "/aiui/iat")
        self.stop_recording_service = rospy.get_param("~stop_recording_service", "/aiui/stop_recording")
        self.pause_head_sound_service = rospy.get_param(
            "~pause_head_toward_sound_service", "/aiui/pause_head_toward_sound"
        )
        self.describe_image_service = rospy.get_param(
            "~describe_image_service", "/langchain/describe_image"
        )
        self.call_describe_service = bool(rospy.get_param("~call_describe_service", True))
        self.describe_service_timeout = float(rospy.get_param("~describe_service_timeout", 5.0))
        self.describe_image_speak = bool(rospy.get_param("~describe_image_speak", True))
        self.speech_timeout = float(rospy.get_param("~speech_timeout", 8.0))
        self.screenshot_done_text = rospy.get_param("~screenshot_done_text", "截图完毕")
        self.restart_command = rospy.get_param("~restart_command", "重新截图")
        self.finish_command = rospy.get_param("~finish_command", "开始识别")
        self.tts_vcn = rospy.get_param("~tts_vcn", "qige")
        self.tts_speed = int(rospy.get_param("~tts_speed", 50))
        self.tts_pitch = int(rospy.get_param("~tts_pitch", 50))
        self.tts_volume = int(rospy.get_param("~tts_volume", 50))

        self.bridge = CvBridge()
        self.latest_image_msg = None
        self.latest_frame = None
        self.latest_debug_frame = None
        self.captured_image_msg = None
        self.filtered_target = None
        self.pending_target = None
        self.pending_target_count = 0
        self.workflow_active = False
        self.start_event = threading.Event()
        if not self.wait_for_start_service:
            self.start_event.set()

        self.head_pub = rospy.Publisher(
            "/MediumSize/BodyHub/HeadPosition", JointControlPoint, queue_size=10
        )
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        self.start_server = rospy.Service(self.start_service, StartFind, self.handle_start)

    def image_callback(self, msg):
        self.latest_image_msg = msg
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as err:
            rospy.logwarn("Could not convert image: %s", err)

    def handle_start(self, request):
        if not request.start:
            return StartFindResponse(False, "start is false, pointing node ignored the request")
        if self.workflow_active:
            return StartFindResponse(False, "pointing node is already running")
        self.start_event.set()
        return StartFindResponse(True, "pointing centering started")

    def wait_for_start(self):
        if not self.wait_for_start_service:
            return True

        rospy.loginfo(
            "Waiting for pointing start service: rosservice call %s \"start: true\"",
            self.start_service,
        )
        while not rospy.is_shutdown():
            if self.start_event.wait(0.2):
                self.start_event.clear()
                return True
        return False

    def update_control_id(self):
        if not self.use_master_id_service:
            return
        try:
            rospy.wait_for_service("/MediumSize/BodyHub/GetMasterID", timeout=2.0)
            client = rospy.ServiceProxy("/MediumSize/BodyHub/GetMasterID", SrvTLSstring)
            response = client("get")
            if response.data != 0:
                self.control_id = int(response.data)
        except Exception as err:
            rospy.logwarn("Could not update BodyHub control id: %s", err)

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def clamp_step(self, value, max_step):
        return self.clamp(value, -max_step, max_step)

    def publish_head(self):
        msg = JointControlPoint()
        msg.positions = [self.pan, -self.tilt]
        msg.mainControlID = self.control_id
        self.head_pub.publish(msg)

    def normalize_command(self, text):
        for punctuation in ["，", "。", "？", "！", "：", "“", "”", "《", "》", "；", "、", " ", "\t", "\n"]:
            text = text.replace(punctuation, "")
        return text

    def set_head_sound_paused(self, is_paused):
        try:
            rospy.wait_for_service(self.pause_head_sound_service, timeout=1.0)
            rospy.ServiceProxy(self.pause_head_sound_service, SetBool)(is_paused)
        except Exception as err:
            rospy.logdebug("Could not change head-toward-sound status: %s", err)

    def speak(self, text):
        if not self.enable_voice_commands or not text:
            return
        try:
            rospy.wait_for_service(self.tts_service, timeout=2.0)
            client = rospy.ServiceProxy(self.tts_service, textToSpeakMultipleOptions)
            client(text, self.tts_vcn, self.tts_speed, self.tts_pitch, self.tts_volume)
        except Exception as err:
            rospy.logwarn("Text-to-speech failed: %s", err)

    def listen_once(self):
        self.set_head_sound_paused(True)
        try:
            rospy.wait_for_service(self.speech_request_service, timeout=2.0)
            rospy.ServiceProxy(self.speech_request_service, SrvWakeupMute)(False)
            msg = rospy.wait_for_message(
                self.speech_result_topic, String, timeout=self.speech_timeout
            )
            return self.normalize_command(msg.data)
        except Exception as err:
            rospy.logwarn("Voice command timeout or failed: %s", err)
            try:
                rospy.wait_for_service(self.stop_recording_service, timeout=1.0)
                rospy.ServiceProxy(self.stop_recording_service, Empty)()
            except Exception:
                pass
            return ""
        finally:
            self.set_head_sound_paused(False)

    def wait_for_next_action(self):
        if not self.enable_voice_commands:
            return "finish" if self.exit_after_capture else "restart"

        while not rospy.is_shutdown():
            rospy.loginfo(
                "Waiting for voice command: %s / %s",
                self.restart_command,
                self.finish_command,
            )
            command = self.listen_once()
            if not command:
                continue
            rospy.loginfo("Voice command: %s", command)
            if self.restart_command in command:
                return "restart"
            if self.finish_command in command:
                return "finish"
            rospy.loginfo("Ignored voice command: %s", command)
        return "finish"

    def skin_mask(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect_pointing_target(self, frame):
        mask = self.skin_mask(frame)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = frame.copy()
        if not contours:
            self.publish_debug(debug)
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_hand_area:
            self.publish_debug(debug)
            return None

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            self.publish_debug(debug)
            return None
        palm_x = float(moments["m10"] / moments["m00"])
        palm_y = float(moments["m01"] / moments["m00"])

        hull = cv2.convexHull(contour, returnPoints=True)
        points = hull.reshape(-1, 2).astype(np.float32)
        palm = np.array([palm_x, palm_y], dtype=np.float32)
        distances = np.linalg.norm(points - palm, axis=1)
        tip = points[int(np.argmax(distances))]
        tip_distance = float(np.max(distances))
        if tip_distance < self.min_finger_distance:
            self.publish_debug(debug)
            return None

        direction = tip - palm
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            self.publish_debug(debug)
            return None
        direction = direction / norm

        height, width = frame.shape[:2]
        target = tip + direction * tip_distance * self.pointing_extension
        target_x = self.clamp(float(target[0]), 0.0, float(width - 1))
        target_y = self.clamp(float(target[1]), 0.0, float(height - 1))

        cv2.drawContours(debug, [contour], -1, (0, 255, 255), 2)
        cv2.circle(debug, (int(palm_x), int(palm_y)), 6, (255, 0, 0), -1)
        cv2.circle(debug, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
        cv2.line(
            debug,
            (int(palm_x), int(palm_y)),
            (int(target_x), int(target_y)),
            (0, 255, 0),
            2,
        )
        cv2.drawMarker(
            debug,
            (int(target_x), int(target_y)),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        if self.is_edge_target(target_x, target_y, width, height):
            cv2.putText(
                debug,
                "rejected: edge target",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            self.publish_debug(debug)
            return None
        self.publish_debug(debug)
        return target_x, target_y

    def is_edge_target(self, target_x, target_y, width, height):
        return (
            target_x <= self.target_edge_margin
            or target_x >= float(width - 1) - self.target_edge_margin
            or target_y <= self.target_edge_margin
            or target_y >= float(height - 1) - self.target_edge_margin
        )

    def stabilize_target(self, target):
        target_vec = np.array(target, dtype=np.float32)
        if self.filtered_target is not None:
            jump = float(np.linalg.norm(target_vec - self.filtered_target))
            if jump > self.max_target_jump:
                rospy.logwarn(
                    "Rejected unstable pointing target: target=(%.1f, %.1f), jump=%.1f",
                    target[0],
                    target[1],
                    jump,
                )
                self.pending_target = target_vec
                self.pending_target_count = 1
                return None
            alpha = self.clamp(self.target_smoothing, 0.0, 1.0)
            self.filtered_target = alpha * target_vec + (1.0 - alpha) * self.filtered_target
            return float(self.filtered_target[0]), float(self.filtered_target[1])

        if self.pending_target is None:
            self.pending_target = target_vec
            self.pending_target_count = 1
            return None

        jump = float(np.linalg.norm(target_vec - self.pending_target))
        if jump <= self.max_target_jump:
            self.pending_target_count += 1
            self.pending_target = (
                target_vec + self.pending_target * float(self.pending_target_count - 1)
            ) / float(self.pending_target_count)
        else:
            self.pending_target = target_vec
            self.pending_target_count = 1
            return None

        if self.pending_target_count < self.target_confirm_frames:
            return None

        self.filtered_target = self.pending_target
        return float(self.filtered_target[0]), float(self.filtered_target[1])

    def publish_debug(self, debug):
        self.latest_debug_frame = debug
        if self.debug_pub.get_num_connections() == 0:
            return
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except Exception as err:
            rospy.logdebug("Could not publish debug image: %s", err)

    def search_for_pointer(self, lost_count):
        if not self.enable_search or lost_count < self.search_start_frames:
            return
        if (lost_count - self.search_start_frames) % self.search_step_frames != 0:
            return

        next_pan = self.pan + self.search_direction * self.search_pan_step
        if next_pan > self.search_pan_max:
            next_pan = self.search_pan_max
            self.search_direction = -1.0
            self.search_tilt_index = (self.search_tilt_index + 1) % len(self.search_tilt_levels)
        elif next_pan < self.search_pan_min:
            next_pan = self.search_pan_min
            self.search_direction = 1.0
            self.search_tilt_index = (self.search_tilt_index + 1) % len(self.search_tilt_levels)

        self.pan = self.clamp(next_pan, self.pan_min, self.pan_max)
        self.tilt = self.clamp(
            self.search_tilt_levels[self.search_tilt_index], self.tilt_min, self.tilt_max
        )
        rospy.loginfo(
            "Searching pointing hand: lost_frames=%d, head=(%.2f, %.2f)",
            lost_count,
            self.pan,
            self.tilt,
        )
        self.publish_head()

    def capture_snapshot(self):
        if self.latest_frame is None:
            raise RuntimeError("No image frame available for snapshot")
        os.makedirs(self.output_dir, exist_ok=True)
        filename = "pointing_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, self.latest_frame)
        if self.latest_debug_frame is not None:
            debug_path = os.path.join(
                self.output_dir,
                "pointing_debug_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            cv2.imwrite(debug_path, self.latest_debug_frame)
        rospy.loginfo("Saved pointing snapshot: %s", path)
        return path

    def call_describe_image(self):
        if not self.call_describe_service:
            rospy.loginfo("Describe image service call is disabled.")
            return True
        if self.captured_image_msg is None:
            rospy.logerr("No captured image message is available for describe image service.")
            return False

        try:
            from ros_langchain_node.srv import DescribeImage, DescribeImageRequest
        except ImportError as err:
            rospy.logerr("Cannot import ros_langchain_node/DescribeImage: %s", err)
            return False

        try:
            rospy.loginfo("Waiting for describe image service: %s", self.describe_image_service)
            rospy.wait_for_service(self.describe_image_service, timeout=self.describe_service_timeout)
            client = rospy.ServiceProxy(self.describe_image_service, DescribeImage)
            request = DescribeImageRequest()
            request.image = self.captured_image_msg
            request.speak = self.describe_image_speak
            response = client(request)
            if response.success:
                rospy.loginfo("Describe image service succeeded: %s", response.text)
            else:
                rospy.logwarn("Describe image service failed: %s", response.error_msg)
            return bool(response.success)
        except Exception as err:
            rospy.logerr("Describe image service call failed: %s", err)
            return False

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if not self.wait_for_start():
                return

            self.workflow_active = True
            self.captured_image_msg = None
            self.filtered_target = None
            self.pending_target = None
            self.pending_target_count = 0
            self.update_control_id()
            self.publish_head()
            stable_count = 0
            lost_count = 0

            while not rospy.is_shutdown() and self.workflow_active:
                if self.latest_frame is None:
                    rate.sleep()
                    continue

                target = self.detect_pointing_target(self.latest_frame)
                if target is None:
                    lost_count += 1
                    stable_count = 0
                    if lost_count % self.max_lost_frames == 0:
                        rospy.logwarn("Pointing hand not found. Check lighting and background.")
                    self.search_for_pointer(lost_count)
                    rate.sleep()
                    continue

                target = self.stabilize_target(target)
                if target is None:
                    stable_count = 0
                    rate.sleep()
                    continue

                lost_count = 0
                target_x, target_y = target
                error_x = self.center_x - target_x
                error_y = self.center_y - target_y

                rospy.loginfo(
                    "point_target=(%.1f, %.1f), error=(%.1f, %.1f), head=(%.2f, %.2f)",
                    target_x,
                    target_y,
                    error_x,
                    error_y,
                    self.pan,
                    self.tilt,
                )

                centered = abs(error_x) <= self.tolerance_x and abs(error_y) <= self.tolerance_y
                if centered:
                    stable_count += 1
                    if stable_count >= self.stable_frames:
                        rospy.loginfo("Pointing target centered.")
                        if self.capture_on_centered:
                            time.sleep(0.5)
                            try:
                                self.captured_image_msg = self.latest_image_msg
                                self.capture_snapshot()
                                self.speak(self.screenshot_done_text)
                            except Exception as err:
                                rospy.logerr("Snapshot failed: %s", err)

                        next_action = self.wait_for_next_action()
                        if next_action == "restart":
                            rospy.loginfo("Restarting pointing recognition.")
                            stable_count = 0
                            lost_count = 0
                            continue

                        self.call_describe_image()
                        self.workflow_active = False
                        if self.exit_after_capture and not self.wait_for_start_service:
                            return
                        break
                else:
                    stable_count = 0
                    if abs(error_x) > self.tolerance_x:
                        self.pan = self.clamp(
                            self.pan
                            + self.clamp_step(self.pan_gain * error_x, self.max_head_step),
                            self.pan_min,
                            self.pan_max,
                        )
                    if abs(error_y) > self.tolerance_y:
                        self.tilt = self.clamp(
                            self.tilt
                            + self.clamp_step(self.tilt_gain * error_y, self.max_head_step),
                            self.tilt_min,
                            self.tilt_max,
                        )
                    self.publish_head()

                rate.sleep()


if __name__ == "__main__":
    rospy.init_node("pointing_centering_node")
    PointingCenteringNode().run()
