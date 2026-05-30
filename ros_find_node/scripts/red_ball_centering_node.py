#!/usr/bin/env python3
# coding=utf-8

import json
import os
import threading
import time
from datetime import datetime
from urllib.request import urlopen

import rospy
from bodyhub.msg import JointControlPoint
from bodyhub.srv import SrvTLSstring
from ros_AIUI_node.srv import SrvWakeupMute, textToSpeakMultipleOptions
from ros_find_node.srv import StartFind, StartFindResponse
from ros_vision_node.srv import BallDetectInAreaSrv
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool


class RedBallCenteringNode(object):
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.vision_service = rospy.get_param(
            "~vision_service", "/ros_vision_node/ball_detection_in_area"
        )
        self.snapshot_url = rospy.get_param(
            "~snapshot_url",
            "http://localhost:8080/snapshot?topic=/camera/color/image_raw",
        )
        self.output_dir = os.path.expanduser(
            rospy.get_param("~output_dir", "~/red_ball_snapshots")
        )

        self.center_x = float(rospy.get_param("~center_x", 320.0))
        self.center_y = float(rospy.get_param("~center_y", 240.0))
        self.tolerance_x = float(rospy.get_param("~tolerance_x", 20.0))
        self.tolerance_y = float(rospy.get_param("~tolerance_y", 20.0))
        self.stable_frames = int(rospy.get_param("~stable_frames", 8))
        self.max_lost_frames = int(rospy.get_param("~max_lost_frames", 30))

        self.enable_search = bool(rospy.get_param("~enable_search", True))
        self.search_start_frames = int(rospy.get_param("~search_start_frames", 3))
        self.search_step_frames = int(rospy.get_param("~search_step_frames", 3))
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
        self.pan = float(rospy.get_param("~initial_pan", 0.0))
        self.tilt = float(rospy.get_param("~initial_tilt", 0.0))

        self.control_id = int(rospy.get_param("~control_id", 2))
        self.use_master_id_service = bool(rospy.get_param("~use_master_id_service", True))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.capture_on_centered = bool(rospy.get_param("~capture_on_centered", True))
        self.exit_after_capture = bool(rospy.get_param("~exit_after_capture", True))

        self.wait_for_start_service = bool(rospy.get_param("~wait_for_start_service", True))
        self.start_service = rospy.get_param("~start_service", "/ros_find_node/start")

        self.enable_voice_commands = bool(rospy.get_param("~enable_voice_commands", True))
        self.tts_service = rospy.get_param("~tts_service", "/aiui/text_to_speak_multiple_options")
        self.speech_request_service = rospy.get_param("~speech_request_service", "/aiui/wakeup_mute")
        self.speech_result_topic = rospy.get_param("~speech_result_topic", "/aiui/iat")
        self.stop_recording_service = rospy.get_param("~stop_recording_service", "/aiui/stop_recording")
        self.pause_head_sound_service = rospy.get_param(
            "~pause_head_toward_sound_service", "/aiui/pause_head_toward_sound"
        )
        self.finish_signal_topic = rospy.get_param(
            "~finish_signal_topic", "/ros_find_node/finish_signal"
        )
        self.finish_signal_text = rospy.get_param("~finish_signal_text", "start_recognition")
        self.enable_finish_topic = bool(rospy.get_param("~enable_finish_topic", False))
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

        self.head_pub = rospy.Publisher(
            "/MediumSize/BodyHub/HeadPosition", JointControlPoint, queue_size=10
        )
        self.finish_pub = None
        if self.enable_finish_topic:
            self.finish_pub = rospy.Publisher(self.finish_signal_topic, String, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        self.detect_ball = rospy.ServiceProxy(self.vision_service, BallDetectInAreaSrv)

        self.latest_image_msg = None
        self.captured_image_msg = None
        self.latest_snapshot_path = ""
        self.workflow_active = False
        self.start_event = threading.Event()
        if not self.wait_for_start_service:
            self.start_event.set()
        self.start_server = rospy.Service(self.start_service, StartFind, self.handle_start)

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def handle_start(self, request):
        if not request.start:
            return StartFindResponse(False, "start is false, ros_find_node ignored the request")
        if self.workflow_active:
            return StartFindResponse(False, "ros_find_node is already centering a red ball")
        self.start_event.set()
        return StartFindResponse(True, "ros_find_node started red ball centering")

    def wait_for_start(self):
        if not self.wait_for_start_service:
            return True

        rospy.loginfo(
            "Waiting for first-stage service call: rosservice call %s \"start: true\"",
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

    def publish_finish_signal(self):
        if not self.finish_pub:
            return
        msg = String()
        msg.data = self.finish_signal_text
        self.finish_pub.publish(msg)
        rospy.loginfo(
            "Published finish signal on %s: %s",
            self.finish_signal_topic,
            self.finish_signal_text,
        )

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
            rospy.logerr(
                "Cannot import ros_langchain_node/DescribeImage. "
                "Make sure your teammate's srv is built in this catkin workspace: %s",
                err,
            )
            return False

        try:
            rospy.loginfo("Waiting for describe image service: %s", self.describe_image_service)
            rospy.wait_for_service(self.describe_image_service, timeout=self.describe_service_timeout)
            client = rospy.ServiceProxy(self.describe_image_service, DescribeImage)
            request = DescribeImageRequest()
            request.image = self.captured_image_msg
            if hasattr(request, "speak"):
                request.speak = self.describe_image_speak

            response = client(request)
            success = bool(getattr(response, "success", False))
            error_msg = getattr(response, "error_msg", "")
            text = getattr(response, "text", "")
            if success:
                rospy.loginfo("Describe image service succeeded: %s", text)
            else:
                rospy.logwarn("Describe image service failed: %s", error_msg)
            return success
        except Exception as err:
            rospy.logerr("Describe image service call failed: %s", err)
            return False

    def search_for_ball(self, lost_count):
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
            "Searching red ball: lost_frames=%d, head=(%.2f, %.2f)",
            lost_count,
            self.pan,
            self.tilt,
        )
        self.publish_head()

    def get_ball_center(self):
        try:
            response = self.detect_ball(self.image_topic)
        except Exception as err:
            rospy.logwarn("Ball detection service failed: %s", err)
            return None

        if not response.is_successful:
            rospy.logwarn("Ball detection failed: %s", response.result_msg)
            return None

        try:
            result = json.loads(response.result_msg)
        except ValueError as err:
            rospy.logwarn("Invalid ball detection JSON: %s", err)
            return None

        center = result.get("ball_mask_center")
        if center is None or len(center) != 2:
            return None
        return float(center[0]), float(center[1])

    def capture_snapshot(self):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = "red_ball_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, filename)
        data = urlopen(self.snapshot_url, timeout=3.0).read()
        with open(path, "wb") as image_file:
            image_file.write(data)
        rospy.loginfo("Saved snapshot: %s", path)
        return path

    def run(self):
        rospy.loginfo("Waiting for ball detection service: %s", self.vision_service)
        rospy.wait_for_service(self.vision_service)

        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if not self.wait_for_start():
                return

            self.workflow_active = True
            self.captured_image_msg = None
            self.latest_snapshot_path = ""
            self.update_control_id()
            self.publish_head()

            stable_count = 0
            lost_count = 0

            while not rospy.is_shutdown() and self.workflow_active:
                center = self.get_ball_center()
                if center is None:
                    lost_count += 1
                    stable_count = 0
                    if lost_count % self.max_lost_frames == 0:
                        rospy.logwarn("Red ball not found. Check camera view and lighting.")
                    self.search_for_ball(lost_count)
                    rate.sleep()
                    continue

                lost_count = 0
                ball_x, ball_y = center
                error_x = self.center_x - ball_x
                error_y = self.center_y - ball_y

                rospy.loginfo(
                    "ball=(%.1f, %.1f), error=(%.1f, %.1f), head=(%.2f, %.2f)",
                    ball_x,
                    ball_y,
                    error_x,
                    error_y,
                    self.pan,
                    self.tilt,
                )

                centered = abs(error_x) <= self.tolerance_x and abs(error_y) <= self.tolerance_y
                if centered:
                    stable_count += 1
                    if stable_count >= self.stable_frames:
                        rospy.loginfo("Red ball centered.")
                        if self.capture_on_centered:
                            time.sleep(0.5)
                            try:
                                self.captured_image_msg = self.latest_image_msg
                                self.latest_snapshot_path = self.capture_snapshot()
                                self.speak(self.screenshot_done_text)
                            except Exception as err:
                                rospy.logerr("Snapshot failed: %s", err)

                        next_action = self.wait_for_next_action()
                        if next_action == "restart":
                            rospy.loginfo("Restarting red ball recognition.")
                            stable_count = 0
                            lost_count = 0
                            continue

                        self.publish_finish_signal()
                        self.call_describe_image()
                        self.workflow_active = False
                        if self.exit_after_capture and not self.wait_for_start_service:
                            return
                        break
                else:
                    stable_count = 0
                    if abs(error_x) > self.tolerance_x:
                        self.pan = self.clamp(
                            self.pan + self.pan_gain * error_x, self.pan_min, self.pan_max
                        )
                    if abs(error_y) > self.tolerance_y:
                        self.tilt = self.clamp(
                            self.tilt + self.tilt_gain * error_y,
                            self.tilt_min,
                            self.tilt_max,
                        )
                    self.publish_head()

                rate.sleep()


if __name__ == "__main__":
    rospy.init_node("ros_find_node")
    RedBallCenteringNode().run()
