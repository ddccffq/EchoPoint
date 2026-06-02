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
from ros_AIUI_node.srv import textToSpeakMultipleOptions
from ros_find_node.srv import StartFind, StartFindResponse
from ros_vision_node.srv import BallDetectInAreaSrv
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float64MultiArray

try:
    from motion.bodyhub_client import BodyhubClient
    _BODYHUB_CLIENT_AVAILABLE = True
except ImportError:
    BodyhubClient = None
    _BODYHUB_CLIENT_AVAILABLE = False


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
        self.enable_body_search = bool(rospy.get_param("~enable_body_search", True))
        self.body_search_step_degrees = abs(
            float(rospy.get_param("~body_search_step_degrees", 10.0))
        )
        self.body_search_total_degrees = abs(
            float(rospy.get_param("~body_search_total_degrees", 360.0))
        )
        self.body_search_head_pans = rospy.get_param(
            "~body_search_head_pans", [-45.0, 0.0, 45.0]
        )
        if not self.body_search_head_pans:
            self.body_search_head_pans = [0.0]
        self.body_search_head_pans = [float(pan) for pan in self.body_search_head_pans]
        self.body_search_head_index = 0
        self.body_search_turned_degrees = 0.0

        self.pan_gain = float(rospy.get_param("~pan_gain", 0.01))
        self.tilt_gain = float(rospy.get_param("~tilt_gain", 0.01))
        self.pan_min = float(rospy.get_param("~pan_min", -90.0))
        self.pan_max = float(rospy.get_param("~pan_max", 90.0))
        self.tilt_min = float(rospy.get_param("~tilt_min", -25.0))
        self.tilt_max = float(rospy.get_param("~tilt_max", 25.0))
        self.pan = float(rospy.get_param("~initial_pan", 0.0))
        self.tilt = float(rospy.get_param("~initial_tilt", 0.0))

        self.control_id = int(rospy.get_param("~control_id", 2))
        self.body_control_id = int(rospy.get_param("~body_control_id", 30))
        self.use_master_id_service = bool(rospy.get_param("~use_master_id_service", True))
        self.enable_bodyhub_walk_setup = bool(
            rospy.get_param("~enable_bodyhub_walk_setup", False)
        )
        self.require_walking_for_body_search = bool(
            rospy.get_param("~require_walking_for_body_search", True)
        )
        self.walking_status_topic = rospy.get_param(
            "~walking_status_topic", "/MediumSize/BodyHub/WalkingStatus"
        )
        self.gait_command_topic = rospy.get_param("~gait_command_topic", "/gaitCommand")
        self.request_gait_topic = rospy.get_param(
            "~request_gait_topic", "/requestGaitCommand"
        )
        self.gait_handshake_timeout = float(rospy.get_param("~gait_handshake_timeout", 3.0))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.capture_on_centered = bool(rospy.get_param("~capture_on_centered", True))
        self.exit_after_capture = bool(rospy.get_param("~exit_after_capture", True))

        self.wait_for_start_service = bool(rospy.get_param("~wait_for_start_service", True))
        self.start_service = rospy.get_param("~start_service", "/ros_find_node/start")

        self.enable_tts = bool(
            rospy.get_param("~enable_tts", rospy.get_param("~enable_voice_commands", True))
        )
        self.tts_service = rospy.get_param("~tts_service", "/aiui/text_to_speak_multiple_options")
        self.describe_image_service = rospy.get_param(
            "~describe_image_service", "/langchain/describe_image"
        )
        self.call_describe_service = bool(rospy.get_param("~call_describe_service", True))
        self.describe_service_timeout = float(rospy.get_param("~describe_service_timeout", 5.0))
        self.describe_image_speak = bool(rospy.get_param("~describe_image_speak", True))
        self.screenshot_done_text = rospy.get_param("~screenshot_done_text", "截图完毕")
        self.tts_vcn = rospy.get_param("~tts_vcn", "qige")
        self.tts_speed = int(rospy.get_param("~tts_speed", 50))
        self.tts_pitch = int(rospy.get_param("~tts_pitch", 50))
        self.tts_volume = int(rospy.get_param("~tts_volume", 50))

        self.head_pub = rospy.Publisher(
            "/MediumSize/BodyHub/HeadPosition", JointControlPoint, queue_size=10
        )
        self.gait_pub = rospy.Publisher(
            self.gait_command_topic, Float64MultiArray, queue_size=2
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        rospy.Subscriber(
            self.walking_status_topic, Float64, self.walking_status_callback, queue_size=5
        )
        self.detect_ball = rospy.ServiceProxy(self.vision_service, BallDetectInAreaSrv)

        self.latest_image_msg = None
        self.walking_status = 0.0
        self.captured_image_msg = None
        self.latest_snapshot_path = ""
        self.workflow_active = False
        self.bodyhub_client = None
        self.bodyhub_ready = False
        if _BODYHUB_CLIENT_AVAILABLE:
            try:
                self.bodyhub_client = BodyhubClient(self.body_control_id)
                rospy.loginfo("BodyhubClient created with control_id=%d", self.body_control_id)
            except Exception as err:
                rospy.logwarn("Could not create BodyhubClient: %s", err)
        else:
            rospy.logwarn("motion.bodyhub_client is unavailable; body search may not move.")
        self.start_event = threading.Event()
        if not self.wait_for_start_service:
            self.start_event.set()
        self.start_server = rospy.Service(self.start_service, StartFind, self.handle_start)

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def walking_status_callback(self, msg):
        self.walking_status = float(msg.data)

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

    def bodyhub_walk(self):
        if not self.enable_bodyhub_walk_setup:
            rospy.loginfo(
                "BodyHub walk setup disabled in find node; expecting locate node to keep walking state."
            )
            return True
        if self.bodyhub_client is None:
            rospy.logwarn_throttle(
                5.0,
                "BodyhubClient not available; publishing gait without BodyHub walk state.",
            )
            return True

        try:
            result = self.bodyhub_client.reset()
            if result is not True:
                result = self.bodyhub_client.reset(root=True)
                if result is not True:
                    rospy.logwarn("BodyHub reset failed (result=%s), forcing reset", result)
                    self.bodyhub_client.reset(root=True)

            result = self.bodyhub_client.ready()
            if result is not True:
                rospy.logwarn("BodyHub ready() failed (result=%s)", result)
                return False

            result = self.bodyhub_client.walk()
            if result is not True:
                rospy.logwarn("BodyHub walk() failed (result=%s)", result)
                return False

            self.bodyhub_ready = True
            rospy.loginfo("BodyHub state machine: walking for red ball search")
            return True
        except Exception as err:
            rospy.logwarn("BodyHub walk state setup failed: %s", err)
            return False

    def bodyhub_reset(self, force=False):
        if self.bodyhub_client is None:
            return
        try:
            self.bodyhub_client.reset(root=force)
            self.bodyhub_ready = False
            rospy.loginfo("BodyHub reset after red ball search (force=%s)", force)
        except Exception as err:
            rospy.logwarn("BodyHub reset exception: %s", err)

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def publish_head(self):
        msg = JointControlPoint()
        msg.positions = [self.pan, -self.tilt]
        msg.mainControlID = self.control_id
        self.head_pub.publish(msg)

    def is_walking(self):
        return self.walking_status > 0.5

    def publish_body_turn(self):
        if self.body_search_step_degrees <= 0.0:
            return False
        if self.require_walking_for_body_search and not self.is_walking():
            rospy.logwarn_throttle(
                3.0,
                "Skipping body search turn because WalkingStatus is not walking. "
                "Start from locate_node or set require_walking_for_body_search:=false.",
            )
            return False
        if self.body_search_turned_degrees >= self.body_search_total_degrees:
            rospy.logwarn_throttle(
                5.0,
                "Full %.1f degree red ball body search completed.",
                self.body_search_total_degrees,
            )
            self.body_search_turned_degrees = 0.0

        try:
            rospy.wait_for_message(
                self.request_gait_topic,
                Bool,
                timeout=self.gait_handshake_timeout,
            )
            rospy.loginfo("requestGaitCommand handshake received for red ball search")
        except rospy.ROSException:
            rospy.logwarn(
                "requestGaitCommand timeout (%.1fs), publishing body search turn anyway",
                self.gait_handshake_timeout,
            )

        turn_step = min(
            self.body_search_step_degrees,
            max(self.body_search_total_degrees - self.body_search_turned_degrees, 0.0),
        )
        if turn_step <= 0.0:
            turn_step = self.body_search_step_degrees
        self.gait_pub.publish(Float64MultiArray(data=[0.0, 0.0, turn_step]))
        self.body_search_turned_degrees += abs(turn_step)
        rospy.loginfo(
            "Published red ball body search turn: %.1f deg (%.1f/%.1f)",
            turn_step,
            self.body_search_turned_degrees,
            self.body_search_total_degrees,
        )
        return True

    def speak(self, text):
        if not self.enable_tts or not text:
            return
        try:
            rospy.wait_for_service(self.tts_service, timeout=2.0)
            client = rospy.ServiceProxy(self.tts_service, textToSpeakMultipleOptions)
            client(text, self.tts_vcn, self.tts_speed, self.tts_pitch, self.tts_volume)
        except Exception as err:
            rospy.logwarn("Text-to-speech failed: %s", err)

    def call_describe_image(self):
        if not self.call_describe_service:
            rospy.loginfo("Describe image service call is disabled.")
            return True
        if self.captured_image_msg is None and not self.latest_snapshot_path:
            rospy.logerr("No captured image is available for describe image service.")
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
            if self.captured_image_msg is not None:
                request.image = self.captured_image_msg
            if hasattr(request, "image_path"):
                request.image_path = self.latest_snapshot_path
            if hasattr(request, "image_mime"):
                request.image_mime = "image/jpeg"
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
        if self.enable_body_search:
            if self.is_walking():
                rospy.loginfo_throttle(2.0, "Body is turning during red ball search.")
                return

            pans_count = len(self.body_search_head_pans)
            tilts_count = len(self.search_tilt_levels)
            total_head_steps = max(pans_count * tilts_count, 1)
            pan = self.body_search_head_pans[self.body_search_head_index % pans_count]
            tilt = self.search_tilt_levels[
                (self.body_search_head_index // pans_count) % tilts_count
            ]
            self.pan = self.clamp(pan, self.pan_min, self.pan_max)
            self.tilt = self.clamp(tilt, self.tilt_min, self.tilt_max)
            self.publish_head()
            rospy.loginfo(
                "Searching red ball with head: lost_frames=%d, head=(%.2f, %.2f), "
                "body_search=%.1f/%.1f",
                lost_count,
                self.pan,
                self.tilt,
                self.body_search_turned_degrees,
                self.body_search_total_degrees,
            )

            self.body_search_head_index += 1
            if self.body_search_head_index >= total_head_steps:
                self.body_search_head_index = 0
                self.publish_body_turn()
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
            self.body_search_head_index = 0
            self.body_search_turned_degrees = 0.0
            self.bodyhub_walk()
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
                self.body_search_head_index = 0
                self.body_search_turned_degrees = 0.0
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

                        self.call_describe_image()
                        self.workflow_active = False
                        self.bodyhub_reset()
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
