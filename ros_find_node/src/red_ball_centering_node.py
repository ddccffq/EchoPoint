#!/usr/bin/env python3
# coding=utf-8

import json
import os
import time
from datetime import datetime
from urllib.request import urlopen

import rospy
from bodyhub.msg import JointControlPoint
from bodyhub.srv import SrvTLSstring
from ros_vision_node.srv import BallDetectInAreaSrv


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

        self.head_pub = rospy.Publisher(
            "/MediumSize/BodyHub/HeadPosition", JointControlPoint, queue_size=10
        )
        self.detect_ball = rospy.ServiceProxy(self.vision_service, BallDetectInAreaSrv)

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
        self.update_control_id()
        self.publish_head()

        stable_count = 0
        lost_count = 0
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            center = self.get_ball_center()
            if center is None:
                lost_count += 1
                stable_count = 0
                if lost_count % self.max_lost_frames == 0:
                    rospy.logwarn("Red ball not found. Check camera view and lighting.")
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
                            self.capture_snapshot()
                        except Exception as err:
                            rospy.logerr("Snapshot failed: %s", err)
                    if self.exit_after_capture:
                        return
                    stable_count = 0
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
    rospy.init_node("red_ball_centering_node")
    RedBallCenteringNode().run()
