#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ros_langchain_node.srv import DescribeImage, DescribeImageResponse
from ros_AIUI_node.srv import textToSpeak, textToSpeakRequest

class LangchainNode:
    def __init__(self):
        rospy.init_node("langchain_node")
        self.bridge = CvBridge()

        rospy.wait_for_service('/aiui/text_to_speak')
        self.tts_cli = rospy.ServiceProxy('/aiui/text_to_speak', textToSpeak)

        self.server = rospy.Service('/langchain/describe_image', DescribeImage, self.handle_describe)
        rospy.loginfo("Service ready: /langchain/describe_image")

    def call_agent_with_image(self, cv_img):
        # TODO: 这里替换成你真实的大模型调用
        # 例如：text = self.agent.invoke(cv_img)
        return "我看到了一个人站在桌子旁边。"

    def speak(self, text):
        req = textToSpeakRequest(text=text)
        resp = self.tts_cli(req)
        return resp.is_success

    def handle_describe(self, req):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='bgr8')
            text = self.call_agent_with_image(cv_img)

            if req.speak and text.strip():
                ok = self.speak(text)
                if not ok:
                    rospy.logwarn("TTS调用失败")

            return DescribeImageResponse(True, text, "")
        except Exception as e:
            return DescribeImageResponse(False, "", str(e))

if __name__ == '__main__':
    LangchainNode()
    rospy.spin()
