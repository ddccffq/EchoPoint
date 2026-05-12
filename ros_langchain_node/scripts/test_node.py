#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class TestNode:
    def __init__(self):
        rospy.init_node('test_node', anonymous=True)
        self.pub = rospy.Publisher('/test_topic', String, queue_size=10)
        rospy.Subscriber('/test_response', String, self.callback)
        rospy.loginfo("✅ 测试节点启动")
    
    def callback(self, msg):
        rospy.loginfo("📨 收到响应: %s", msg.data)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = TestNode()
    rospy.sleep(1)
    node.run()
