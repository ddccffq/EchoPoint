#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String


def one_time_publisher():
    rospy.init_node('ros_find_node', anonymous=False)
    pub = rospy.Publisher('find_topic', String, queue_size=10)

    # 等待 publisher 就绪
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.loginfo("Waiting for subscribers...")
        rospy.sleep(0.5)

    # 发布消息
    msg = String()
    msg.data = "Hello"
    pub.publish(msg)
    rospy.loginfo("Published: %s", msg.data)

    # 保持节点运行，但不再发布新消息
    # 如果节点需要一直运行等待其他操作，可以用 spin()
    rospy.spin()  # 节点会一直运行，但不会再发布消息


if __name__ == '__main__':
    try:
        one_time_publisher()
    except rospy.ROSInterruptException:
        pass