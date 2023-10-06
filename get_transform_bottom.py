#! /usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":

    rospy.init_node('tf_listener_bottom', log_level=rospy.INFO)

    listener = tf.TransformListener()

    # create a publisher to publish the transform as a message
    publisher = rospy.Publisher('/transform_bottom', TransformStamped, queue_size=10)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('excavator/base_link', 'excavator/link_6', rospy.Time(0))
            # create a TransformStamped message
            transform_msg = TransformStamped()
            transform_msg.header.stamp = rospy.Time.now()
            transform_msg.header.frame_id = 'excavator/base_link'
            transform_msg.child_frame_id = 'excavator/link_6'
            transform_msg.transform.translation.x = trans[0]
            transform_msg.transform.translation.y = trans[1]
            transform_msg.transform.translation.z = trans[2]
            transform_msg.transform.rotation.x = rot[0]
            transform_msg.transform.rotation.y = rot[1]
            transform_msg.transform.rotation.z = rot[2]
            transform_msg.transform.rotation.w = rot[3]

            # publish the transform message
            publisher.publish(transform_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue



