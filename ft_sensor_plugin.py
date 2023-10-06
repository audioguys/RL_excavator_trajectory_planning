#! /usr/bin/env python


import rospy
from geometry_msgs.msg import WrenchStamped

def ft_sensor_plugin():
    rospy.init_node('ft_sensor_plugin')

    # Create a publisher for the FT sensor topic
    pub = rospy.Publisher('ft_sensor_topic', WrenchStamped, queue_size=10)

    # Create a rate to control the publishing frequency
    rate = rospy.Rate(120.0)

    while not rospy.is_shutdown():
        # Create a WrenchStamped message
        wrench = WrenchStamped()
        # Set the values for force and torque components
        # ... Add your code here ...

        # Publish the WrenchStamped message
        pub.publish(wrench)

        # Sleep to maintain the desired publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        ft_sensor_plugin()
    except rospy.ROSInterruptException:
        pass


