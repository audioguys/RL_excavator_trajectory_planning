#! /usr/bin/env python


import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import os

def spawn_ball(model_name, model_pose):
    # Load the SDF file for the ball
    with open(os.path.join(os.path.dirname(__file__), '../urdf/ball.urdf'), 'r') as f:
        sdf = f.read()

    # Call the Gazebo SpawnModel service to spawn the ball
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    spawn_model(model_name, sdf, '', model_pose, 'ground_plane')

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('spawn_balls_1')
    i = 0
    # Spawn a large number of balls
    model_pose = Pose()
    for z in range(3):
        model_pose.position.z = 0.3 + 0.6 * z
        for y in range(8):
            model_pose.position.y = 1.6 + y * 0.7
            for x in range(8):
                model_pose.position.x = 4.8 + x * 0.7
                model_name = f'ball{i}'
                spawn_ball(model_name, model_pose)
                i += 1


    # Spin the ROS node to keep it running
    rospy.spin()


