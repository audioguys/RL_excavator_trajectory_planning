import gym
import rospy
import time
import numpy as np
import math
import copy
import time
import os
from gym import utils, spaces
from gym.spaces import Box
import numpy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose



from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import WrenchStamped


reg = register(
    id='excavator-v2',
    entry_point='excavator_env:ExcavatorEnv',
    max_episode_steps=256,
    )

class ExcavatorEnv(gym.Env):

    # metadata = {'render.modes': ['human']}

    def __init__(self):
        

        # number_actions = rospy.get_param('/excavator/n_actions')

        # Define continue actions space
        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (4,),
            dtype = np.float32
        )

        # Define continue observation space
        self.observation_space = spaces.Box(
            low = -10,
            high = 10,
            shape = (6,),
            dtype = np.float32
        )


        self._seed()
        
        #get configuration parameters
        self.init_roll_vel = rospy.get_param('/excavator/init_roll_vel')

        #store the reward
        self.res = []
        self.result = 0
        # self.num_step = 0
        '''
        # Actions
        self.link_1_clockwise_value = rospy.get_param('/excavator/link_1_clockwise_value')
        self.link_1_conterclockwise_value = rospy.get_param('/excavator/link_1_conterclockwise_value')
        self.link_2_raise_value = rospy.get_param('/excavator/link_2_raise_value')
        self.link_2_drop_value = rospy.get_param('/excavator/link_2_drop_value')
        self.link_3_raise_value = rospy.get_param('/excavator/link_3_raise_value')
        self.link_3_drop_value = rospy.get_param('/excavator/link_3_drop_value')
        self.link_4_dig_value = rospy.get_param('/excavator/link_4_dig_value')
        self.link_4_drop_value = rospy.get_param('/excavator/link_4_drop_value')
        '''

        self.desired_joint_1_position_digging = rospy.get_param("/excavator/desired_joint_1_position_digging")
        self.desired_joint_1_position_unloading = rospy.get_param("/excavator/desired_joint_1_position_unloading")

        self.dig_point = Point()
        self.dig_point.x = rospy.get_param("/excavator/init_dig_pose/x")
        self.dig_point.y = rospy.get_param("/excavator/init_dig_pose/y")
        self.dig_point.z = rospy.get_param("/excavator/init_dig_pose/z")

        # initialize the velocity of the previous joint_state
        self.joint_1_preVelocity = 0.0
        self.joint_2_preVelocity = 0.0
        self.joint_3_preVelocity = 0.0
        self.joint_4_preVelocity = 0.0

        # initialize of the velocity of the current joint_state
        self.joint_1_velocity = 0.0
        self.joint_2_velocity = 0.0
        self.joint_3_velocity = 0.0
        self.joint_4_velocity = 0.0

        # Record the obs
        self.x_pre = []
        self.x_dig = []
        self.x_unload = []
        self.x_bot1 = []
        self.x_end = []
        self.joint1 = []
        self.joint2 = []
        self.joint3 = []
        self.joint4 = []

        # Done
        self.check_done_interval = rospy.get_param('/excavator/check_done_interval')
        self.check_dig_depth = rospy.get_param("/excavator/check_dig_depth")

        # Rewards
        self.end_episode_points = rospy.get_param("/excavator/end_episode_points")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        self.controllers_list = ['joint_state_controller',
                                 'joint_1_controller',
                                 'joint_2_controller',
                                 'joint_3_controller',
                                 'joint_4_controller',
                                ]
        self.controllers_object = ControllersConnection(namespace="excavator",
                                                        controllers_list=self.controllers_list)

        # initialize all suscirbers and publishers
        # rospy.init_node('ft_sensor_node')
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()

        rospy.Subscriber("/excavator/joint_states", JointState, self.joints_callback)
        rospy.Subscriber("/excavator/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/transform_target", TransformStamped, self.transformTarget_callback)
        rospy.Subscriber("/transform_bottom", TransformStamped, self.transformBottom_callback)
        rospy.Subscriber("/ft_sensor_topic", WrenchStamped, self.torque_callback)
        
        
        self._joint_1_velocity_pub = rospy.Publisher('/excavator/joint_1_controller/command', Float64, queue_size=1)
        self._joint_2_velocity_pub = rospy.Publisher('/excavator/joint_2_controller/command', Float64, queue_size=1)
        self._joint_3_velocity_pub = rospy.Publisher('/excavator/joint_3_controller/command', Float64, queue_size=1)
        self._joint_4_velocity_pub = rospy.Publisher('/excavator/joint_4_controller/command', Float64, queue_size=1)
        
        self.check_publishers_connection()
        
        '''
        with open('/home/xiaofei/catkin_ws/src/excavator_description/urdf/ball.urdf', 'r') as f:
            sdf = f.read()
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        for i in range(100):
            model_name = f'ball{i}'
            model_pose = Pose()
            model_pose.position.x = 5 + i % 5
            model_pose.position.y = 2 + i // 5
            model_pose.position.z = 1
            spawn_model(model_name, sdf, '', model_pose, 'ground_plane')
        '''

        self.gazebo.pauseSim()


    def _seed(self, seed=None): #overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):# overriden function

        self.gazebo.unpauseSim()
        self.set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_observation()
        done = self.is_done(obs)
        info = {}
        reward = self.compute_reward(obs, done)
        simplified_obs = self.convert_obs_to_state(obs)
        self.joint_1_preVelocity = self.joint_1_velocity
        self.joint_2_preVelocity = self.joint_2_velocity
        self.joint_3_preVelocity = self.joint_3_velocity
        self.joint_4_preVelocity = self.joint_4_velocity

        return simplified_obs, reward, done, info
        
    
    def reset(self):

        # reset the records
        self.x_pre = []
        self.x_dig = []
        self.x_unload = []
        self.x_bot1 = []
        self.x_end = []
        self.joint1 = []
        self.joint2 = []
        self.joint3 = []
        self.joint4 = []

        self.sum_reward = 0
        self.rewards = []

        # reset the records
        self.act1 = []
        self.act2 = []
        self.act3 = []
        self.act4 = []

        # reset the velocity of the previous joint_state
        self.joint_1_preVelocity = 0.0
        self.joint_2_preVelocity = 0.0
        self.joint_3_preVelocity = 0.0
        self.joint_4_preVelocity = 0.0

        # reset of the velocity of the current joint_state
        self.joint_1_velocity = 0.0
        self.joint_2_velocity = 0.0
        self.joint_3_velocity = 0.0
        self.joint_4_velocity = 0.0

        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()
       # self.set_init_pose()
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()

        self.gazebo.pauseSim()
        # self.init_env_variables()
        obs = self._get_observation()
        simplified_obs = self.convert_obs_to_state(obs)
       # print('heihei')

        return simplified_obs
        
    '''
    def init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.total_distance_moved = 0.0
        self.current_y_distance = self.get_y_dir_distance_from_start_point(self.start_point)
        self.roll_turn_speed = rospy.get_param('/excavator/init_roll_vel')
        
    '''

    '''
    ### phase 1 
    def is_done(self, observations):

        distance = observations[0]

        if  self.check_done_interval > distance:
            rospy.loginfo("Bucket reach the dig point==>" + str(distance))
            done = True
            #done = False
        else:
            rospy.logdebug("Bucket haven't reach the dig point yet==>" + str(distance))
            done = False

        if self.evamass == True:
            done = True

        return done
    '''

    ### phase 2
    def is_done(self, observations):

        distance = observations[0]

        if  self.check_done_interval > distance:
            #rospy.loginfo("Bucket reach the dig point==>" + str(distance))
            # done = True
            done = False
        else:
            #rospy.logdebug("Bucket haven't reach the dig point yet==>" + str(distance))
            done = False

        #if self.evamass == True:
        # done = True

        return done

    def set_action(self, action):

        '''
        self.move_joints(1, action(0))
        self.move_joints(2, action(1))
        self.move_joints(3, action(2))
        self.move_joints(4, action(3))
        '''
        # print(action)
        
        rospy.logdebug(" Velocity>>" + str(action))
        self.joint_1_velocity = action[0]
        self.joint_2_velocity = action[1]
        self.joint_3_velocity = action[2]
        self.joint_4_velocity = action[3]
        self._joint_1_velocity_pub.publish(self.joint_1_velocity)
        self._joint_2_velocity_pub.publish(self.joint_2_velocity)
        self._joint_3_velocity_pub.publish(self.joint_3_velocity)
        self._joint_4_velocity_pub.publish(self.joint_4_velocity)
        time.sleep(0.15)

        self.act1.append(self.joint_1_velocity)
        self.act2.append(self.joint_2_velocity)
        self.act2.append(self.joint_3_velocity)
        self.act4.append(self.joint_4_velocity)
        time.sleep(0.15)

        savedata = np.r_[self.act1, self.act2, self.act3, self.act4]
        savedata.tofile("/home/xiaofei/catkin_ws/src/excavator_training/plots/act_noAcc.bin")
        '''
        self._joint_1_velocity_pub.publish(0.0)
        self._joint_2_velocity_pub.publish(0.0)
        self._joint_3_velocity_pub.publish(0.0)
        self._joint_4_velocity_pub.publish(0.0)
        time.sleep(0.025)
        '''

        
        '''
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:# turn the link_1 clockwise
            self.link_1_speed = self.link_1_clockwise_value
            # We tell the Joint to roll the Link at the selected speed
            self.move_joints(1,self.link_1_speed)
        elif action == 1:# turn the link_1 conterclockwise
            self.link_1_speed = self.link_1_conterclockwise_value
            self.move_joints(1, self.link_1_speed)
        elif action == 2:# Stop link_1
            self.link_1_speed = 0.0
            self.move_joints(1, self.link_1_speed)
        elif action == 3:# rasie the link_2
            self.link_2_speed = self.link_2_raise_value
            self.move_joints(2, self.link_2_speed)
        elif action == 4:# drop the link_2
            self.link_2_speed = self.link_2_drop_value
            self.move_joints(2, self.link_2_speed)
        elif action == 5:# stop the link_2
            self.link_2_speed = 0.0
            self.move_joints(2, self.link_2_speed)
        elif action == 6:# raise the link_3
            self.link_3_speed = self.link_3_raise_value
            self.move_joints(3, self.link_3_speed)
        elif action == 7:# drop the link_3
            self.link_3_speed = self.link_3_drop_value
            self.move_joints(3, self.link_3_speed)
        elif action == 8:# stop the link_3
            self.link_3_speed = 0.0
            self.move_joints(3, self.link_3_speed)
        elif action == 9:# the link_4 digging process
            self.link_4_speed = self.link_4_dig_value
            self.move_joints(4, self.link_4_speed)
        elif action == 10:# drop the link_4
            self.link_4_speed = self.link_4_drop_value
            self.move_joints(4, self.link_4_speed)
        elif action == 11:# stop the link_4
            self.link_4_speed = 0.0
            self.move_joints(4, self.link_4_speed)
        '''


        
    def _get_observation(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        MyCubeSingleDiskEnv API DOCS
        :return:
        """

        # desicde a way to define the robot

        # We get the distance from the destination
        x_distance_to_dig = self.get_x_distance_from_dig_point(self.dig_point)
        y_distance_to_dig = self.get_y_distance_from_dig_point(self.dig_point)
        z_distance_to_dig = self.get_z_distance_from_dig_point(self.dig_point)

        sq_distance_to_dig = np.square(x_distance_to_dig)+np.square(y_distance_to_dig)+np.square(z_distance_to_dig)

        # We get the current speed of the Roll Disk
        # current_disk_roll_vel = self.get__velocity()

        # We get the linear speed in the y axis
        # y_linear_speed = self.get_y_linear_speed()

        # We get the current positions of each revolute joint
        joint_1_position = self.joints.position[0]
        joint_2_position = self.joints.position[1]
        joint_3_position = self.joints.position[2]
        joint_4_position = self.joints.position[3]

        # We get the current height of the bottom of the bucket
        bottom = self.transform_bottom.transform.translation.z

        # We get the current torque of joint_2
       # joint_4_torque = self.torque.wrench.torque.y

        excavator_observations = [
            round(sq_distance_to_dig, 5),
            round(joint_1_position, 5),
            round(joint_2_position, 5),
            round(joint_3_position, 5),
            round(joint_4_position, 5),
            round(bottom, 5)
        ]

        '''
        self.x_pre.append(excavator_observations[0])
        self.x_bot1.append(excavator_observations[5])
        self.joint1.append(excavator_observations[1])
        self.joint2.append(excavator_observations[2])
        self.joint3.append(excavator_observations[3])
        self.joint4.append(excavator_observations[4])

        print("here")
        savedata = np.r_[self.x_pre, self.joint1, self.joint2, self.joint3, self.joint4, self.x_bot1]
        savedata.tofile("/home/xiaofei/catkin_ws/src/excavator_training/plots/obs_noAcc_1.bin")
        '''

        return excavator_observations


    def get_x_distance_from_dig_point(self, dig_point):
        """
        Calculates the distance from the given point and the current position
        given by odometry. In this case the increase or decrease in x.
        :param start_point:
        :return:
        """
        x_distance_to_dig = self.transform_target.transform.translation.x - dig_point.x
    
        return x_distance_to_dig


    def get_y_distance_from_dig_point(self, dig_point):

        y_distance_to_dig = self.transform_target.transform.translation.y - dig_point.y

        return y_distance_to_dig


    def get_z_distance_from_dig_point(self, dig_point):
        z_distance_to_dig = self.transform_target.transform.translation.z - dig_point.z

        return z_distance_to_dig


    def compute_reward(self, observations, done):

        ##### phase 1 reward

        if not done:

            distance = observations[0]
            rospy.logdebug("distance=" + str(distance))
            reward_distance = -100 * distance + 50 # first time -50, not converge
                                               # second time -1000 converge but slow

            # Negativ reward if the joints exceed the limits >
            joint_2_angle = observations[2]
            if 0.25 < joint_2_angle < 0.35:
                reward_joint_2 = - joint_2_angle * 1000
            elif -0.35 < joint_2_angle < -0.25:
                reward_joint_2 =  joint_2_angle * 1000
               # rospy.logdebug("joint_2 exceeded the limit")
            else:
                reward_joint_2 = 0

            joint_3_angle = observations[3]
            if 1.47 < joint_3_angle < 1.57:
                reward_joint_3 = - joint_3_angle * 200
               # rospy.logdebug("joint_3 did not exceed the limit")
            elif 0 < joint_3_angle < 0.1:
                reward_joint_3 = -300 + 1000 * joint_3_angle
               # rospy.logdebug("joint_3 exceeded the limit")
            else:
                reward_joint_3 = 0

            joint_4_angle = observations[4]
            if -1.75 < joint_4_angle < -1.65:
                reward_joint_4 = joint_4_angle * 200
                # rospy.logdebug("joint_4 did not exceed the limit")
            elif 0.2 < joint_4_angle < 0.3:
                reward_joint_4 = - joint_4_angle * 1000
               # rospy.logdebug("joint_4 exceeded the limit")
            else:
                reward_joint_4 = 0

            # guarantee that the bottom of the bucket on hit the ground
            bottom = observations[5]
            if bottom < 2.3:
                reward_bottom = -(bottom - 2.3) * 1000
            else:
                reward_bottom = 0

            vel_change_1 = np.square(self.joint_1_velocity - self.joint_1_preVelocity)
            vel_change_2 = np.square(self.joint_2_velocity - self.joint_2_preVelocity)
            vel_change_3 = np.square(self.joint_3_velocity - self.joint_3_preVelocity)
            vel_change_4 = np.square(self.joint_4_velocity - self.joint_4_preVelocity)

            if vel_change_1 > 0.1:
                reward_vel_1 = - 10000 * vel_change_1
            else:
                reward_vel_1 = 0

            if self.joint_1_velocity * self.joint_1_preVelocity < 0:
                reward_vel_1 += -100

            if vel_change_2 > 0.1:
                reward_vel_2 = - 10000 * vel_change_2
            else:
                reward_vel_2 = 0

            if self.joint_2_velocity * self.joint_2_preVelocity < 0:
                reward_vel_2 += -100

            if vel_change_3 > 0.1:
                reward_vel_3 = - 10000 * vel_change_3
            else:
                reward_vel_3 = 0

            if self.joint_3_velocity * self.joint_3_preVelocity < 0:
                reward_vel_3 += -100

            if vel_change_4 > 0.1:
                reward_vel_4 = - 10000 * vel_change_4
            else:
                reward_vel_4 = 0

            if self.joint_4_velocity * self.joint_4_preVelocity < 0:
                reward_vel_4 += -100

            '''
            if self.check_done_interval > distance:
                reward = self.end_episode_points
                rospy.loginfo("reward to the check point" + str(reward))
            else:
            '''

            reward = round(reward_distance, 3)  + reward_joint_2 + reward_joint_3 + reward_joint_4 + reward_bottom + reward_vel_1 + reward_vel_2 + reward_vel_3 + reward_vel_4
            rospy.logdebug("reward_distance=" + str(reward_distance))
            rospy.logdebug("reward_joint_2=" + str(reward_joint_2))
            rospy.logdebug("reward_joint_3=" + str(reward_joint_3))
            rospy.logdebug("reward_joint_4=" + str(reward_joint_4))
            rospy.logdebug("reward=" + str(reward))

            self.sum_reward += reward
            self.rewards.append(self.sum_reward)
            savedata = np.r_[self.rewards]
            savedata.tofile("/home/xiaofei/catkin_ws/src/excavator_training/plots/obs_noAcc_reward.bin")

        else:
            reward = self.end_episode_points

        '''
        ##### phase 2 reward

        # joints do not exceed the limits
        joint_2_angle = observations[2]
        if -0.35 < joint_2_angle < 0.35:
            reward_joint_2 = 0
            rospy.logdebug("joint_2 did not exceed the limit")
        else:
            reward_joint_2 = -1
            rospy.logdebug("joint_2 exceeded the limit")

        joint_3_angle = observations[3]
        if 0 < joint_3_angle < 1.57:
            reward_joint_3 = 0
            rospy.logdebug("joint_3 did not exceed the limit")
        else:
            reward_joint_3 = -1
            rospy.logdebug("joint_3 exceeded the limit")

        joint_4_angle = observations[4]

        if -1.75 < joint_3_angle < 0:
            reward_joint_4 = 0
            rospy.logdebug("joint_4 did not exceed the limit")
        else:
            reward_joint_4 = -1
            rospy.logdebug("joint_4 exceeded the limit")

        # the velocity should not change so fast
        if observations[5] > 0.2:
            reward_vel_2 = - 10
        else:
            reward_vel_2 = 1
        if observations[6] > 0.2:
            reward_vel_3 = -10
        else:
            reward_vel_3 = 1
        if observations[7] > 0.2:
            reward_vel_4 = -10
        else:
            reward_vel_4 = 1

        # check the joint_2 torque
        joint_4_torque = observations[8]
        if self.numstep % 64 == 0:
            rospy.loginfo("Torque of joint_4 is =>" + str(joint_4_torque))
            reward = 100 * (-6000 - joint_4_torque)
            self.result += reward
            self.res.append(self.result)
            savedata = np.r_[self.res]
            savedata.tofile('/home/xiaofei/catkin_ws/src/excavator_training/training_results/Phase_2_reward.bin')
            self.result = 0
            return reward

        else:
            if joint_4_torque > 9000000 or -9000000 > joint_4_torque:
                reward_contact = -10000
            else:
                reward_contact = 1

        '''
        '''
        # bucket should open
        joint_4_angle = observations[4]

        reward_joint_4 = - 10 * joint_4_angle
        '''

        '''
        # bucket should reach the dig point
        distance_to_dig = observations[0]
        reward_distance_to_dig = -1000 * distance_to_dig

        if  distance_to_dig < self.check_done_interval:
            reward_distance_to_dig = 1000
            rospy.loginfo("Reach the dig depth =>" + str(reward_distance_to_dig))
        '''

        return reward

    def torque_callback(self, data):
        self.torque = data
        # print(self.torque.wrench)

    def joints_callback(self, data):
        self.joints = data

    def transformTarget_callback(self, data):
        self.transform_target = data

    def transformBottom_callback(self, data):
        self.transform_bottom = data

    def odom_callback(self,data):
        self.odom = data
        
    def check_all_sensors_ready(self):
        self.check_joint_states_ready()
        # self.check_ft_sensor_topic_ready()
        #  self.check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/excavator/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current excavator/joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current excavator/joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def check_ft_sensor_topic_ready(self):
        self.torque = None
        while self.torque is None and not rospy.is_shutdown():
            try:
                self.torque = rospy.wait_for_message("/ft_sensor_topic", WrenchStamped, timeout=10.0)
                rospy.logdebug("Current /ft_sensor_topic READY=>" + str(self.torque))

            except:
                rospy.logerr("Current /ft_sensor_topic not ready yet, retrying for getting WrenchStamped")
        return self.torque

    def check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/excavator/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /excavator/odom READY=>" + str(self.odom))

            except:
                rospy.logerr("Current /excavator/odom not ready yet, retrying for getting odom")

        return self.odom

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(40)  # 10hz
        while (self._joint_1_velocity_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _joint_1_velocity_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        while (self._joint_2_velocity_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _joint_2_velocity_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        while (self._joint_3_velocity_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _joint_3_velocity_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        while (self._joint_4_velocity_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _joint_4_velocity_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_base_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
        
    def move_joints(self, joint_name, roll_speed):
        joint_speed_value = Float64()
        joint_speed_value.data = roll_speed
        rospy.logdebug(str(joint_name) + " Velocity>>" + str(joint_speed_value))
        if joint_name == 1:
            self._joint_1_velocity_pub.publish(joint_speed_value)
            time.sleep(0.5)
            # self.wait_until_roll_is_in_vel(1, joint_speed_value.data)
        elif joint_name == 2:
            self._joint_2_velocity_pub.publish(joint_speed_value)
            time.sleep(0.5)
            # self.wait_until_roll_is_in_vel(2, joint_speed_value.data)
        elif joint_name == 3:
            self._joint_3_velocity_pub.publish(joint_speed_value)
            time.sleep(0.5)
            #  self.wait_until_roll_is_in_vel(3, joint_speed_value.data)
        elif joint_name == 4:
            self._joint_4_velocity_pub.publish(joint_speed_value)
            time.sleep(0.5)
            #  self.wait_until_roll_is_in_vel(4, joint_speed_value.data)
    
    def wait_until_roll_is_in_vel(self, joint_name, velocity):
    
        rate = rospy.Rate(40)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.3
        v_plus = velocity + epsilon
        v_minus = velocity - epsilon
        while not rospy.is_shutdown():
            joint_data = self.check_joint_states_ready()
            if joint_name == 1:
                roll_vel = joint_data.velocity[0]
            elif joint_name == 2:
                roll_vel = joint_data.velocity[1]
            elif joint_name == 3:
                roll_vel = joint_data.velocity[2]
            elif joint_name == 4:
                roll_vel = joint_data.velocity[3]
            rospy.logdebug("VEL=" + str(roll_vel) + ", ?RANGE=[" + str(v_minus) + ","+str(v_plus)+"]")
            are_close = (roll_vel <= v_plus) and (roll_vel > v_minus)
            if are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        return delta_time


    def set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_joints(1, self.init_roll_vel)
        self.move_joints(2, self.init_roll_vel)
        self.move_joints(3, self.init_roll_vel)
        self.move_joints(4, self.init_roll_vel)

        return True
        
        
    def convert_obs_to_state(self,observations):
        """
        Converts the observations used for reward and so on to the essentials for the robot state
        In this case we only need the orientation of the cube and the speed of the disc.
        The distance doesnt condition at all the actions
        """
        distance_to_dig = observations[0]
        joint_1 = observations[1]
        joint_2 = observations[2]
        joint_3 = observations[3]
        joint_4 = observations[4]
        bottom = observations[5]

        state_converted = [distance_to_dig, joint_1 ,joint_2, joint_3, joint_4, bottom]
    
        return state_converted

