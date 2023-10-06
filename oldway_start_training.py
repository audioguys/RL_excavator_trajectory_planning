#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import gym
import numpy as np
import time
import qlearn
from gym import wrappers
import logging
import math
import os.path
import sys

# ROS packages required
import rospy
import rospkg

# import our training environment
import excavator_env
# import excavator_env_noAcc
import excavator_env_RewAcc
# import excavator_env_HardAcc
# import excavator_env_entire
import  excavator_env_last
# import excavator_env_DQN
from gazebo_connection import GazeboConnection
from sensor_msgs.msg import JointState

# reduce is moved to functools
from functools import reduce

# import reinforcement learning algorithm
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from std_msgs.msg import Float64




CHECKPOINT_DIR = '/home/xiaofei/catkin_ws/src/excavator_training/'
logdir = "/home/xiaofei/catkin_ws/src/excavator_training/training_results/logs"

'''
def puber():
    stop_2 = False
    stop_1 = False
    stop_3 = False
    stop_4 = False
    pub_1 = rospy.Publisher('/excavator/joint_1_controller/command', Float64, queue_size=1)
    pub_2 = rospy.Publisher('/excavator/joint_2_controller/command', Float64, queue_size=1)
    pub_3 = rospy.Publisher('/excavator/joint_3_controller/command', Float64, queue_size=1)
    pub_4 = rospy.Publisher('/excavator/joint_4_controller/command', Float64, queue_size=1)
    rate = rospy.Rate(40)
    print('here')
    while not rospy.is_shutdown():
        print('there')
        pub_1.publish(-0.05)
'''

'''
class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose=0)

    def _on_step(self) -> bool:
        self.total_reward += self.locals["rewards"][0]
        self.episode_lens.append(self.total_reward)
        return True

    def _on_training_end(self) -> None:
'''


if __name__ == '__main__':

    # The root logger always default to WARNING level, so that we can't see the Loginfo in the shell. We change that here.
    rospy.init_node('excavator_gym', anonymous=True, log_level=rospy.INFO)

    # logging.getLogger().setLevel(logging.INFO)
    
    # Create the Gym environment
    env = gym.make('excavator-v4')


    env = Monitor(env)
    env = DummyVecEnv([lambda : env])
    # env = VecNormalize(env, norm_obs =True, norm_reward=True)
    rospy.loginfo ( "Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('excavator_training')
    outdir = "/home/xiaofei/catkin_ws/src/excavator_training/log/"

    rospy.loginfo ( "Monitor Wrapper started")


    last_time_steps = np.ndarray(0)


    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/excavator/alpha")
    Epsilon = rospy.get_param("/excavator/epsilon")
    Gamma = rospy.get_param("/excavator/gamma")
    epsilon_discount = rospy.get_param("/excavator/epsilon_discount")
    nepisodes = rospy.get_param("/excavator/nepisodes")
    nsteps = rospy.get_param("/excavator/nsteps")

    running_step = rospy.get_param("/excavator/running_step")

    '''
    # DQN
    dqn_params = {
        'learning_starts': 10000,
    }

    model = DQN("MlpPolicy", env, verbose = 1, **dqn_params, tensorboard_log=logdir)
    print(model.policy)
    model.learn(total_timesteps=128000)
    model.save(CHECKPOINT_DIR + 'DQN_model')
    env.save(CHECKPOINT_DIR + 'DQN_env')
    '''


    ppo_params = {
        'learning_rate': 3e-5,
        'gamma': 0.99,
        # Discount factor for future reward, reduced from 0.99 because the agent only need to see a few step ahead
        'n_steps': 255,  # Steps to run per update, enough to pass an episode
        'batch_size': 255
    }
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    
    # Define the PPO agent
    model = PPO('MlpPolicy', env, verbose=1, **ppo_params, tensorboard_log=logdir, policy_kwargs=policy_kwargs)
    print(model.policy)
    
    model.learn(total_timesteps=255)
    # model.save(CHECKPOINT_DIR + 'latest_phase4_model')
    # env.save(CHECKPOINT_DIR + 'latest_phase4_env')
    


    '''
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_model.zip', env = env)
    print(model.policy)
    
    model.learn(total_timesteps=409600)
    model.save(CHECKPOINT_DIR + 'latest_phase4_1_model')
    env.save(CHECKPOINT_DIR + 'latest_phase4_1_env')
    '''

    '''
    # eva noAcc_1
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/phase1_env_noAcc', env)
    env = VecNormalize(env, training=False)
    # env.norm_reward = False
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/phase1_noAcc.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    '''



    
    '''
    # eva rewAcc_1
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/phase1_env_ReAccLimit', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/phase1_model_ReAccLimit.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
    '''

    

    '''
    # eva hardAcc_1
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_model.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    '''



    '''
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase3_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase3_model.zip', env = env)
    print(model.policy)

    obs = env.reset()
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    '''

    '''
    # eva phase_4
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_model.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(80):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    '''


    '''
    # last eva without balls
    # evaluate model
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_model.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    
    print("phase = 2")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_2/latest_phase2_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_2/latest_phase2_model.zip', env = env)
    print(model.policy)

    for i in range(14):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
    

    print("phase = 3")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_3/latest_phase3_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_3/latest_phase3_model.zip', env = env)
    print(model.policy)

    for i in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    # eva phase_4
    print("phase = 4")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_model.zip', env=env)
    print(model.policy)

    for i in range(83):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    '''

    '''
    # last eva with balls
    # evaluate model
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_1/last_phase1_model.zip', env=env)
    print(model.policy)

    obs = env.reset()
    for i in range(14):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    print("phase = 2")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_2/latest_phase2_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_2/latest_phase2_model.zip', env=env)
    print(model.policy)

    for i in range(14):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    print("phase = 3")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_3/latest_phase3_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/phase_3/latest_phase3_model.zip', env=env)
    print(model.policy)

    for i in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    # eva phase_4
    print("phase = 4")
    env = VecNormalize.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_env', env)
    env = VecNormalize(env, training=False)
    model = PPO.load('/home/xiaofei/catkin_ws/src/excavator_training/latest_phase4_1_model.zip', env=env)
    print(model.policy)

    for i in range(81):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    for i in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
    '''


    

    


    '''
    phase_1 position: [-0.49114737630687433, 0.0523408589266765, 0.7642673768446624, -0.6083870989100744]
                    [-0.475636017551615, 0.14823904835989943, 0.7340632957934776, -0.698880713042275]
    position: [-0.4690018297464862, 0.07158254373337236, 0.7445968539404086, -0.6233999732775386]
    position: [-0.5433857598282748, 0.043993875894791756, 0.8594084231472605, -0.7154292064730496]
    '''

    '''
    phase_2 position: [-0.5836187734552656, -0.23171819711319142, 0.8633224398175425, -0.9929149433360909]
    position: [-0.5365399144079275, -0.22369516895462738, 0.8311059924135753, -1.102864339259341]
    position: [-0.6013070597316394, -0.10188598666150384, 0.8015462106200335, -1.0780459165994882]
    
    phase_3 position: [-0.5389425936703756, -0.3414042346738553, 1.1573708540596446, 0.30083234862756925]
    '''







    '''
    # Define PPO parameters phase 2
    ppo_params = {
        'learning_rate': 3e-4,
        'gamma': 0.9,
        # Discount factor for future reward, reduced from 0.99 because the agent only need to see a few step ahead
        # 'n_steps': 512,  # Steps to run per update, enough to pass an episode
        'batch_size': 64
    }
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    # Define the PPO agent
    model = PPO('MlpPolicy', env, verbose=1, **ppo_params, tensorboard_log=logdir, policy_kwargs=policy_kwargs)
    print(model.policy)

    # callback = CustomCallback()

    # Train the agent
    # reset_num_timesteps=False
    model.learn(total_timesteps=4096)
    model.save(CHECKPOINT_DIR + 'latest_phase2_model')
    '''

    '''
    
    # TD3 model
    TD3_params = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'learning_starts': 1024,
        # Discount factor for future reward, reduced from 0.99 because the agent only need to see a few step ahead
        'batch_size': 64,
        'train_freq': (256, "step")
    }

    model = TD3('MlpPolicy', env, verbose=2, **TD3_params, tensorboard_log=logdir)

    for iters in range(300):
        env.stats_recorder.done = None

        obs = env.reset()
        done = False
        rospy.loginfo("############### Start episode=>" + str(iters))

        model.learn(total_timesteps=1024, reset_num_timesteps=False)
        model.save(CHECKPOINT_DIR + 'TD3_model')
    '''
    '''

    

    # Evaluate the agent
    joint_1 = []
    joint_2 = []
    joint_3 = []
    joint_4 = []
    distance = []
    env.stats_recorder.done = None
    obs = env.reset()
    done = False
    rospy.loginfo("############### Start evaluating")
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        distance.append(obs[0])
        joint_1.append(obs[1])
        joint_2.append(obs[2])
        joint_3.append(obs[3])
        joint_4.append(obs[4])
        if done:
            break
    savedata = np.r_[distance,joint_1,joint_2,joint_3,joint_4]
    savedata.tofile(logdir + 'phase_1.3.bin')
    plt.plot(np.arange(0, len(joint_1)), joint_1)
    plt.show()
    '''

    '''
    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = -math.inf

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # Initialize the environment and get first state of the robot
        env.stats_recorder.done = None

        observation = env.reset()
        state = ''.join(map(str, observation))
        
        episode_time = rospy.get_rostime().to_sec()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.loginfo("############### Start Step=>"+str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.loginfo ("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.loginfo(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("############### state we were=>" + str(state))
            rospy.logwarn("############### action that we took=>" + str(action))
            rospy.logwarn("############### reward that action gave=>" + str(reward))
            rospy.logwarn("############### State in which we will start nect step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.loginfo("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            #rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+
                         " - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    
    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    #print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
    '''


