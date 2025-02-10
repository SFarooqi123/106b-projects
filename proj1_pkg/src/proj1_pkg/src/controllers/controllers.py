#!/usr/bin/env python

"""
Starter script for Project 1B. 
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import sys
import numpy as np
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from trac_ik_python.trac_ik import IK
from sawyer_pykdl import sawyer_kinematics
# Lab imports
from utils.utils import *
from trajectories import Trajectory, LinearTrajectory
import time

# ROS imports
try:
    import tf
    import tf2_ros
    import rospy
    import baxter_interface
    import intera_interface
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import RobotTrajectory
except:
    pass

NUM_JOINTS = 7

#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from paths.paths import MotionPath

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=20, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.
        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):

    def __init__(self, start_position, goal_position, total_time):

        Trajectory.__init__(self, total_time)
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance = self.goal_position - self.start_position
        self.acceleration = (self.distance * 4.0) / (self.total_time ** 2) # keep constant magnitude acceleration
        self.v_max = (self.total_time / 2.0) * self.acceleration # maximum velocity magnitude
        self.desired_orientation = np.array([0, 1, 0, 0])

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate the position of the end effector at time t, 
            # For the first half of the trajectory, maintain a constant acceleration
            pos =  self.start_position + (1/2)*self.acceleration*time**2
        else:
            # TODO: Calculate the position of the end effector at time t, 
            # For the second half of the trajectory, maintain a constant acceleration
            # Hint: Calculate the remaining distance to the goal position. 

            # pos = ((self.start_position + self.goal_position)/2)+self.v_max*(time-self.total_time/2)-(1/2)*self.acceleration*(time-self.total_time/2)**2
            t_dec = time - self.total_time / 2.0
            pos = (self.goal_position + self.start_position)/2 + self.v_max * t_dec - 0.5 * self.acceleration * (t_dec ** 2)



        return np.hstack((pos, self.desired_orientation))

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        The function get_g_matrix from utils may be useful to perform some frame
        transformations.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate velocity using the acceleration and time
            # For the first half of the trajectory, we maintain a constant acceleration

            
            linear_vel = self.acceleration*time
        else:
            # TODO: start slowing the velocity down from the maximum one
            # For the second half of the trajectory, maintain a constant deceleration


            # linear_vel = self.v_max - self.acceleration * (time-self.total_time/2)
            t_dec = time - self.total_time / 2.0
            linear_vel = self.v_max - self.acceleration * t_dec
        return np.hstack((linear_vel, np.zeros(3)))

class CircularTrajectory(Trajectory):
    def __init__(self, center_position, radius, total_time, loops = 1):
        Trajectory.__init__(self, total_time)
        self.center_position = center_position
        self.radius = radius
        self.angular_acceleration = (2 * np.pi * 4.0) / (self.total_time ** 2) # keep constant magnitude acceleration
        self.angular_v_max = (self.total_time / 2.0) * self.angular_acceleration # maximum velocity magnitude
        self.desired_orientation = np.array([0, 1, 0, 0])
        self.loops = loops

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """

        #time = (time * self.loops) % self.total_time
        if time < self.total_time / 2.0:
            # TODO: calculate the ANGLE of the end effector at time t, 
            # For the first half of the trajectory, maintain a constant acceleration
            

            theta = 0.5 * self.angular_acceleration * time**2

        else:
            # TODO: Calculate the ANGLE of the end effector at time t, 
            # For the second half of the trajectory, maintain a constant acceleration
            # Hint: Calculate the remaining angle to the goal position. 

            time_half = self.total_time / 2.0
            theta = np.pi + self.angular_v_max*(time-time_half) - 0.5 * self.angular_acceleration * (time - time_half)**2 

        #elif time <= self.total_time *.75:

        #     time_half = self.total_time*.75
        #     theta = np.pi + self.angular_v_max*(time-time_half) - 0.5 * self.angular_acceleration * (time - time_half)**2


        # else:
        #     time_half = self.total_time / 2.0
        #     theta = np.pi + self.angular_v_max*(time-time_half) - 0.5 * self.angular_acceleration * (time - time_half)**2


        pos_d = np.ndarray.flatten(self.center_position + self.radius * np.array([np.cos(theta), np.sin(theta), 0]))
        return np.hstack((pos_d, self.desired_orientation))


    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        The function get_g_matrix from utils may be useful to perform some frame
        transformations.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        if time <= self.total_time / 2.0:
            # TODO: calculate ANGULAR position and velocity using the acceleration and time
            # For the first half of the trajectory, we maintain a constant acceleration


            theta = 0.5 * self.angular_acceleration * time**2
            theta_dot = self.angular_acceleration * time 
    
        else:
            # TODO: start slowing the ANGULAR velocity down from the maximum one
            # For the second half of the trajectory, maintain a constant deceleration
            
            time_half = self.total_time / 2.0
            theta = np.pi + self.angular_v_max*(time-time_half) - 0.5 * self.angular_acceleration * (time - time_half)**2
            theta_dot = self.angular_v_max - self.angular_acceleration * (time - time_half)
        vel_d = np.ndarray.flatten(self.radius * theta_dot * np.array([-np.sin(theta), np.cos(theta), 0]))
        return np.hstack((vel_d, np.zeros(3)))

class PolygonalTrajectory(Trajectory):
    def __init__(self, points, total_time):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.
        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, total_time)
        self.start_position = start_position
        self.goal_positions = points
        self.num_points = len(points)
        self.distance = self.goal_position - self.start_position
        self.acceleration = (self.distance * 4.0) / (self.total_time ** 2) # keep constant magnitude acceleration
        self.v_max = (self.total_time / 2.0) * self.acceleration # maximum velocity magnitude
        self.desired_orientation = np.array([0, 1, 0, 0])


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """

        sim_time = time % (self.total_time / num_points + 1)
        leg = time // (self.total_time / num_points + 1)
        time_per_point = self.total_time / (num_points + 1)
        if (leg == 0):
            self.distance = point[0] - self.start_position
        else:
            self.distance = point[leg] - point[leg - 1]
        
        self.acceleration = (self.distance * 4.0) / (time_per_point ** 2)
        

        if sim_time <= time_per_point / 2.0:
            # TODO: calculate the position of the end effector at time t, 
            # For the first half of the trajectory, maintain a constant acceleration
            if (leg == 0):
                pos =  self.start_position + (1/2)*self.acceleration*sim_time**2
            else:
                pos =  points[leg - 1] + (1/2)*self.acceleration*sim_time**2
        
        else:
            # TODO: Calculate the position of the end effector at time t, 
            # For the second half of the trajectory, maintain a constant acceleration
            # Hint: Calculate the remaining distance to the goal position. 
            # pos = ((self.start_position + self.goal_position)/2)+self.v_max*(time-self.total_time/2)-(1/2)*self.acceleration*(time-self.total_time/2)**2
            if (leg == 0):
                t_dec = sim_time - time_per_point / 2.0
                pos = (leg[0] + self.start_position)/2 + self.v_max * t_dec - 0.5 * self.acceleration * (t_dec ** 2)

        
    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """

        sim_time = time % (self.total_time / num_points + 1)
        leg = time // (self.total_time / num_points + 1)
        time_per_point = self.total_time / (num_points + 1)

        if sim_time <= time_per_point / 2.0:
            # TODO: calculate velocity using the acceleration and time
            # For the first half of the trajectory, we maintain a constant acceleration

            
            linear_vel = self.acceleration*sim_time
        else:
            # TODO: start slowing the velocity down from the maximum one
            # For the second half of the trajectory, maintain a constant deceleration


            # linear_vel = self.v_max - self.acceleration * (time-self.total_time/2)
            t_dec = sim_time - time_per_point / 2.0
            linear_vel = self.v_max - self.acceleration * t_dec
        return np.hstack((linear_vel, np.zeros(3)))

def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory()
    elif args.task == 'circle':
        trajectory = CircularTrajectory()
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory()
    return trajectory

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)
def lookup_tag(tag_number):
    """
    Given an AR tag number, this returns the position of the AR tag in the robot's base frame.
    You can use either this function or try starting the scripts/tag_pub.py script.  More info
    about that script is in that file.  

    Parameters
    ----------
    tag_number : int

    Returns
    -------
    3x' :obj:`numpy.ndarray`
        tag position
    """

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    to_frame = 'ar_marker_{}'.format(tag_number)

    try:
        trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)
        print("Retrying ...")

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    return np.array(tag_pos)

def get_trajectory(limb, kin, ik_solver, tag_pos, args):
    """
    Returns an appropriate robot trajectory for the specified task.  You should 
    be implementing the path functions in paths.py and call them here
    
    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    tag_pos : 3x' :obj:`numpy.ndarray`
        
    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """

    num_way = args.num_way
    controller_name = args.controller_name
    task = args.task

    # target_position = tag_pos[0]
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        # trans = tfBuffer.lookup_transform('base', 'stp_022412TP99883_base', rospy.Time(0), rospy.Duration(10.0))
        trans = tfBuffer.lookup_transform('base', 'right_gripper_base', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)
    print(task)
    if task == 'line':
        target_pos = tag_pos[0]
        target_pos[2] += 0.5
        trajectory = LinearTrajectory(start_position=current_position, goal_position=target_pos, total_time=8)

    elif task == 'circle':
        target_pos = tag_pos[0]
        target_pos[2] += 0.5
        print("TARGET POSITION:", target_pos)
        trajectory = CircularTrajectory(center_position=target_pos, radius=0.1, total_time=15)


    elif task == 'polygon':
        pos1 = tag_pos + [.1, .1, .4]
        pos2 = tag_pos + [-.1, .1, .5]
        pos_3 = tag_pos + [-.1, -.1, .3]
        trajectory = PolygonalTrajectory([pos1, pos2, pos3])
    else:
        raise ValueError('task {} not recognized'.format(task))
    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, controller_name!='workspace')

class Controller:

    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin

        # Set this attribute to True if the present controller is a jointspace controller.
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly.  

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        """
        pass

    def interpolate_path(self, path, t, current_index = 0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------intera_interface/
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        current_index : int
            waypoint index at which search was terminated 
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points)-1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if (path.joint_trajectory.points[current_index].time_from_start.to_sec() > t):
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown() and \
            current_index < max_index and \
            path.joint_trajectory.points[current_index+1].time_from_start.to_sec() < t+epsilon
        ):
            current_index = current_index+1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[current_index].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[current_index+1].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index+1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index+1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index+1].accelerations
            )

            target_position = target_position_low + \
                (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
            target_velocity = target_velocity_low + \
                (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + \
                (t - time_low)/(time_high - time_low)*(target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(path.joint_trajectory.points[current_index].positions)
            target_velocity = np.array(path.joint_trajectory.points[current_index].velocities)
            target_acceleration = np.array(path.joint_trajectory.points[current_index].velocities)

        return (target_position, target_velocity, target_acceleration, current_index)


    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)

    def plot_results(
        self,
        times,
        actual_positions, 
        actual_velocities, 
        target_positions, 
        target_velocities
    ):
        """
        Plots results.
        If the path is in joint space, it will plot both workspace and jointspace plots.
        Otherwise it'll plot only workspace

        Inputs:
        times : nx' :obj:`numpy.ndarray`
        actual_positions : nx7 or nx6 :obj:`numpy.ndarray`
            actual joint positions for each time in times
        actual_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            actual joint velocities for each time in times
        target_positions: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace positions for each time in times
        target_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace velocities for each time in times
        """

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)

        # Find the actual workspace positions and velocities
        actual_workspace_positions = np.zeros((len(times), 3))
        actual_workspace_velocities = np.zeros((len(times), 3))
        actual_workspace_quaternions = np.zeros((len(times), 4))

        for i in range(len(times)):
            positions_dict = joint_array_to_dict(actual_positions[i], self._limb)
            fk = self._kin.forward_position_kinematics(joint_values=positions_dict)
            
            actual_workspace_positions[i, :] = fk[:3]
            actual_workspace_velocities[i, :] = \
                self._kin.jacobian(joint_values=positions_dict)[:3].dot(actual_velocities[i])
            actual_workspace_quaternions[i, :] = fk[3:]
        # check if joint space
        if self.is_jointspace_controller:
            # it's joint space

            target_workspace_positions = np.zeros((len(times), 3))
            target_workspace_velocities = np.zeros((len(times), 3))
            target_workspace_quaternions = np.zeros((len(times), 4))

            for i in range(len(times)):
                positions_dict = joint_array_to_dict(target_positions[i], self._limb)
                target_workspace_positions[i, :] = \
                    self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
                target_workspace_velocities[i, :] = \
                    self._kin.jacobian(joint_values=positions_dict)[:3].dot(target_velocities[i])
                target_workspace_quaternions[i, :] = np.array([0, 1, 0, 0])

            # Plot joint space
            plt.figure()
            joint_num = len(self._limb.joint_names())
            for joint in range(joint_num):
                plt.subplot(joint_num,2,2*joint+1)
                plt.plot(times, actual_positions[:,joint], label='Actual')
                plt.plot(times, target_positions[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Position Error")
                plt.legend()

                plt.subplot(joint_num,2,2*joint+2)
                plt.plot(times, actual_velocities[:,joint], label='Actual')
                plt.plot(times, target_velocities[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Velocity Error")
                plt.legend()
            print("Close the plot window to continue")
            plt.show()

        else:
            # it's workspace
            target_workspace_positions = target_positions
            target_workspace_velocities = target_velocities
            target_workspace_quaternions = np.zeros((len(times), 4))
            target_workspace_quaternions[:, 1] = 1

        plt.figure()
        workspace_joints = ('X', 'Y', 'Z')
        joint_num = len(workspace_joints)
        for joint in range(joint_num):
            plt.subplot(joint_num,2,2*joint+1)
            plt.plot(times, actual_workspace_positions[:,joint], label='Actual')
            plt.plot(times, target_workspace_positions[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Position Error")
            plt.legend()

            plt.subplot(joint_num,2,2*joint+2)
            plt.plot(times, actual_velocities[:,joint], label='Actual')
            plt.plot(times, target_velocities[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Velocity Error")
            plt.legend()

        print("Close the plot window to continue")
        plt.show()

        # Plot orientation error. This is measured by considering the
        # axis angle representation of the rotation matrix mapping
        # the desired orientation to the actual orientation. We use
        # the corresponding angle as our metric. Note that perfect tracking
        # would mean that this "angle error" is always zero.
        angles = []
        for t in range(len(times)):
            quat1 = target_workspace_quaternions[t]
            quat2 = actual_workspace_quaternions[t]
            theta = axis_angle(quat1, quat2)
            angles.append(theta)

        plt.figure()
        plt.plot(times, angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Error Angle of End Effector (rad)")
        print("Close the plot window to continue")
        plt.show()
        

    def execute_path(self, path, rate=500, timeout=None, log=False):
        """
        takes in a path and moves the baxter in order to follow the path.  

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For plotting
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()

        # For interpolation
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            (
                target_position, 
                target_velocity, 
                target_acceleration, 
                current_index
            ) = self.interpolate_path(path, t, current_index)

            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position)
                actual_velocities.append(current_velocity)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                self.stop_moving()
                break

        if log:
            self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
        return True

    def follow_ar_tag(self, tag, args, rate=200, timeout=None, log=True):
        """
        takes in an AR tag number and follows it with the baxter's arm.  You 
        should look at execute_path() for inspiration on how to write this. 

        Parameters
        ----------
        tag : int
            which AR tag to useif log:"""
        # ik_solver = IK("base", "right_hand")
        # limb = intera_interface.Limb('right')

        # kin = sawyer_kinematics('right')
        # tag_pos = [lookup_tag(marker) for marker in args.ar_marker]
        # path = get_trajectory(limb, kin, ik_solver, tag_pos, args)




       # For plotting
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()

        # # For interpolation
        # max_index = len(path.joint_trajectory.points)-1
        # current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        path_update_interval = 6.0  # Recalculate path every 1 second
        last_path_update_time = 0

        ik_solver = IK("base", "right_hand")
        limb = intera_interface.Limb('right')

        kin = sawyer_kinematics('right')
        tag_pos = [lookup_tag(marker) for marker in args.ar_marker]
        path = get_trajectory(limb, kin, ik_solver, tag_pos, args)
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0

        while not rospy.is_shutdown():
            

            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            if t > 40:
                break

            if t - last_path_update_time >= path_update_interval:
                ik_solver = IK("base", "right_hand")
                limb = intera_interface.Limb('right')

                kin = sawyer_kinematics('right')
                tag_pos = [lookup_tag(marker) for marker in args.ar_marker]
                path = get_trajectory(limb, kin, ik_solver, tag_pos, args)
                max_index = len(path.joint_trajectory.points)-1
                current_index = 0
                last_path_update_time = t

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            (
                target_position, 
                target_velocity, 
                target_acceleration, 
                current_index
            ) = self.interpolate_path(path, t, current_index)

            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position)
                actual_velocities.append(current_velocity)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            # if current_index >= max_index:
            #     self.stop_moving()
            #     break

        if log:
            self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
        return True

class FeedforwardJointVelocityController(Controller):
    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        self._limb.set_joint_velocities(joint_array_to_dict(target_velocity, self._limb))

class WorkspaceVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the
    PDJointVelocityController is that this controller compares the baxter's current WORKSPACE position and
    velocity desired WORKSPACE position and velocity to come up with a WORKSPACE velocity command to be sent
    to the baxter.  Then this controller should convert that WORKSPACE velocity command into a joint velocity
    command and sends that to the baxter.  Notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 6x' :obj:`numpy.ndarray`
        Kv : 6x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time.
        target_position will be a 7 vector describing the desired SE(3) configuration where the first
        3 entries are the desired position vector and the next 4 entries are the desired orientation as
        a quaternion, all written in spatial coordinates.
        target_velocity is the body-frame se(3) velocity of the desired SE(3) trajectory gd(t). This velocity
        is given as a 6D Twist (vx, vy, vz, wx, wy, wz).
        This method should call self._kin.forward_position_kinematics() to get the current workspace 
        configuration and self._limb.set_joint_velocities() to set the joint velocity to something.  
        Remember that we want to track a trajectory in SE(3), and implement the controller described in the
        project document PDF.
        Parameters
        ----------
        target_position: (7,) ndarray of desired SE(3) position (px, py, pz, qx, qy, qz, qw) (position + quaternion).
        target_velocity: (6,) ndarray of desired body-frame se(3) velocity (vx, vy, vz, wx, wy, wz).
        target_acceleration: ndarray of desired accelerations (should you need this?).
        """
        raise NotImplementedError
        control_input = None        
        self._limb.set_joint_velocities(joint_array_to_dict(control_input, self._limb))


class PDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the 
    PDJointVelocityController is that this controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the baxter's 
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the baxter.  notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. his method should call
        get_joint_positions and get_joint_velocities from the utils package to get the current joint 
        position and velocity and self._limb.set_joint_velocities() to set the joint velocity to something.  
        You may find joint_array_to_dict() in utils.py useful as well.

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        
        # control_input = None
        # self._limb.set_joint_velocities(joint_array_to_dict(control_input, self._limb))
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)
        
        # TODO: implement PID control to set the joint velocities. 
        error = (target_position - current_position)
        error_dot = (target_velocity - current_velocity)
        prop = np.dot(self.Kp, error)

        derv = np.dot(self.Kv, error_dot)

        controller_velocity = target_velocity + prop + derv

        self._limb.set_joint_velocities(joint_array_to_dict(controller_velocity, self._limb))

class PDJointTorqueController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time. This method should call
        get_joint_positions and get_joint_velocities from the utils package to get the current joint 
        position and velocity and self._limb.set_joint_torques() to set the joint torques to something. 
        You may find joint_array_to_dict() in utils.py useful as well.
        Recall that in order to implement a torque based controller you will need access to the 
        dynamics matrices M, C, G such that
        M ddq + C dq + G = u
        For this project, you will access the inertia matrix and gravity vector as follows:
        Inertia matrix: self._kin.inertia(positions_dict)
        Coriolis matrix: self._kin.coriolis(positions_dict, velocity_dict)
        Gravity matrix: self._kin.gravity(positions_dict) (You might want to scale this matrix by 0.01 or another scalar)
        These matrices were computed by a library and the coriolis matrix is approximate, 
        so you should think about what this means for the kinds of trajectories this 
        controller will be able to successfully track.
        Look in section 4.5 of MLS.
        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        raise NotImplementedError
        control_input = None
        self._limb.set_joint_torques(joint_array_to_dict(control_input, self._limb))
