#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

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

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
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
        self.desired_orientation = np.array([-(2**.5)/2, (2**.5)/2, 0, 0])

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