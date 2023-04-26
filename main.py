# Model design
import os
import random

import agentpy as ap
import numpy as np
import pandas as pd
# Visualization
import matplotlib.pyplot as plt
import matplotlib as mp
import webbrowser

mp.rcParams['animation.embed_limit'] = 2 ** 25
np.random.seed(123)
random.seed(123)


def euclidean_norm(v, keep_zero=False):
    """ Calculate euclidean norm of a 2D vector.
    ||v|| = sqrt(x^2 + y^2)"""
    norm = np.linalg.norm(v)
    return norm


def unit_vector(v):
    """ Return the unit directional vector of a 2D vector. """
    norm = euclidean_norm(v)
    if norm != 0:
        return v / norm
    else:
        return np.zeros(2)


def rotation(theta):
    """ Planar rotation matrix R """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def angle(x, y):
    """ Finds angle between vectors x and y """
    return np.arccos(np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def stack_overflow_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def noise(xi_p):
    """ Generate noise for sheep i at time step k, where xi_p is
    the upper limit for the speed of the sheep """
    mi = np.random.uniform()  # uniform random between 0 and 1
    pi = np.random.uniform()  # uniform random between 0 and 1

    constant = 1 / 4 * xi_p * mi
    vector = np.array([np.cos(2 * np.pi * pi), np.sin(2 * np.pi * pi)])

    return constant * vector


def saturate(v, xi_p):
    """ Saturation function where xi_p is the saturation threshold. """
    if euclidean_norm(v) <= xi_p:
        return v
    else:
        return xi_p * unit_vector(v)


def only_sheep(arr):
    return arr.select([True if agent.type == "Sheep" else False for agent in arr])


def only_agents(arr):
    return arr.select([True if agent.type == "Sheep" or agent.type == "Dog" else False for agent in arr])
def only_obstacle(arr):
    return arr.select([True if agent.type == "Obstacle" else False for agent in arr])
def colliding_circle(x, y, r,timestep,agent):
    # Define the radius of the circle
    radius = r

    # Check if the circle intersects with any agents
    #for agent_x, agent_y in agents:
    agent_x = agent[0]
    agent_y = agent[1]
    if abs(x - agent_x)**2 + abs(y - agent_y)**2 < (radius + 10)**2:
        # The circle intersects with this agent,
        print("Sheep ", agent.id, " intersected with obs at timestep " , timestep )
        return True
    return False


class Dog(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    which follows the Social Force Model for herding. """

    def setup(self):

        # self.velocity = np.zeros(2)
        self.velocity = np.array([0.0, 0.0])
        self.state_flag = 1
        self.V_lf = None
        self.V_rf = None
        self.V_la = None
        self.V_ra = None

    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def dis_func(self, vision):
        dis_array = []
        for sheep in vision:
            dis_array.append(euclidean_norm(sheep.pos - self.p.centroid_sheep))

        return np.max(dis_array)

    def H(self, sheep, Dcdlr):
        return np.inner(sheep.pos - self.pos, Dcdlr)

    def F(self, sheep, Dqc):
        return np.inner(sheep.pos - self.pos, Dqc)

    def G(self, sheep):
        return np.linalg.norm(sheep.pos - self.pos)

    def V_front(self, sheep_list, Dcdlr):
        # if(len(sheep_list))<= 0:
        #    return None
        H_distances = []
        for sheep in sheep_list:
            H_distances.append(self.H(sheep, Dcdlr))

        max_index = np.argmax(H_distances)
        return sheep_list[max_index]

    def P_left_front(self, sheep_list):
        G_distances = []
        for sheep in sheep_list:
            G_distances.append(self.G(sheep))

        max_index = np.argmax(G_distances)
        return sheep_list[max_index]

    def V_anchor(self, sheep_list, Dqc):
        # if(len(sheep_list))<= 0:
        #    return None
        F_distances = []
        for sheep in sheep_list:
            F_distances.append(self.F(sheep, Dqc))

        min_index = np.argmin(F_distances)
        return sheep_list[min_index]

    def left_right(self, position, vision, pdf=None):
        subposition = self.p.destination_center
        if pdf is not None:
            subposition = pdf

        A = unit_vector(position - subposition)
        C = -1 * A
        # Checking the left
        left_check = True
        right_check = True

        for sheep in vision:
            B = unit_vector(sheep.pos - subposition)
            if np.cross(A, B) >= 0 and np.cross(C, B) < 0:
                left_check = False
                sheep.color = 'teal'  # right
            elif np.cross(A, B) < 0 and np.cross(C, B) >= 0:
                right_check = False
                sheep.color = 'lavender'  # left
            else:
                "Oops"
        return left_check, right_check

    def getLeft_RightSheep(self, vision, A):
        leftSheepList = []
        rightSheepList = []

        # A = unit_vector(position - self.p.destination_center)
        C = -1 * A

        for sheep in vision:
            B = unit_vector(sheep.pos - self.p.centroid_sheep)
            if np.cross(A, B) >= 0 and np.cross(C, B) < 0:

                rightSheepList.append(sheep)
            elif np.cross(A, B) < 0 and np.cross(C, B) >= 0:

                leftSheepList.append(sheep)
            else:
                "Oops"

        if (len(leftSheepList) <= 0) or (len(rightSheepList) <= 0):
            print("Error !! no sheep in left or right of sheep centroid!!")
        return leftSheepList, rightSheepList

    def update_velocity(self):

        self.p.time_step += 1
        pos = self.pos

        for sheep in self.space.agents:
            sheep.color = 'aqua'
            sheep.critical = [0, 0, 0, 0]

        # Social Force (Repelled when close, attracted when far)
        # rho_s: minimal sheep-to-sheep safety distance
        # rho_r: maximal action distance of the repulsive effect between two sheep
        # rho_g: minimal action distance of the attractive effect between two sheep

        # Sheepdog vision
        vision = self.neighbors(self, distance=self.p.rho_vq).to_list()

        # Calculating centroid of visible sheep (p^c(k))
        centroid_list = []
        for sheep in vision:
            centroid_list.append(sheep.pos)

        self.p.centroid_sheep = np.mean(centroid_list, axis=0)
        Dcd = unit_vector(np.array(self.p.destination_center) - self.p.centroid_sheep)
        Dcdl = np.matmul(rotation(np.pi / 2), Dcd)
        Dcdr = np.matmul(rotation(-np.pi / 2), Dcd)
        Dqc = unit_vector(self.p.centroid_sheep - pos)

        # Calculate dynamic far-end
        pdf = None
        if self.p.algorithm == 2 or self.p.algorithm == 4:
            pdf = self.p.destination_center + (self.p.destination_radius * Dcd)

        left_check, right_check = self.left_right(pos, vision, pdf)
        # Calculating critical sheep (V_lf = P_lf under the assumption V_lf is always singleton)
        self.V_lf = self.V_front(vision, Dcdl)
        self.V_rf = self.V_front(vision, Dcdr)

        # if self.V_lf != None:
        self.V_lf.critical[0] = 1
        self.V_rf.critical[1] = 1

        leftSheepList, rightSheepList = self.getLeft_RightSheep(vision, Dqc)

        self.V_la = self.V_anchor(leftSheepList, Dqc)
        self.V_ra = self.V_anchor(rightSheepList, Dqc)

        self.V_la.critical[2] = 1
        self.V_ra.critical[3] = 1

        theta_lt = angle(Dcd, pos - self.V_lf.pos)
        theta_rt = angle(Dcd, pos - self.V_rf.pos)

        # Pausing mechanism
        if self.p.algorithm == 3 or self.p.algorithm == 4:
            theta_b = angle(Dcd, Dqc)
            r = euclidean_norm(pos - self.p.centroid_sheep) / self.dis_func(vision)

        # Algorithm 1 START
        # u_p: velocity of sheepdog
        u_p = np.zeros(2)
        if left_check and theta_lt < self.p.theta_t:
            #print("Should turn left here - ", self.p.time_step)
            self.state_flag = 1
            u_p = self.p.velocity_coefficient * np.matmul(rotation(- self.p.detouring_theta), (self.V_ra.pos - pos))
        elif right_check and theta_rt < self.p.theta_t:
            #print("Should turn right here - ", self.p.time_step)
            self.state_flag = -1
            u_p = self.p.velocity_coefficient * np.matmul(rotation(self.p.detouring_theta), (self.V_la.pos - pos))
        elif self.state_flag == 1:
            if self.p.algorithm == 3 or self.p.algorithm == 4:
                if theta_b < self.p.theta_b and r < self.p.r_underbar:
                    #print("Should be pausing here - ", self.p.time_step)
                    u_p = -self.velocity
                elif r > self.p.r_overbar:
                    u_p = self.p.velocity_coefficient * (self.p.centroid_sheep - pos)
                else:
                    u_p = self.p.velocity_coefficient * np.matmul(rotation(- self.p.detouring_theta), (self.V_ra.pos - pos))
            else:
                u_p = self.p.velocity_coefficient * np.matmul(rotation(- self.p.detouring_theta), (self.V_ra.pos - pos))
        else:
            if self.p.algorithm == 3 or self.p.algorithm == 4:
                if theta_b < self.p.theta_b and r < self.p.r_underbar:
                    #print("Should be pausing here - ", self.p.time_step)
                    u_p = -self.velocity
                elif r > self.p.r_overbar:
                    u_p = self.p.velocity_coefficient * (self.p.centroid_sheep - pos)
                else:
                    u_p = self.p.velocity_coefficient * np.matmul(rotation(self.p.detouring_theta), (self.V_la.pos - pos))
            else:
                u_p = self.p.velocity_coefficient * np.matmul(rotation(self.p.detouring_theta), (self.V_la.pos - pos))

        # if left_check:
        #     print("All on left, time step: ", self.p.time_step)
        # elif right_check:
        #     print("All on right, time step: ", self.p.time_step)
        # else:
        #     print("Mixed between right and left, time step: ", self.p.time_step)

        # Rule 4 - Borders
        border_repulsive = np.zeros(2)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(2):
            if pos[i] < d:
                border_repulsive[i] += s
            elif pos[i] > self.space.shape[i] - d:
                border_repulsive[i] -= s
        # Noise vector
        # nv = noise(self.p.speed_limit)

        # Update velocity using attractive, repulsive, boundary, and noise forces
        # self.velocity += v_pi_attractive + v4 + nv
        # self.velocity += v_pi_repulsive + v4 + nv
        velocity_vector = border_repulsive + u_p
        self.velocity += velocity_vector
        self.velocity = saturate(self.velocity, self.p.speed_limit_d)

    def update_position(self):
        self.space.move_by(self, self.velocity)


class Sheep(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    which exerts a force on the sheep agents. """

    def setup(self):
        self.velocity = np.zeros(2)
        self.color = 'aqua'
        self.critical = [0, 0, 0, 0]

    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]
        self.prevPos = []

    def update_velocity(self):

        pos = self.pos

        # Social Force (Repelled when close, attracted when far)

        # rho_s: minimal sheep-to-sheep safety distance
        # rho_r: maximal action distance of the repulsive effect between two sheep
        # rho_g: minimal action distance of the attractive effect between two sheep

        # Zone 0 (physical sheep)
        nbs_z0 = self.neighbors(self, distance=self.p.rho_s).to_list()
        nbs_z0 = only_sheep(nbs_z0)

        # Zone 1 (repel)
        nbs_z1 = self.neighbors(self, distance=self.p.rho_r).to_list()
        nbs_z1 = only_sheep(nbs_z1)

        # Zone 2 (neutral)
        nbs_z2 = self.neighbors(self, distance=self.p.rho_g).to_list()
        nbs_z2 = only_sheep(nbs_z2)

        # Zone 3 (attract)
        nbs_z3 = self.neighbors(self, distance=self.p.rho_vp).to_list()
        nbs_z3 = only_agents(nbs_z3)

        nbs_z3_sheep = only_sheep(nbs_z3)

        # Look for sheepdog
        v_qi = np.zeros(2)
        a_i = 1
        for agent in nbs_z3:
            if agent.type == "Dog":
                p_i = pos - agent.pos
                norm_p_i = euclidean_norm(p_i)
                phi = np.abs(self.p.alpha * (self.p.rho_vp - norm_p_i) / (norm_p_i - self.p.rho_x))
                v_qi = a_i * phi * unit_vector(p_i)

        # Get lists of Agent IDs to exclude zone 0 from zone 1
        zone_0_ids = [x.id for x in nbs_z0]
        zone_1_ids = [x.id for x in nbs_z1]
        zone_1_exclusive_ids = [x for x in zone_1_ids if x not in zone_0_ids]
        nbs_z1_revised = nbs_z1.select([True if x.id in zone_1_exclusive_ids else False for x in nbs_z1])

        # Get lists of Agent IDs to exclude zone 2 from zone 3
        zone_2_ids = [x.id for x in nbs_z2]
        zone_3_ids = [x.id for x in nbs_z3_sheep]
        zone_3_exclusive_ids = [x for x in zone_3_ids if x not in zone_2_ids]
        nbs_z3_revised = nbs_z3_sheep.select([True if x.id in zone_3_exclusive_ids else False for x in nbs_z3_sheep])

        v_pi_attractive = np.zeros(2)
        for sheep in nbs_z3_revised:
            p_ij = sheep.pos - pos
            phi = self.p.gamma * np.sqrt(euclidean_norm(p_ij) - self.p.rho_g)
            v_pi_attractive += phi * unit_vector(p_ij)

        v_pi_repulsive = np.zeros(2)
        for sheep in nbs_z1_revised:
            p_ij = pos - sheep.pos
            p_ij_norm = euclidean_norm(p_ij)
            phi = self.p.beta * (self.p.rho_r - p_ij_norm) / (p_ij_norm - self.p.rho_s)
            v_pi_repulsive += phi * unit_vector(p_ij)

        # Rule 4 - Borders
        border_repulsive = np.zeros(2)

        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(2):
            if pos[i] < d:
                border_repulsive[i] += s
            elif pos[i] > self.space.shape[i] - d:
                border_repulsive[i] -= s
        
        # Rule 5 - Obstacle
        obstacle_repulsive = np.zeros(2)
        if(self.p.obstacle !=0):
            d = self.p.obstacle_distance
            s = self.p.obstacle_strength


            obstacle_repulsive = .3 * unit_vector(pos - self.p.obstacle_center)
            #for i in range(2):
                
             #   if pos[i] < d:
             #       obstacle_repulsive[i] += s
             #   elif pos[i] > self.p.obstacle_center[i] - d:
             #       obstacle_repulsive[i] -= s

        # Noise vector
        nv = noise(self.p.speed_limit)

        # Update velocity using attractive, repulsive, boundary, and noise forces
        velocity_vector = v_pi_attractive + v_pi_repulsive + border_repulsive + v_qi  + nv + obstacle_repulsive
        self.velocity += velocity_vector
        self.velocity = saturate(self.velocity, self.p.speed_limit)

    def update_position(self):
        self.prevPos = self.pos
        self.space.move_by(self, self.velocity)

        #check if it is in obstacle move to previous step 
        #get obstacle
        colliding = False
        allAgents = self.neighbors(self, distance=500).to_list()
        agent_obs = only_obstacle(allAgents)

        #obstacle difference
        """
        #if(self.p.obstacle == 1): #rectangle
        if(self.p.obstacle == 2): #circle
            #print(self.p.obstacle_center[0],self.p.obstacle_center[1], agent_obs.radius, self.p.time_step, self.pos)

            print(type(agent_obs))
            agentsInObstacleBoundary = self.neighbors(agent_obs,distance = agent_obs.radius)

            #.to_list()
            print("here")
            for agent in agentsInObstacleBoundary:
                if agent.id == self.id:
                    colliding == True
                    break
            #colliding = colliding_circle(self.p.obstacle_center[0],self.p.obstacle_center[1], agent_obs.radius, self.p.time_step, self.pos)
        #elif(self.p.obstacle == 3): #triangle
        """
        if(colliding):
            print("Sheep ", self.id, " intersected with obstacle at timestep " , self.p.timestep )
            self.space.move_to(self, self.prevPos)

class TargetArea(ap.Agent):
    def setup(self, **kwargs):
        self.type = "TargetArea"
        self.radius = self.p.destination_radius
        self.color = "green"
        # self.setPos = [50, 87]
        # self.velocity
        return super().setup(**kwargs)

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def drawTargetArea(self, ax):
        c = plt.Circle((self.pos[0], self.pos[1]), radius=self.radius, edgecolor=self.color,
                       facecolor=(0.5, 0.5, 0.5))  # border of zone 3 and 4
        ax.add_patch(c)
        return ax
    # def update_velocity(self):
    #    return
    # def update_positions(self):
    #    return

class Obstacles(ap.Agent):
    def setup(self,**kwargs):
        self.type="Obstacle"
        self.points =[] #list of (x,y) pts
        self.radius = -1 # radius if applicable
        return super().setup(**kwargs)
    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

class SheepModel(ap.Model):
    """
    An agent-based model of sheep herding behavior,
    based on Social Force Model [1]
    and Cai et. al [2].

    [1] https://ieeexplore.ieee.org/document/5206641
    [2] https://www.mdpi.com/2079-9292/12/2/285
    """

    def setup(self):
        """ Initializes the agents and network of the model. """

        self.space = ap.Space(self, shape=[self.p.size] * 2)
        self.agents = ap.AgentList(self, self.p.population, Sheep)  # add sheeps

        agent_ta = TargetArea(self)
        self.agents.insert(0, agent_ta)  # add target area
        # create position list using random generated numbers within  area
        # for x (0,100) 
        # for y (25,85)

        agent_dog = Dog(self)
        self.agents.insert(1, agent_dog)  # Add dog

        #obstacle creation
        obstaclesDict = {0:'None',1:[ (0,0), (0, 100), (100, 100), (100, 0)],2:4,3:[ (0, 100), (100, 100), (100, 0)]} #obstacles 0:'None',1:'rectangle',2:'circle',3:'trianlge'
        if(self.p.obstacle != 0):#adding obstacle
            agent_obstacle = Obstacles(self)
            self.agents.insert(2,agent_obstacle) 

            #TODO: add coordinates of obs
            if(self.p.obstacle == 1): #rectangle
                agent_obstacle.points= obstaclesDict[1]
            elif(self.p.obstacle == 2): #circle
                agent_obstacle.radius= obstaclesDict[2]
            elif(self.p.obstacle == 3): #triangle
                agent_obstacle.points= obstaclesDict[3]

        xPos = random.sample(range(35, 65), self.p.population)
        yPos = random.sample(range(25, 55), self.p.population)
        xPos = [float(x) for x in xPos]
        yPos = [float(y) for y in yPos]
        self.positionList = np.column_stack((xPos, yPos))
        agentListFront = np.append([self.p.destination_center], [(50.0, 5.0)], axis=0)
        if(self.p.obstacle != 0):
            agentListF_obstacle = np.append(agentListFront, [self.p.obstacle_center], axis =0)
            self.positionList = np.append(agentListF_obstacle, self.positionList, axis=0)
        else:
            self.positionList = np.append(agentListFront, self.positionList, axis=0)

        posList = [(x[0], x[1]) for x in self.positionList]
        #print("positionList: ", posList)
        self.space.add_agents(self.agents, positions=posList, random=False)
        self.agents.setup_pos(self.space)

    def step(self):
        """ Defines the models' events per simulation step. """
        agentList = only_sheep(self.agents)
        agentList.append(self.agents[1])
        agentList.update_velocity()  # Adjust direction
        agentList.update_position()  # Move into new direction

    def update(self):
        """ Updates after every time step """
        sheep_herded = only_sheep(self.agents[0].neighbors(self.agents[0], self.p.destination_radius).to_list())
        self.record('percentage', len(sheep_herded) / self.p.population)


def animation_plot_single(m, ax):
    ndim = 2
    ax.set_title(f"Sheep Herding Model (Algorithm {m.p.algorithm}) t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.scatter(*pos, s=1, c='black')
    # centroid = plt.Circle((m.p.centroid_sheep[0], m.p.centroid_sheep[1]), radius=5, edgecolor="pink",
    #                       facecolor="pink")  # target area
    # ax.add_patch(centroid)
    critical_text = {
        0: "Plf",
        1: "Prf",
        2: "Pla",
        3: "Pra"
    }
    # updating each agent
    for agent in m.agents:
        if agent.type == "Sheep":
            # TODO: drawing boundaries for each zone ( zone1 repulsive, zone2 nuetral, zone3 attractive, zone4 nothing as out of sight)
            for (i, elem) in enumerate(agent.critical):
                if elem == 1:
                    agent.color = 'red'
                    ax.text(agent.pos[0], agent.pos[1] - 1.0, critical_text[i])
            c0 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_s - 0.6, edgecolor=agent.color,
                            facecolor=agent.color)  # size of sheep
            # c1 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_r, edgecolor="red",
            #                 facecolor=(0, 0, 0, 0))  # border of zone 1 and zone 2
            # c2 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_g, edgecolor="yellow",
            #                 facecolor=(0, 0, 0, 0))  # border of zone 2 and zone 3
            # c3 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_vp, edgecolor="pink",
            #                 facecolor=(0, 0, 0, 0))  # border of zone 3 and 4
            ax.add_patch(c0)
            # ax.add_patch(c1)
            # ax.add_patch(c2)
            # ax.add_patch(c3)
            ax.text(agent.pos[0], agent.pos[1], str(agent.id), ha='center')
        if agent.type == "TargetArea":
            c = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.radius, edgecolor=agent.color,facecolor=(0.5, 0.5, 0.5))  # target area
            ax.add_patch(c)
        if agent.type == "Dog":
            c = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_s - 0.6, edgecolor="black",facecolor="blue")  # size of sheep
            ax.add_patch(c)
        if agent.type == "Obstacle":
            #if(m.p.obstacle == 1): #rectangle
            if(m.p.obstacle == 2): #circle
                c_o = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.radius, edgecolor="yellow",facecolor="yellow") 
                ax.add_patch(c_o) 
            #elif(m.p.obstacle == 3): #triangle

    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    ax.set_axis_off()


def animation_plot(m, p):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=None)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    return animation.to_jshtml(fps=30)


parameters = {
    'size': 100,
    'seed': 128,
    'steps': 1000,
    'population': 24,
    'rho_x': 1.5,
    'rho_s': 1.2,
    'rho_r': 2.2,
    'rho_g': 5.4,
    'rho_vp': 20.0,
    'rho_vq': 30.0,
    'alpha': 0.7,
    'beta': 5.0,
    'gamma': 0.02,
    'velocity_coefficient': 0.6,
    'detouring_theta': ((.6 * 360) / np.pi),
    'theta_t': 2.5,
    'theta_b': 0.1,
    'r_underbar': 10.0/7.0,
    'r_overbar': 4.0,
    'destination_center': (50.0, 87.0),  # p_d
    'destination_radius': 10.0,  # rho_d
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'separation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5,
    'speed_limit': 0.1,
    'centroid_sheep': (0.0, 0.0),
    'speed_limit_d': 5.0 / 30.0,
    'time_step': 0,
    'algorithm': 4,
    'obstacle':2,
    'obstacle_center': (50.0,63.0),
    'obstacle_distance': .5,
    'obstacle_strength': 5.0
}

runTrials = False
if runTrials:
    num_trials = 3

    results_dict = {"Trial": [],
                    "Success": [],
                    "Highest Percentage": [],
                    "Highest Percentage Time Step": [],
                    "Time Steps to Success": []}
    df1 = None
    df2 = None
    df3 = None
    df4 = None
    for algorithm in range(1, 5):
        parameters['algorithm'] = algorithm
        print(f"Testing Algorithm {algorithm}")
        for trial in range(num_trials):
            # Init and run model
            model = SheepModel(parameters)
            results = model.run()

            max_perc_row = results.variables.SheepModel.query('percentage == percentage.max()')
            time_step = max_perc_row.iloc[[0]].index[0]
            max_perc = max_perc_row.loc[time_step]['percentage']

            results_dict["Trial"].append(trial + 1)
            if max_perc == 1.0:
                results_dict["Success"].append(True)
                results_dict["Time Steps to Success"].append(time_step)
            else:
                results_dict["Success"].append(False)
                results_dict["Time Steps to Success"].append(-1)
            results_dict["Highest Percentage"].append(max_perc)
            results_dict["Highest Percentage Time Step"].append(time_step)

            parameters['seed'] += 1

        if algorithm == 1:
            df1 = pd.DataFrame(results_dict)
        elif algorithm == 2:
            df2 = pd.DataFrame(results_dict)
        elif algorithm == 3:
            df3 = pd.DataFrame(results_dict)
        elif algorithm == 4:
            df4 = pd.DataFrame(results_dict)
        else:
            raise Exception("No algorithms beyond algorithm 4")

        parameters['seed'] = 1
        parameters['time_step'] = 0
        parameters['centroid_sheep'] = (0.0, 0.0)

        results_dict = {"Trial": [],
                        "Success": [],
                        "Highest Percentage": [],
                        "Highest Percentage Time Step": [],
                        "Time Steps to Success": []}


    with pd.ExcelWriter('results.xlsx') as writer:
        df1.to_excel(writer, sheet_name=f"Algorithm 1")
        df2.to_excel(writer, sheet_name=f"Algorithm 2")
        df3.to_excel(writer, sheet_name=f"Algorithm 3")
        df4.to_excel(writer, sheet_name=f"Algorithm 4")

else:
    model = SheepModel(parameters)
    results = model.run()

    html = animation_plot(SheepModel, parameters)

    path = os.path.abspath('anim.html')
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(html)
    webbrowser.open(url)
