# Model design
import os
import random

import agentpy as ap
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mp
import webbrowser

mp.rcParams['animation.embed_limit'] = 2 ** 25


def euclidean_norm(v):
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


class Sheep(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    which follows the Social Force Model for herding. """

    def setup(self):

        self.velocity = np.zeros(2)

    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos

        # Social Force (Repelled when close, attracted when far)
        # rho_s: minimal sheep-to-sheep safety distance
        # rho_r: maximal action distance of the repulsive effect between two sheep
        # rho_g: minimal action distance of the attractive effect between two sheep

        # Zone 0 (physical sheep)
        nbs_z0 = self.neighbors(self, distance=self.p.rho_s).to_list()

        # Zone 1 (repel)
        nbs_z1 = self.neighbors(self, distance=self.p.rho_r).to_list()

        # Zone 2 (neutral)
        nbs_z2 = self.neighbors(self, distance=self.p.rho_g).to_list()

        # Zone 3 (attract)
        nbs_z3 = self.neighbors(self, distance=self.p.rho_pv).to_list()

        # # Look for sheepdog
        # v_qi = np.zeros(2)
        # for agent in nbs_z3:
        #     if agent.type == "Dog":
        #         v_qi = self.p.alpha *


        # Get lists of Agent IDs to exclude zone 0 from zone 1
        zone_0_ids = [x.id for x in nbs_z0]
        zone_1_ids = [x.id for x in nbs_z1]
        zone_1_exclusive_ids = [x for x in zone_1_ids if x not in zone_0_ids]
        nbs_z1_revised = nbs_z1.select([True if x.id in zone_1_exclusive_ids else False for x in nbs_z1])

        # Get lists of Agent IDs to exclude zone 2 from zone 3
        zone_2_ids = [x.id for x in nbs_z2]
        zone_3_ids = [x.id for x in nbs_z3]
        zone_3_exclusive_ids = [x for x in zone_3_ids if x not in zone_2_ids]
        nbs_z3_revised = nbs_z3.select([True if x.id in zone_3_exclusive_ids else False for x in nbs_z3])

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

        # Noise vector
        nv = noise(self.p.speed_limit)

        # Update velocity using attractive, repulsive, boundary, and noise forces
        # self.velocity += v_pi_attractive + v4 + nv
        # self.velocity += v_pi_repulsive + v4 + nv
        velocity_vector = v_pi_attractive + v_pi_repulsive + border_repulsive + nv
        self.velocity += velocity_vector
        self.velocity = saturate(self.velocity, self.p.speed_limit)

    def update_position(self):
        self.space.move_by(self, self.velocity)


class Dog(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    which exerts a force on the sheep agents. """

    def setup(self):
        self.velocity = np.zeros(2)

    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos

        # Social Force (Repelled when close, attracted when far)
        # rho_s: minimal sheep-to-sheep safety distance
        # rho_r: maximal action distance of the repulsive effect between two sheep
        # rho_g: minimal action distance of the attractive effect between two sheep

        # Zone 0 (physical sheep)
        nbs_z0 = self.neighbors(self, distance=self.p.rho_s).to_list()

        # Zone 1 (repel)
        nbs_z1 = self.neighbors(self, distance=self.p.rho_r).to_list()

        # Zone 2 (neutral)
        nbs_z2 = self.neighbors(self, distance=self.p.rho_g).to_list()

        # Zone 3 (attract)
        nbs_z3 = self.neighbors(self, distance=self.p.rho_pv).to_list()

        # Get lists of Agent IDs to exclude zone 0 from zone 1
        zone_0_ids = [x.id for x in nbs_z0]
        zone_1_ids = [x.id for x in nbs_z1]
        zone_1_exclusive_ids = [x for x in zone_1_ids if x not in zone_0_ids]
        nbs_z1_revised = nbs_z1.select([True if x.id in zone_1_exclusive_ids else False for x in nbs_z1])

        # Get lists of Agent IDs to exclude zone 2 from zone 3
        zone_2_ids = [x.id for x in nbs_z2]
        zone_3_ids = [x.id for x in nbs_z3]
        zone_3_exclusive_ids = [x for x in zone_3_ids if x not in zone_2_ids]
        nbs_z3_revised = nbs_z3.select([True if x.id in zone_3_exclusive_ids else False for x in nbs_z3])

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

        # Noise vector
        nv = noise(self.p.speed_limit)

        # Update velocity using attractive, repulsive, boundary, and noise forces
        # self.velocity += v_pi_attractive + v4 + nv
        # self.velocity += v_pi_repulsive + v4 + nv
        velocity_vector = v_pi_attractive + v_pi_repulsive + border_repulsive + nv
        self.velocity += velocity_vector
        self.velocity = saturate(self.velocity, self.p.speed_limit)

    def update_position(self):
        self.space.move_by(self, self.velocity)


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
        self.agents = ap.AgentList(self, self.p.population, Sheep)
        np.random.seed(self.p.seed)
        #ap.AgentList.append()
        xPos = random.sample(range(0, 100), self.p.population)
        yPos = random.sample(range(25, 85), self.p.population)
        self.positionList = np.column_stack((xPos, yPos))
        #self.positionList = np.append(self.positionList, [agent_ta.setPos], axis=0)

        posList = [(float(x[0]), float(x[1])) for x in self.positionList]
        print("positionList: ", posList)
        self.space.add_agents(self.agents, positions=posList, random=False)
        self.agents.setup_pos(self.space)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction


def animation_plot_single(m, ax):
    ndim = 2
    ax.set_title(f"Sheep Herding Model 2D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.scatter(*pos, s=1, c='black')
    # updating each agent
    for agent in m.agents:
        if (agent.type == "Sheep"):
            # TODO: drawing boundaries for each zone ( zone1 repulsive, zone2 nuetral, zone3 attractive, zone4 nothing as out of sight)
            c0 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_s, edgecolor="black",
                            facecolor="black")  # size of sheep
            c1 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_r, edgecolor="red",
                            facecolor=(0, 0, 0, 0))  # border of zone 1 and zone 2
            c2 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_g, edgecolor="yellow",
                            facecolor=(0, 0, 0, 0))  # border of zone 2 and zone 3
            c3 = plt.Circle((agent.pos[0], agent.pos[1]), radius=agent.p.rho_pv, edgecolor="pink",
                            facecolor=(0, 0, 0, 0))  # border of zone 3 and 4
            ax.add_patch(c0)
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.add_patch(c3)

    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()


def animation_plot(m, p):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=None)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    return animation.to_jshtml(fps=30)


parameters2D = {
    'size': 100,
    'seed': 128,
    'steps': 200,
    'population': 24,
    'rho_s': 1.2,
    'rho_r': 2.2,
    'rho_g': 5.4,
    'rho_pv': 20,
    'gamma': 0.02,
    'beta': 5,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'separation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5,
    'speed_limit': 0.1
}

html = animation_plot(SheepModel, parameters2D)

path = os.path.abspath('anim.html')
url = 'file://' + path

with open(path, 'w') as f:
    f.write(html)
webbrowser.open(url)
