# Model design
import os

import agentpy as ap
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mp
import webbrowser

mp.rcParams['animation.embed_limit'] = 2**25

def normalize(v):
    """ Normalize a vector to length 1. """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Sheep(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds three rules of flocking behavior;
    plus a fourth rule to avoid the edges of the simulation space. """

    def setup(self):

        # self.velocity = normalize(
        #     self.model.nprandom.random(self.p.ndim) - 0.5)
        self.velocity = normalize(np.array([0.0, 0.0]))
        self.type = "Sheep"
    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos
        ndim = self.p.ndim

        # Social Force (Repelled when close, attracted when far)
        # 4 zones (repel, neutral, attract, neutral), only first 3 are calculated

        # Zone 2 (neutral)
        nbs_z2 = self.neighbors(self, distance=self.p.radius_two).to_list()

        # Zone 3 (attract)
        nbs_z3 = self.neighbors(self, distance=self.p.radius_three).to_list()

        # Zone 3 should exclude all agents from zones 1 and 2
        # Zone 2 will do nothing
        # Zone 1 will repel

        # Get lists of Agent IDs to exclude zone 2 from zone 3
        zone_2_ids = [x.id for x in nbs_z2]
        zone_3_ids = [x.id for x in nbs_z3]
        zone_3_exclusive_ids = [x for x in zone_3_ids if x not in zone_2_ids]
        nbs_z3_revised = nbs_z3.select([True if x.id in zone_3_exclusive_ids else False for x in nbs_z3])

        # Calculate center point of attraction for zone 3
        nbs_len = len(nbs_z3_revised)
        nbs_pos_array = np.array(nbs_z3.pos)
        nbs_vec_array = np.array(nbs_z3.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # # Calculate center point of repulsion for zone 1
        # nbs_len = len(nbs_z1)
        # nbs_pos_array = np.array(nbs_z1.pos)
        # nbs_vec_array = np.array(nbs_z1.velocity)
        # if nbs_len > 0:
        #     center = np.sum(nbs_pos_array, 0) / nbs_len
        #     v2 = (pos - center) * self.p.cohesion_strength
        # else:
        #     v2 = np.zeros(ndim)

        # # Rule 1 - Cohesion
        # nbs = self.neighbors(self, distance=self.p.outer_radius)
        # nbs_len = len(nbs)
        # nbs_pos_array = np.array(nbs.pos)
        # nbs_vec_array = np.array(nbs.velocity)
        # if nbs_len > 0:
        #     center = np.sum(nbs_pos_array, 0) / nbs_len
        #     v1 = (center - pos) * self.p.cohesion_strength
        # else:
        #     v1 = np.zeros(ndim)

        # Rule 2 - Separation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.radius_one):
            v2 -= nb.pos - pos
        v2 *= self.p.separation_strength

        # # Rule 3 - Alignment
        # if nbs_len > 0:
        #     average_v = np.sum(nbs_vec_array, 0) / nbs_len
        #     v3 = (average_v - self.velocity) * self.p.alignment_strength
        # else:
        #     v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v4
        self.velocity = normalize(self.velocity)

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

        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Sheep)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)
        

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction


def animation_plot_single(m, ax): # for a single stepp
    #print(m.type)
    ndim = m.p.ndim
    ax.set_title(f"Sheep Flocking Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.scatter(*pos, s=1, c='black')
    

    #updating each agent
    for agent in m.agents:
        if(agent.type == "Sheep"):
            #TODO: drawing boundaries for each zone ( zone1 repulsive, zone2 nuetral, zone3 attractive, zone4 nothing as out of sight)
            c1 = plt.Circle((agent.pos[0],agent.pos[1]),radius=agent.p.radius_one,edgecolor="red",facecolor=(0,0,0,0)) #border of zone 1 and zone 2 
            c2 = plt.Circle((agent.pos[0],agent.pos[1]),radius=agent.p.radius_two,edgecolor="yellow",facecolor=(0,0,0,0)) #border of zone 2 and zone 3
            c3 = plt.Circle((agent.pos[0],agent.pos[1]),radius=agent.p.radius_three,edgecolor="pink",facecolor=(0,0,0,0)) #border of zone 3 and 4
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.add_patch(c3)


    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    ax.set_axis_off()

def animation_plot(m, p):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=None)

    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    return animation.to_jshtml(fps=30)


parameters = {
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 2,
    'population': 50,
    'radius_one': 2,
    'radius_two': 5,
    'radius_three': 10,
    'border_distance': 10,
    'inner_radius': 3,   
    'outer_radius': 10,
    'cohesion_strength': 0.005,
    'separation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}


html = animation_plot(SheepModel, parameters)

path = os.path.abspath('anim.html')
url = 'file://' + path

with open(path, 'w') as f:
    f.write(html)
webbrowser.open(url)