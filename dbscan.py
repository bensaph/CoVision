import numpy as np
import random
from tkinter import *
from enum import Enum

# adjustable constants

# for drawing
dim = 900
x_dim = dim * 1
y_dim = dim * 1
point_radius = dim * 0.05

# for dbscan
pts_size = 50 # number of entities
minPts = 3 # number of neighbors within epsilon to be considered a core point
epsilon = point_radius * 2 # within epsilon if two circles overlap

# for movement (after one timestep)
lin_speed_min = dim * 0.005 # linear speed
lin_speed_range = dim * 0.01 # randomized lin speed between min and min+range
ang_speed_range = (np.pi * 2) * 0.25 # range of possible rotation

# https://en.wikipedia.org/wiki/DBSCAN
class Label(Enum):
    unlabeled = 0 # not processed
    core = 1
    reachable = 2 # "border points"
    noise = 3 # outliers (not part of cluster)

class Point:
    def __init__(self, x=0.0, y=0.0, dir=0.0, master=None, canvas=None):
        self.master = master
        self.canvas = canvas
        self.canvas.pack()
        self.x = random.random() * (x_dim - 2 * point_radius) + point_radius
        self.y = random.random() * (y_dim - 2 * point_radius) + point_radius
        self.dir = random.random() * (np.pi * 2)
        self.label = Label.unlabeled
        self.cluster = -1 # indicates noise

        self.point = self.canvas.create_oval(
            self.x - point_radius,
            self.y - point_radius,
            self.x + point_radius,
            self.y + point_radius,
            outline = get_color(self.label)
        )

    def movement(self):
        speed = (random.random() * lin_speed_range) + lin_speed_min
        angle = (random.random() * ang_speed_range * 2) - ang_speed_range
        dx = speed * np.cos(self.dir)
        dy = speed * np.sin(self.dir)
        self.x += dx
        self.y += dy
        self.dir += angle
        self.canvas.move(self.point, dx, dy)

    # distance formula
    def distance(self, other):
        return np.sqrt(np.power(self.x - other.x, 2) + np.power(self.y - other.y, 2))

def main():
    master = Tk()
    canvas = Canvas(master, width=x_dim, height=y_dim)

    # initialize starting positions
    pts = [Point(master=master, canvas=canvas) for i in range(pts_size)]

    moveAll(pts)
    master.bind("<space>", lambda e: moveAll(pts)) # press space to step through
    canvas.mainloop()

def moveAll(pts):
    # move to next position
    for p in pts:
        p.movement()

    # recalculate clusters
    for p in pts:
        p.label = Label.unlabeled
        p.cluster = -1
    dbscan(pts, minPts, epsilon)

    # recolor
    for p in pts:
        p.canvas.itemconfig(p.point, outline=get_color(p.label))

# called every timestep
def dbscan(pts, minPts=3, epsilon=point_radius*2):
    c = 0 # cluster counter
    for p in pts:
        if p.label is not Label.unlabeled:
            continue
        n = within_range(pts, p, epsilon)
        if len(n) < minPts:
            p.label = Label.noise # initialize as noise, can change later
            continue

        p.cluster = c
        p.label = Label.core

        # classifies a core node's neighbors as either
        # another core node and adds its neighbors to the set
        # or as a reachable node
        n = n.difference({p})
        while len(n) > 0:
            q = n.pop()
            if q.label is Label.noise:
                q.label = Label.reachable
                q.cluster = c
            if q.label is not Label.unlabeled:
                continue

            q.cluster = c
            q.label = Label.reachable

            nq = within_range(pts, q, epsilon)
            if len(nq) >= minPts:
                q.label = Label.core
                n = n.union(nq)
        c += 1

# returns set of Points within range of epsilon including itself
def within_range(pts, p_org, epsilon):
    n = set()
    for p in pts:
        if p_org.distance(p) <= epsilon:
            n.add(p)
    return n

# probably a better way to do this but not important
def get_color(label):
    if label is Label.noise:
        return "orange"
    if label is Label.reachable:
        return "red"
    if label is Label.core:
        return "black"
    if label is Label.unlabeled: # shouldn't be any
        return "green"

if __name__ == "__main__":
    main()