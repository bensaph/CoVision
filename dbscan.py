import numpy as np
import random
from tkinter import *
from enum import Enum

# https://en.wikipedia.org/wiki/DBSCAN
class Label(Enum):
    unlabeled = 0 # not processed
    core = 1
    reachable = 2 # "border points"
    noise = 3 # outliers (not part of cluster)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = Label.unlabeled
        self.cluster = -1 # indicates noise

    # distance formula
    def distance(self, other):
        return np.sqrt(np.power(self.x - other.x, 2) + np.power(self.y - other.y, 2))

# adjustable constants
dim = 500
x_dim = dim * 1
y_dim = dim * 1
point_radius = dim * 0.05

pts_size = 50 # number of entities
minPts = 3 # number of neighbors within epsilon to be considered a core point
epsilon = point_radius * 2 # within epsilon if two circles overlap

def main():
    # initialize starting positions
    pts = [Point(
        random.random() * (x_dim - 2 * point_radius) + point_radius,
        random.random() * (y_dim - 2 * point_radius) + point_radius)
        for i in range(pts_size)]

    dbscan(pts, minPts, epsilon)

    master = Tk()
    w = Canvas(master, width=x_dim, height=y_dim)
    w.pack()

    clusters = []
    for p in pts:
        w.create_oval(
            p.x - point_radius,
            p.y - point_radius,
            p.x + point_radius,
            p.y + point_radius,
            outline = get_color(p.label)
        )
        print('{}\t{} {}'.format(p.label, "cluster:", p.cluster))
    #print('{}{}'.format("minPts:", minPts))
    #print('{}{}'.format("epsilon:", epsilon))

    w.mainloop()

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