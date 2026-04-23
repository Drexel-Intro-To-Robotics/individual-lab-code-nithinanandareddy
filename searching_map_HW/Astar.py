#!/usr/bin/env python3
"""
ME571 - Introduction to Robotic Technology
Homework 3 - A* Algorithm

Run: python astar.py <map_name> <heuristic>
Example: python astar.py trivial.gif euclidean
         python astar.py trivial.gif manhattan
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from queue import PriorityQueue

# ---------------------------------------------------------------------------
# Global variables - set at runtime
# ---------------------------------------------------------------------------
start      = (0, 0)
end        = (0, 0)
difficulty = ""
heuristic_name = "euclidean"   # "euclidean" or "manhattan"

# ---------------------------------------------------------------------------
# Algorithm data structures
# ---------------------------------------------------------------------------
path        = []
expanded    = {}
frontier    = {}
came_from   = {}
cost_so_far = {}

# ---------------------------------------------------------------------------
# PIL display colors
# ---------------------------------------------------------------------------
NEON_GREEN = (0,   255, 0)
PURPLE     = (85,  26,  139)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY  = (100, 100, 100)


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------
def heuristic_manhattan(a, b):
    """
    Manhattan distance: sum of absolute differences in x and y.
    Best for grids with only 4-directional movement.
    Never overestimates on a unit-cost grid -> admissible.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a, b):
    """
    Euclidean distance: straight-line distance between two points.
    Also admissible on a unit-cost grid since straight-line
    is always <= actual path length.
    Tends to expand fewer nodes than Manhattan in open areas.
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def heuristic(a, b):
    """Wrapper that calls the selected heuristic function."""
    if heuristic_name == "manhattan":
        return heuristic_manhattan(a, b)
    else:
        return heuristic_euclidean(a, b)


# ---------------------------------------------------------------------------
# Helper: valid 4-directional neighbors
# ---------------------------------------------------------------------------
def get_neighbors(node, map_data, width, height):
    """
    Returns passable 4-directional neighbors of a node.
    White pixels (True) = free space, Black (False) = wall.
    """
    x, y = node
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if map_data[nx, ny]:
                neighbors.append((nx, ny))
    return neighbors


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------
def search(map_data):
    """
    A* algorithm on a pixel grid.

    How it works:
      - Like Dijkstra but the priority in the queue is:
            f(n) = g(n) + h(n)
        where g(n) = actual cost from start to n  (same as Dijkstra)
              h(n) = heuristic estimate from n to goal
      - The heuristic guides the search toward the goal, so A* expands
        far fewer nodes than Dijkstra in most cases.
      - As long as h(n) never overestimates the true cost (admissible),
        A* still guarantees the optimal path.

    Fills: path, expanded, frontier, came_from, cost_so_far.
    """
    global path, expanded, frontier, came_from, cost_so_far

    im = Image.open(difficulty)
    width, height = im.size

    # Priority queue stores (f_cost, node)
    pq = PriorityQueue()
    pq.put((0, start))
    came_from[start]    = None
    cost_so_far[start]  = 0
    frontier[start]     = 0

    while not pq.empty():
        f_cost, current = pq.get()

        if current == end:
            print("Goal reached!")
            break

        # Skip stale queue entries
        g_current = cost_so_far.get(current, float('inf'))
        if f_cost > g_current + heuristic(current, end):
            continue

        # Expand this node
        expanded[current] = g_current
        frontier.pop(current, None)

        for neighbor in get_neighbors(current, map_data, width, height):
            new_g = cost_so_far[current] + 1   # step cost = 1

            if new_g < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_g
                came_from[neighbor]   = current
                # A* priority = g + h
                f_new = new_g + heuristic(neighbor, end)
                pq.put((f_new, neighbor))
                frontier[neighbor] = new_g

    # Reconstruct path
    if end in came_from:
        current = end
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
    else:
        print("No path found.")
        return

    print(f"  Heuristic used : {heuristic_name}")
    print(f"  Nodes expanded : {len(expanded)}")
    print(f"  Nodes in frontier : {len(frontier)}")
    print(f"  Path length (steps) : {len(path) - 1}")
    print(f"  Final path cost : {cost_so_far[end]}")


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------
def visualize_matplotlib(save_file="astar_plot.png"):
    """
    Overlays color-coded results on the maze using matplotlib.
    Blue = expanded, Yellow = frontier, Red = path, Green = start/end.
    """
    im        = Image.open(difficulty).convert("RGB")
    width, height = im.size
    img_array = np.array(im)

    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    for (x, y) in expanded:
        overlay[y, x] = [30, 144, 255, 160]   # blue - expanded

    for (x, y) in frontier:
        overlay[y, x] = [255, 215, 0, 200]    # yellow - frontier

    for (x, y) in path:
        overlay[y, x] = [220, 20, 60, 255]    # red - final path

    overlay[start[1], start[0]] = [0, 255, 0, 255]   # green - start
    overlay[end[1],   end[0]]   = [0, 255, 0, 255]   # green - end

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.imshow(overlay)

    ax.annotate("START", xy=(start[0], start[1]),
                xytext=(start[0] + 3, start[1] + 6),
                color="lime", fontsize=7, fontweight="bold")
    ax.annotate("END",   xy=(end[0], end[1]),
                xytext=(end[0]   + 3, end[1]   + 6),
                color="lime", fontsize=7, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#1E90FF", label=f"Expanded nodes  ({len(expanded)})"),
        mpatches.Patch(color="#FFD700", label=f"Frontier nodes  ({len(frontier)})"),
        mpatches.Patch(color="#DC143C",
                       label=f"Final path  (cost = {cost_so_far.get(end,'N/A')})"),
        mpatches.Patch(color="lime",    label="Start / End"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.9)

    map_name = difficulty.split("/")[-1].replace(".gif", "")
    ax.set_title(
        f"A* Algorithm  ({heuristic_name} heuristic)  —  {map_name} map\n"
        f"Path cost: {cost_so_far.get(end,'N/A')}   |   "
        f"Nodes expanded: {len(expanded)}",
        fontsize=11
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {save_file}")


# ---------------------------------------------------------------------------
# PIL visualization
# ---------------------------------------------------------------------------
def visualize_pil(save_file="astar_result.png"):
    """Pixel-level visualization saved as image."""
    im           = Image.open(difficulty).convert("RGB")
    pixel_access = im.load()

    for pixel in expanded:
        pixel_access[pixel[0], pixel[1]] = DARK_GRAY

    for pixel in frontier:
        pixel_access[pixel[0], pixel[1]] = LIGHT_GRAY

    for pixel in path:
        pixel_access[pixel[0], pixel[1]] = PURPLE

    pixel_access[start[0], start[1]] = NEON_GREEN
    pixel_access[end[0],   end[1]]   = NEON_GREEN

    im.save(save_file)
    print(f"  Saved PIL image to {save_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python astar.py <map_name> <heuristic>")
        print("Example: python astar.py trivial.gif euclidean")
        print("         python astar.py trivial.gif manhattan")
        sys.exit(1)

    map_name       = sys.argv[1]
    heuristic_name = sys.argv[2].lower()
    difficulty     = "maps/" + map_name

    if heuristic_name not in ("euclidean", "manhattan"):
        print("Heuristic must be 'euclidean' or 'manhattan'")
        sys.exit(1)

    if map_name == "trivial.gif":
        start = (8,   1);   end = (20,  1)
    elif map_name == "medium.gif":
        start = (8,   201); end = (110, 1)
    elif map_name == "very_hard.gif":
        start = (1,   324); end = (580, 1)
    elif map_name == "my_maze.gif":
        start = (0,   0);   end = (500, 205)
    elif map_name == "my_maze2.gif":
        start = (0,   0);   end = (599, 350)
    elif map_name == "super_hard.gif":
        start = (1,   1);   end = (999, 999)
    else:
        print(f"Unknown map: {map_name}")
        sys.exit(1)

    print(f"\n{'='*45}")
    print(f"  A* ({heuristic_name}) — {map_name}")
    print(f"  Start: {start}   End: {end}")
    print(f"{'='*45}")

    im = Image.open(difficulty).convert("1")
    search(im.load())

    stem = map_name.replace(".gif", "")
    visualize_matplotlib(f"astar_{stem}_{heuristic_name}_plot.png")
    visualize_pil(f"astar_{stem}_{heuristic_name}_pil.png")
