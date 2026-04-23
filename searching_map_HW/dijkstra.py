#!/usr/bin/env python3
"""
ME571 - Introduction to Robotic Technology
Homework 3 - Dijkstra's Algorithm

Run: python dijkstra.py <map_name>
Example: python dijkstra.py trivial.gif
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from queue import PriorityQueue

# ---------------------------------------------------------------------------
# Global variables - set at runtime from command line
# ---------------------------------------------------------------------------
start      = (0, 0)   # (x, y) start pixel
end        = (0, 0)   # (x, y) goal pixel
difficulty = ""       # path to map file

# ---------------------------------------------------------------------------
# Algorithm data structures - filled during search()
# ---------------------------------------------------------------------------
path       = []   # ordered list of (x,y) tuples from start to goal
expanded   = {}   # nodes that have been popped and fully processed
frontier   = {}   # nodes currently sitting in the priority queue
came_from  = {}   # tracks the parent of each node for path reconstruction
cost_so_far = {}  # tracks the cheapest known cost to reach each node

# ---------------------------------------------------------------------------
# PIL display colors  (R, G, B)
# ---------------------------------------------------------------------------
NEON_GREEN = (0,   255, 0)
PURPLE     = (85,  26,  139)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY  = (100, 100, 100)


# ---------------------------------------------------------------------------
# Helper: return passable 4-directional neighbors
# ---------------------------------------------------------------------------
def get_neighbors(node, map_data, width, height):
    """
    Returns the list of valid neighbors for a given node.
    Only moves up/down/left/right (4-connected grid).
    A neighbor is valid if it is inside the image bounds
    and its pixel is white (passable, value = True in a 1-bit image).
    """
    x, y = node
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if map_data[nx, ny]:          # white pixel = free space
                neighbors.append((nx, ny))
    return neighbors


# ---------------------------------------------------------------------------
# Dijkstra search
# ---------------------------------------------------------------------------
def search(map_data):
    """
    Dijkstra's algorithm on a pixel grid.

    How it works:
      - Uses a min-heap priority queue ordered by cost-so-far.
      - Every step costs 1 (uniform grid, 4-directional movement).
      - Expands the cheapest known node first until the goal is reached.
      - Guarantees the shortest path but explores in all directions equally.

    Fills the global variables: path, expanded, frontier, came_from, cost_so_far.
    """
    global path, expanded, frontier, came_from, cost_so_far

    # Get image dimensions for bounds checking
    im     = Image.open(difficulty)
    width, height = im.size

    # Initialise: put start node in the queue with cost 0
    pq = PriorityQueue()
    pq.put((0, start))
    came_from[start]   = None   # start has no parent
    cost_so_far[start] = 0
    frontier[start]    = 0

    while not pq.empty():
        current_cost, current = pq.get()

        # Stop as soon as we pop the goal node
        if current == end:
            print("Goal reached!")
            break

        # Skip stale entries - a cheaper path to this node was already found
        if current_cost > cost_so_far.get(current, float('inf')):
            continue

        # Move node from frontier to expanded
        expanded[current] = current_cost
        frontier.pop(current, None)

        # Check every valid neighbor
        for neighbor in get_neighbors(current, map_data, width, height):
            new_cost = cost_so_far[current] + 1   # uniform step cost = 1

            # Only update if we found a cheaper route to this neighbor
            if new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor]   = current
                pq.put((new_cost, neighbor))
                frontier[neighbor]    = new_cost

    # --- Reconstruct path by backtracking through came_from ---
    if end in came_from:
        current = end
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()   # flip from [end...start] to [start...end]
    else:
        print("No path found.")
        return

    # Print summary to terminal
    print(f"  Nodes expanded : {len(expanded)}")
    print(f"  Nodes in frontier : {len(frontier)}")
    print(f"  Path length (steps) : {len(path) - 1}")
    print(f"  Final path cost : {cost_so_far[end]}")


# ---------------------------------------------------------------------------
# Matplotlib visualization (for the report)
# ---------------------------------------------------------------------------
def visualize_matplotlib(save_file="dijkstra_plot.png"):
    """
    Overlays color-coded search results on the maze image using matplotlib.
      - Blue  : expanded nodes (fully processed)
      - Yellow: remaining frontier nodes
      - Red   : final path
      - Green : start and end
    Saves the figure and displays it.
    """
    im         = Image.open(difficulty).convert("RGB")
    width, height = im.size
    img_array  = np.array(im)

    # Build an RGBA overlay - start fully transparent
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # Expanded nodes -> blue
    for (x, y) in expanded:
        overlay[y, x] = [30, 144, 255, 160]

    # Frontier nodes -> yellow
    for (x, y) in frontier:
        overlay[y, x] = [255, 215, 0, 200]

    # Final path -> red
    for (x, y) in path:
        overlay[y, x] = [220, 20, 60, 255]

    # Start and end -> bright green
    overlay[start[1], start[0]] = [0, 255, 0, 255]
    overlay[end[1],   end[0]]   = [0, 255, 0, 255]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    ax.imshow(overlay)

    # Annotate start and end
    ax.annotate("START", xy=(start[0], start[1]),
                xytext=(start[0] + 3, start[1] + 6),
                color="lime", fontsize=7, fontweight="bold")
    ax.annotate("END",   xy=(end[0], end[1]),
                xytext=(end[0]   + 3, end[1]   + 6),
                color="lime", fontsize=7, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#1E90FF", label=f"Expanded nodes  ({len(expanded)})"),
        mpatches.Patch(color="#FFD700", label=f"Frontier nodes  ({len(frontier)})"),
        mpatches.Patch(color="#DC143C", label=f"Final path  (cost = {cost_so_far.get(end, 'N/A')})"),
        mpatches.Patch(color="lime",    label="Start / End"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.9)

    map_name = difficulty.split("/")[-1].replace(".gif", "")
    ax.set_title(
        f"Dijkstra's Algorithm  —  {map_name} map\n"
        f"Path cost: {cost_so_far.get(end, 'N/A')}   |   "
        f"Nodes expanded: {len(expanded)}",
        fontsize=11
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {save_file}")


# ---------------------------------------------------------------------------
# PIL visualization (pixel-level, matches original visualize_search style)
# ---------------------------------------------------------------------------
def visualize_pil(save_file="dijkstra_result.png"):
    """
    Draws results directly onto the map image pixel by pixel.
    Saves the colored image to disk.
    """
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
    if len(sys.argv) < 2:
        print("Usage : python dijkstra.py <map_name>")
        print("Example: python dijkstra.py trivial.gif")
        sys.exit(1)

    map_name   = sys.argv[1]
    difficulty = "maps/" + map_name

    # Hard-coded start/end for each map (from original starter code)
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
    print(f"  Dijkstra — {map_name}")
    print(f"  Start: {start}   End: {end}")
    print(f"{'='*45}")

    # Load map as 1-bit image and run search
    im = Image.open(difficulty).convert("1")
    search(im.load())

    # Save visualizations
    stem = map_name.replace(".gif", "")
    visualize_matplotlib(f"dijkstra_{stem}_plot.png")
    visualize_pil(f"dijkstra_{stem}_pil.png")
