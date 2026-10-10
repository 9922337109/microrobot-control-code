import numpy as np
import cv2
import heapq
import os
from environments import MicrorobotEnvGameByTheWall

# --- CONFIGURATION ---
IMAGE_PATH = 'binary_images/default_segmentation_vascular.png'
LOGDIR = "results/astar_baseline"
os.makedirs(LOGDIR, exist_ok=True)

# A* ALGORITHM
def heuristic(a, b):
    # Euclidean distance
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    """
    Finds shortest path on a grid. 
    array: 1 = Obstacle/Wall, 0 = Free Path
    """
    # 8-way movement (including diagonals)
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        # Check if we are close to the goal (within 2 pixels)
        if np.linalg.norm(np.array(current) - np.array(goal)) < 2.0:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1] # Return path
            
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            # Check bounds
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    # Check Collision (1 = Wall)
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else: continue # Out of bounds y
            else: continue # Out of bounds x
                
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return None # No path found

def main():
    print("------------------------------------------------")
    print("   STARTING A* (ORACLE) EVALUATION              ")
    print("------------------------------------------------")
    
    env_config = {
        "image_string": IMAGE_PATH,
        "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
        "reduce_radius": -0.01,
        "name": (0, 0),
        "render_mode": "rgb_array", 
    }
    
    # 1. Initialize Env to get start/end points
    env = MicrorobotEnvGameByTheWall(**env_config)
    env.reset()
    
    # 2. Load Map Directly from Disk (Robust Method)
    # Load as grayscale
    raw_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")
    
    # Resize if necessary (Env usually uses 64x64 or whatever input is)
    # The environment observation is 64x64, let's assume the image is scaled to that.
    # Check env observation size
    obs_shape = env.observation_space['image'].shape[:2]
    if raw_img.shape != obs_shape:
        raw_img = cv2.resize(raw_img, (obs_shape[1], obs_shape[0]))
    
    # Threshold: 0 = Background/Wall, 255 = Vessel/Path
    # We want 1 = Wall (Block), 0 = Path (Free) for A*
    # Usually these vascular images are: White (255) = Vessel, Black (0) = Background
    # So we invert it.
    _, binary_map = cv2.threshold(raw_img, 127, 1, cv2.THRESH_BINARY)
    grid = 1 - binary_map # Now 1=Background(Wall), 0=Vessel(Path)
    
    # 3. Get Coords
    try:
        start_pos = env.unwrapped.agent_location.astype(int)
        target_pos = env.unwrapped.target_location.astype(int)
    except AttributeError:
        # Fallback if names are different
        print("Warning: Using 'agent_position' (normalized) and scaling to grid.")
        start_pos = (env.unwrapped.agent_position * np.array(grid.shape)).astype(int)
        target_pos = (env.unwrapped.target_position * np.array(grid.shape)).astype(int)

    print(f"Map Size: {grid.shape}")
    print(f"Planning from {start_pos} to {target_pos}...")
    
    # 4. Run A*
    import time
    start_time = time.time()
    path = astar(grid, tuple(start_pos), tuple(target_pos))
    duration = time.time() - start_time
    
    if path:
        print(f"A* Success! Path found with {len(path)} steps.")
        print(f"Calculation Time: {duration:.4f} seconds")
    else:
        print("A* Failed. Target might be unreachable (Wall block).")

if __name__ == '__main__':
    main()