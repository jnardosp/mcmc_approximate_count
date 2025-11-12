import numpy as np
import random

# This file is to store util functions related to gibbs_sampler

def create_k_matrix(K):
    # Create a kxk matrix initialized with no particles (you can actually initialize in whichever way you would like lol)
    initial_grid = np.zeros((K, K), dtype=int)
    initial_grid_2 = np.ones((K, K), dtype=int)

    return initial_grid, initial_grid_2

# This only works when connections in the graph are as in hardcore
def run_gibbs_hc_random(grid, K):
    """Move from X_{n} state to X_{n+1} in the hardcore problem using Gibbs random sampler."""
    # First we select a random vertex on our grid
    i = random.randint(0, K-1)
    j = random.randint(0, K-1)

    # Sample another value for vertex given the other vertex
    # Check if neighbor vertex are 0 (for possible particle in factible grid)
    # Boundary cases
    neighbors_empty = True  # Start assuming neighbors are empty

    # Check top neighbor (only if not at top boundary)
    if i > 0 and grid[i-1, j] == 1:
        neighbors_empty = False

    # Check bottom neighbor (only if not at bottom boundary)
    if i < K-1 and grid[i+1, j] == 1:
        neighbors_empty = False

    # Check left neighbor (only if not at left boundary)
    if j > 0 and grid[i, j-1] == 1:
        neighbors_empty = False

    # Check right neighbor (only if not at right boundary)
    if j < K-1 and grid[i, j+1] == 1:
        neighbors_empty = False

    if neighbors_empty:
        coin_flipped = random.randint(0,1) # One for heads, zero for tails
        if coin_flipped == 1:
            grid[i, j] = 1 # Put particle
        else:
            grid[i,j] = 0 # Remove particle
    else:
        grid[i,j] = 0 # If neighbors have particles force no particle in vertex

    return grid

# This only works when connections in the graph are as in hardcore
def run_gibbs_hc_order(grid, K, i, j):
    """Move from X_{n} state to X{n+1} in the hardcore problem using Gibbs sampler with order (for faster convergence)."""
    # Check if neighbor vertex are 0 (for possible particle in factible grid)
    # Boundary cases
    neighbors_empty = True  # Start assuming neighbors are empty

    # Check top neighbor (only if not at top boundary)
    if i > 0 and grid[i-1, j] == 1:
        neighbors_empty = False

    # Check bottom neighbor (only if not at bottom boundary)
    if i < K-1 and grid[i+1, j] == 1:
        neighbors_empty = False

    # Check left neighbor (only if not at left boundary)
    if j > 0 and grid[i, j-1] == 1:
        neighbors_empty = False

    # Check right neighbor (only if not at right boundary)
    if j < K-1 and grid[i, j+1] == 1:
        neighbors_empty = False

    if neighbors_empty:
        coin_flipped = random.randint(0,1) # One for heads, zero for tails
        if coin_flipped == 1:
            grid[i, j] = 1 # Put particle
        else:
            grid[i,j] = 0 # Remove particle
    else:
        grid[i,j] = 0 # If neighbors have particles force no particle in vertex
    # We pass every vertex of grid in order
    if i == K-1:
        if j == K-1:
            # If we changed last column, last row, start in the beginning of the grid
            i = 0
            j = 0
        else:
            j += 1
    else:
        if j == K-1:
            # If we changed last column, i row, change row and j = 0
            i += 1
            j = 0
        else:
            j += 1

    return grid, i, j
