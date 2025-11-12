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

def check_factible_grid(grid, adjacency_matrix):
    """
    Check if a grid satisfies the hardcore property: no two adjacent vertices
    (connected by an edge) can both have a particle (value 1) at the same time.
    
    Args:
        grid: KxK numpy array where 1 means a particle is present, 0 means no particle
        adjacency_matrix: Adjacency matrix representing edges in the graph.
                          # Note that a KxK matrix has KxK elements
                          Should be (K*K) x (K*K) where adjacency_matrix[i, j] = 1
                          means there's an edge between vertex i and vertex j.
                          Each row i represents vertex i, and columns j represent
                          which vertices vertex i is connected to.
    
    Returns:
        bool: True if grid satisfies hardcore property, False otherwise
    """
    K = grid.shape[0]
    n_vertices = K * K
    
    # Flatten the grid to work with 1D vertex indices (row-major order)
    # Vertex at grid[i, j] corresponds to vertex index i*K + j
    flat_grid = grid.flatten()
    
    # Ensure adjacency matrix has the correct shape
    if adjacency_matrix.shape != (n_vertices, n_vertices):
        raise ValueError(
            f"Adjacency matrix shape {adjacency_matrix.shape} is not compatible. "
            f"Expected ({n_vertices}, {n_vertices}) for a {K}x{K} grid."
        )
    
    # Check every edge in the graph (cause' its undirected just check upper triangle (or inferior))
    # For each vertex u (row), check all vertices v it's connected to (columns)
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):  # Start from u+1
            if adjacency_matrix[u, v] == 1:
                if flat_grid[u] == 1 and flat_grid[v] == 1:
                    return False
    
    return True

def run_gibbs_hc_random_adjacency(grid, K, adjacency_matrix):
    """
    Move from X_{n} state to X_{n+1} in the hardcore problem using Gibbs random sampler.
    Uses adjacency matrix to determine neighbors instead of grid structure.
    
    Args:
        grid: KxK numpy array where 1 means a particle is present, 0 means no particle
        K: Size of the grid (KxK)
        adjacency_matrix: (K*K) x (K*K) adjacency matrix where adjacency_matrix[i, j] = 1
                          means there's an edge between vertex i and vertex j.
    
    Returns:
        grid: Updated grid after one Gibbs sampling step
    """
    n_vertices = K * K
    
    # Ensure adjacency matrix has the correct shape
    if adjacency_matrix.shape != (n_vertices, n_vertices):
        raise ValueError(
            f"Adjacency matrix shape {adjacency_matrix.shape} is not compatible. "
            f"Expected ({n_vertices}, {n_vertices}) for a {K}x{K} grid."
        )
    
    # Flatten the grid to work with 1D vertex indices (row-major order)
    # Vertex at grid[i, j] corresponds to vertex index i*K + j
    flat_grid = grid.flatten()
    
    # Select a random vertex
    u = random.randint(0, n_vertices - 1)
    
    # Check if any neighbor has a particle
    neighbors_have_particles = False
    for v in range(n_vertices):
        if adjacency_matrix[u, v] == 1:  # There's an edge between u and v
            if flat_grid[v] == 1:  # Neighbor v has a particle
                neighbors_have_particles = True
                break
    
    # Update the vertex according to hardcore property
    if neighbors_have_particles:
        flat_grid[u] = 0  # Force no particle if neighbors have particles
    else:
        # Randomly assign 0 or 1 if no neighbors have particles
        coin_flipped = random.randint(0, 1)
        flat_grid[u] = coin_flipped
    
    # Reshape back to KxK grid
    grid = flat_grid.reshape(K, K)
    
    return grid

def run_gibbs_hc_order_adjacency(grid, K, adjacency_matrix, current_vertex_idx):
    """
    Move from X_{n} state to X_{n+1} in the hardcore problem using Gibbs sampler with order.
    Uses adjacency matrix to determine neighbors instead of grid structure.
    
    Args:
        grid: KxK numpy array where 1 means a particle is present, 0 means no particle
        K: Size of the grid (KxK)
        adjacency_matrix: (K*K) x (K*K) adjacency matrix where adjacency_matrix[i, j] = 1
                          means there's an edge between vertex i and vertex j.
        current_vertex_idx: Current vertex index (0 to K*K-1) to process
    
    Returns:
        grid: Updated grid after one Gibbs sampling step
        next_vertex_idx: Next vertex index to process (wraps around to 0 after K*K-1)
    """
    n_vertices = K * K
    
    # Ensure adjacency matrix has the correct shape
    if adjacency_matrix.shape != (n_vertices, n_vertices):
        raise ValueError(
            f"Adjacency matrix shape {adjacency_matrix.shape} is not compatible. "
            f"Expected ({n_vertices}, {n_vertices}) for a {K}x{K} grid."
        )
    
    # Flatten the grid to work with 1D vertex indices (row-major order)
    flat_grid = grid.flatten()
    
    # Process the current vertex
    u = current_vertex_idx
    
    # Check if any neighbor has a particle
    neighbors_have_particles = False
    for v in range(n_vertices):
        if adjacency_matrix[u, v] == 1:  # There's an edge between u and v
            if flat_grid[v] == 1:  # Neighbor v has a particle
                neighbors_have_particles = True
                break
    
    # Update the vertex according to hardcore property
    if neighbors_have_particles:
        flat_grid[u] = 0  # Force no particle if neighbors have particles
    else:
        # Randomly assign 0 or 1 if no neighbors have particles
        coin_flipped = random.randint(0, 1)
        flat_grid[u] = coin_flipped
    
    # Reshape back to KxK grid
    grid = flat_grid.reshape(K, K)
    
    # Move to next vertex (wrap around)
    next_vertex_idx = (current_vertex_idx + 1) % n_vertices
    
    return grid, next_vertex_idx