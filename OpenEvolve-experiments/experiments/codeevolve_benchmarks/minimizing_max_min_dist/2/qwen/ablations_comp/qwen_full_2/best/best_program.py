# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances using pdist for efficiency
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0 or len(distances) < 2:
            return float('inf')
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero or near-zero values
        if max_dist <= 1e-10:
            return float('inf')
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        # Add small epsilon to prevent numerical issues
        return -(min_dist + 1e-10) / (max_dist + 1e-10)
    
    # Create better initial configuration using multiple strategies
    np.random.seed(42)
    
    # Strategy 1: Golden spiral with optimized parameters
    golden_points = []
    golden_angle = 2.399963229728653  # 2π(1 - 1/φ)
    for i in range(16):
        r = 0.45 * np.sqrt(i / 15.0) + 0.015  # Better radial scaling
        theta = i * golden_angle + 0.15 * np.sin(i * 0.4)  # Add angular variation
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        golden_points.append([x, y])
    
    # Strategy 2: Hexagonal close packing with reduced jitter
    hex_points = []
    rows = 4
    cols = 4
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    for i in range(rows):
        for j in range(cols):
            x = (j + 1) * spacing_x + np.random.normal(0, 0.003)
            y = (i + 1) * spacing_y + np.random.normal(0, 0.003)
            hex_points.append([x, y])
    
    # Strategy 3: Square grid with controlled jitter
    grid_points = []
    for i in range(4):
        for j in range(4):
            x = (j + 0.5) / 4.0 + np.random.normal(0, 0.01)
            y = (i + 0.5) / 4.0 + np.random.normal(0, 0.01)
            grid_points.append([x, y])
    
    # Strategy 4: Concentric rings with better spacing
    ring_points = []
    n_rings = 4
    points_per_ring = 4
    for ring in range(n_rings):
        radius = 0.15 + ring * 0.2  # Better spacing
        for i in range(points_per_ring):
            angle = i * 2 * np.pi / points_per_ring + ring * 0.05  # Slight angular variation
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            ring_points.append([x, y])
    
    # Strategy 5: Fibonacci-based point distribution for better uniformity
    fib_points = []
    phi = (1 + np.sqrt(5)) / 2
    for i in range(16):
        theta = i * 2 * np.pi / phi + 0.18 * np.sin(i * 0.5)  # Add variation
        radius = np.sqrt(i / 15.0) * 0.48  # Slightly larger radius
        x = 0.5 + radius * np.cos(theta)
        y = 0.5 + radius * np.sin(theta)
        fib_points.append([x, y])
    
    # Strategy 6: Simple regular hexagonal pattern (removing complex hex_lattice)
    simple_hex_points = []
    # Create 6 points in a hexagon around center
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        simple_hex_points.append([x, y])
    # Add 10 more points in a grid pattern
    for i in range(2):
        for j in range(5):
            if len(simple_hex_points) >= 16:
                break
            x = 0.2 + j * 0.15
            y = 0.2 + i * 0.15
            simple_hex_points.append([x, y])
    
    # Evaluate all strategies and choose the best one based on initial min/max ratio
    strategies = [
        np.array(golden_points),
        np.array(hex_points),
        np.array(grid_points),
        np.array(ring_points),
        np.array(fib_points),
        np.array(simple_hex_points)
    ]
    
    best_strategy_idx = 0
    best_ratio = 0
    
    for i, strategy in enumerate(strategies):
        strategy = np.clip(strategy, 0, 1)
        distances = pdist(strategy)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_strategy_idx = i
    
    points = strategies[best_strategy_idx]
    
    # Add moderate initial perturbation to escape local minima
    for i in range(16):
        points[i, 0] += np.random.normal(0, 0.05)
        points[i, 1] += np.random.normal(0, 0.05)
    
    # Ensure points are within bounds
    points = np.clip(points, 0, 1)
    
    # Enhanced optimization approach with multiple phases
    try:
        # Phase 1: Global optimization with balanced settings for better efficiency
        bounds = [(0, 1)] * 32
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=600,  # Reduced iterations to stay within time budget
            popsize=40,    # Moderate population size
            mutation=(0.9, 1),  # Balanced mutation rate
            recombination=0.9,   # Balanced recombination
            atol=1e-12,    # Tight tolerance for precision
            rtol=1e-12
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
    except Exception as e:
        warnings.warn(f"Global optimization failed: {e}")
    
    # Phase 2: Multi-start local optimization with multiple restarts and better strategies
    best_local_result = None
    best_local_value = float('inf')
    
    # Try multiple restarts with different strategies
    for restart in range(8):  # Fewer restarts for better efficiency
        try:
            # Create different starting point strategies
            if restart < 4:  # More aggressive perturbations for first few restarts
                local_start = points.flatten() + np.random.normal(0, 0.07, 32)
            else:  # Smaller perturbations for fine-tuning
                local_start = points.flatten() + np.random.normal(0, 0.03, 32)
            
            # Ensure bounds are respected
            local_start = np.clip(local_start, 0, 1)
            
            # Try different optimization methods
            if restart % 3 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            elif restart % 3 == 1:
                method = 'SLSQP'
                options = {'maxiter': 1200, 'ftol': 1e-15}
            else:
                method = 'TNC'
                options = {'maxiter': 1500, 'ftol': 1e-16}
            
            result = minimize(
                objective,
                local_start,
                method=method,
                bounds=[(0, 1)] * 32,
                options=options
            )
            
            if result.success and result.fun < best_local_value:
                best_local_value = result.fun
                best_local_result = result
        except Exception as e:
            continue
    
    if best_local_result is not None:
        optimized_points = best_local_result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
        return optimized_points
    
    # Phase 3: Additional global optimization fallback for maximum robustness
    try:
        # Try another global optimization with different parameters
        result = differential_evolution(
            objective,
            bounds=[(0, 1) for _ in range(32)],
            seed=42,
            maxiter=800,
            popsize=50,
            mutation=(0.95, 1),
            recombination=0.98,
            strategy='best1bin'
        )
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
    except Exception:
        pass
    
    # Phase 4: Final fallback to best initial configuration
    return points.reshape(-1, 2)


# EVOLVE-BLOCK-END
