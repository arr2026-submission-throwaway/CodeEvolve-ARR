# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 26
    
    # Better initialization using more sophisticated approach - inspired by top performers
    # Use a hexagonal lattice pattern optimized for 26 circles with better spacing
    rows = 5
    cols = 6
    
    # Create hexagonal grid with optimized spacing - similar to top performer #1
    # Fine-tuned parameters for better balance between density and edge utilization
    base_radius = 0.09  # Standard base radius
    spacing = base_radius * 2.05  # Slightly looser spacing for better optimization flexibility
    
    positions = []
    radii = []
    
    # Generate hexagonal grid
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = 0.1 + j * spacing + (i % 2) * spacing/2
            y = 0.1 + i * spacing * 0.866  # sqrt(3)/2
            if x <= 1 - base_radius and y <= 1 - base_radius:
                positions.append([x, y])
                radii.append(base_radius)
    
    # Fill remaining positions with random valid placements but with better distribution
    while len(positions) < n:
        # Use stratified sampling to avoid edge clustering
        x = np.random.uniform(base_radius, 1 - base_radius)
        y = np.random.uniform(base_radius, 1 - base_radius)
        # Start with standard initial radii to allow for better optimization
        radius = np.random.uniform(0.05, base_radius * 1.3)  # Moderate minimum radius
        
        # Check if this circle overlaps with existing ones
        valid = True
        for pos, r in zip(positions, radii):
            dist_sq = (x - pos[0])**2 + (y - pos[1])**2
            min_dist_sq = (radius + r)**2
            if dist_sq < min_dist_sq:
                valid = False
                break
        
        if valid:
            positions.append([x, y])
            radii.append(radius)
    
    # Ensure we have exactly n circles
    positions = positions[:n]
    radii = radii[:n]
    
    # Convert to final array format
    circles = np.column_stack([positions, radii])
    
    # More sophisticated optimization using local search with better strategies
    best_sum = 0
    best_circles = None
    
    # Try multiple random restarts to avoid local minima - inspired by top performers
    for restart in range(20):  # More restarts for better exploration
        # Reset to initial configuration for each restart
        current_circles = circles.copy()
        
        # Use a more intelligent optimization approach with adaptive iteration limits
        max_iterations = 6000 if restart < 8 else 4000  # More iterations for better convergence
        
        # Track improvement for early termination
        last_improvement = 0
        no_improvement_count = 0
        max_no_improvement = 100  # Early termination if no improvement for 100 iterations
        
        for iteration in range(max_iterations):
            improved = False
            
            # Sort circles by current radius to prioritize optimization of smaller circles first
            radii = current_circles[:, 2]
            sorted_indices = np.argsort(radii)
            
            # Try to increase each radius systematically, prioritizing smaller ones
            for i in sorted_indices:
                original_radius = current_circles[i][2]
                
                # Binary search for maximum possible radius with better precision
                left, right = original_radius, 0.5
                best_radius = original_radius
                max_attempts = 35  # More attempts for higher precision
                
                attempts = 0
                while left < right and attempts < max_attempts:
                    attempts += 1
                    mid = (left + right) / 2
                    test_radius = min(mid, 0.5)
                    
                    # Optimized constraint checking - early exit conditions
                    valid = True
                    
                    # Check containment first (fastest check)
                    if (test_radius > current_circles[i][0] or 
                        test_radius > 1 - current_circles[i][0] or
                        test_radius > current_circles[i][1] or 
                        test_radius > 1 - current_circles[i][1]):
                        valid = False
                    else:
                        # Vectorized overlap checking for efficiency
                        dx = current_circles[:, 0] - current_circles[i][0]
                        dy = current_circles[:, 1] - current_circles[i][1]
                        distances_squared = dx*dx + dy*dy
                        
                        # Precompute radii sums to avoid recomputation
                        radii_sums = test_radius + current_circles[:, 2]
                        
                        # Check overlap with all other circles more efficiently
                        # Early termination optimization with better logic
                        overlap_detected = False
                        for j in range(n):
                            if i != j:
                                distance_squared = distances_squared[j]
                                radii_sum = radii_sums[j]
                                
                                # Early exit if definitely not overlapping
                                if distance_squared >= radii_sum * radii_sum:
                                    continue
                                # If we get here, there might be overlap
                                if distance_squared < radii_sum * radii_sum:
                                    overlap_detected = True
                                    break
                        
                        if overlap_detected:
                            valid = False
                    
                    if valid:
                        best_radius = test_radius
                        left = test_radius
                    else:
                        right = test_radius
                
                # Update if we found a better radius
                if best_radius > original_radius:
                    current_circles[i][2] = best_radius
                    improved = True
            
            # If no radius improvement, try position adjustments with smarter strategy
            if not improved:
                # Identify problematic circles (those with tight constraints)
                problem_indices = []
                for i in range(n):
                    min_dist = float('inf')
                    for j in range(n):
                        if i != j:
                            dx = current_circles[i][0] - current_circles[j][0]
                            dy = current_circles[i][1] - current_circles[j][1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            min_dist = min(min_dist, dist)
                    # If the minimum distance is very small relative to radii, it's problematic
                    if min_dist < (current_circles[i][2] * 1.5):
                        problem_indices.append(i)
                
                # Try adjusting positions more strategically
                adjustments_to_try = 250 if problem_indices else 200  # Balanced adjustments
                
                for _ in range(adjustments_to_try):
                    # Prioritize adjusting problem circles if they exist
                    if problem_indices:
                        i = random.choice(problem_indices)
                    else:
                        i = random.randint(0, n-1)
                    
                    old_x, old_y, old_r = current_circles[i]
                    
                    # Adaptive adjustment based on radius size with more conservative deltas
                    if old_r < 0.03:
                        delta = 0.003  # More conservative
                    elif old_r < 0.06:
                        delta = 0.005  # More conservative
                    else:
                        delta = 0.008  # More conservative
                    new_x = old_x + random.uniform(-delta, delta)
                    new_y = old_y + random.uniform(-delta, delta)
                    
                    # Keep within bounds with stricter constraints
                    new_x = max(old_r, min(1-old_r, new_x))
                    new_y = max(old_r, min(1-old_r, new_y))
                    
                    # Check constraints efficiently
                    valid = True
                    # Check containment
                    if (new_x < old_r or new_x > 1-old_r or 
                        new_y < old_r or new_y > 1-old_r):
                        valid = False
                    else:
                        # Vectorized overlap check with precomputed values
                        dx = current_circles[:, 0] - new_x
                        dy = current_circles[:, 1] - new_y
                        distances_squared = dx*dx + dy*dy
                        
                        # Precompute radii sums for efficiency
                        radii_sums = old_r + current_circles[:, 2]
                        
                        # Early termination optimization with better logic
                        overlap_detected = False
                        for j in range(n):
                            if i != j:
                                distance_squared = distances_squared[j]
                                radii_sum = radii_sums[j]
                                
                                # Early exit if definitely not overlapping
                                if distance_squared >= radii_sum * radii_sum:
                                    continue
                                # If we get here, there might be overlap
                                if distance_squared < radii_sum * radii_sum:
                                    overlap_detected = True
                                    break
                        
                        if overlap_detected:
                            valid = False
                    
                    if valid:
                        current_circles[i][0] = new_x
                        current_circles[i][1] = new_y
                        improved = True
            
            # Early termination logic
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break
            else:
                no_improvement_count = 0
            
            # Check for significant improvement to terminate early
            if iteration > 50 and iteration % 25 == 0:  # More frequent checks
                current_sum = np.sum(current_circles[:, 2])
                if current_sum - last_improvement < 0.00003:  # Less strict threshold for more exploration
                    break
                last_improvement = current_sum
        
        # Calculate total sum
        current_sum = np.sum(current_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = current_circles.copy()
    
    # Return the best solution found directly - remove the potentially destabilizing final refinement
    return best_circles

def is_valid_configuration_fast(circles):
    """Fast constraint checking for circle packing."""
    n = len(circles)
    
    # Check containment constraints efficiently
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check overlap constraints more efficiently
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Quick early exit check for obvious overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance_squared = dx*dx + dy*dy
            radii_sum = radii[i] + radii[j]
            
            if distance_squared < radii_sum * radii_sum:
                return False
    
    return True

def simple_grid_placement():
    """Fallback method for simple grid-based placement."""
    n = 26
    circles = np.zeros((n, 3))
    
    # Grid placement with some randomness
    grid_size = 5  # 5x5 grid for 25 circles, leave one space
    spacing = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing
            y = (j + 1) * spacing
            # Add small randomness to avoid perfect grid
            x += random.uniform(-spacing/8, spacing/8)
            y += random.uniform(-spacing/8, spacing/8)
            # Radius based on proximity to edges
            r = min(x, 1-x, y, 1-y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    return circles

# EVOLVE-BLOCK-END
