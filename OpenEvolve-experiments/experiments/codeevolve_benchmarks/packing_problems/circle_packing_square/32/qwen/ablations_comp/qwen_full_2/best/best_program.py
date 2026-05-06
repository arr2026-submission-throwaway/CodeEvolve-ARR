# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    
    # Better initialization using a more sophisticated approach inspired by top performers
    circles = np.zeros((n, 3))
    
    # Use a more efficient hexagonal packing pattern with better spacing and density
    rows = 6
    cols = 6
    # Use tighter spacing to allow for better overall packing
    spacing_x = 0.98 / cols  # Slightly tighter spacing for better density
    spacing_y = 0.98 / rows  # Slightly tighter spacing for better density
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = 0.01 + (j + 0.5) * spacing_x  # Even tighter margins
            y = 0.01 + (i + 0.5) * spacing_y  # Even tighter margins
            # Adjust for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x * 0.5
            
            # Initial radius - more aggressive starting point
            r = min(spacing_x, spacing_y) * 0.5  # Slightly larger initial radius to start with better potential
            
            # Ensure it fits in the square
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions more intelligently with better spatial awareness
    for i in range(idx, n):
        attempts = 0
        while attempts < 700:  # More attempts for better placement
            # Try to place in areas with more available space
            # Use a smarter distribution that considers existing layout
            x = random.uniform(0.01, 0.99)  # Even tighter margins
            y = random.uniform(0.01, 0.99)  # Even tighter margins
            
            # Calculate maximum possible radius based on proximity to edges
            min_dist_to_edges = min(x, 1-x, y, 1-y)
            
            # Find closest existing circle to estimate space availability
            min_dist_to_circles = float('inf')
            closest_circle_idx = -1
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < min_dist_to_circles:
                    min_dist_to_circles = dist
                    closest_circle_idx = j
            
            # Calculate radius more intelligently - prioritize placing in less crowded areas
            if min_dist_to_circles < float('inf') and min_dist_to_circles > 0.001:
                # Radius based on distance to nearest circle, but also consider the circle's own radius
                radius = min(min_dist_to_edges, min_dist_to_circles/2 - 0.001)
                # Also consider how much space the closest circle already has
                if closest_circle_idx >= 0:
                    closest_r = circles[closest_circle_idx, 2]
                    # If we're near a large circle, be more conservative
                    if closest_r > 0.05:
                        radius = min(radius, closest_r * 0.5)
            else:
                radius = min_dist_to_edges * 0.95  # More aggressive placement
            
            # Make sure radius is reasonable and not too small
            radius = max(0.005, min(0.4, radius))  # Slightly increased max radius
            
            # Check containment
            if (x - radius < 0 or x + radius > 1 or 
                y - radius < 0 or y + radius > 1):
                attempts += 1
                continue
                
            # Check overlap with existing circles (optimized with early termination)
            valid = True
            # Use spatial indexing for faster overlap checking
            if i > 0:
                tree = cKDTree(circles[:i, :2])
                # Query nearby circles to check overlaps
                distances, indices = tree.query([x, y], k=min(20, i))  # More neighbors
                nearby_indices = indices[distances < 1.0]
                for j in nearby_indices:
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < radius + existing_r:
                        valid = False
                        break
            else:
                # No existing circles to check against
                valid = True
                    
            if valid:
                circles[i] = [x, y, radius]
                break
            attempts += 1
    
    # Enhanced optimization with multiple stages
    def check_validity(config):
        """Check if configuration is valid"""
        for i in range(n):
            x, y, r = config[i]
            # Check containment
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
            # Check overlaps
            for j in range(i):
                x2, y2, r2 = config[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist < r + r2:
                    return False
        return True
    
    def evaluate_fitness(config):
        """Evaluate sum of radii"""
        return np.sum(config[:, 2])
    
    # Multi-stage optimization approach - more sophisticated than simple local search
    best_config = circles.copy()
    best_fitness = evaluate_fitness(best_config)
    
    # Stage 1: Extensive local search with adaptive perturbations - enhanced
    for iteration in range(20000):  # More iterations for better exploration
        test_config = best_config.copy()
        
        # Choose which circle to modify with preference for smaller radii
        # This helps optimize the overall packing more effectively
        if random.random() < 0.6:  # Higher chance to target smaller circles for more impact
            # Prefer circles with smaller radii for more significant improvement potential
            weights = [1.0/(best_config[i, 2] + 0.001) for i in range(n)]
            idx = random.choices(range(n), weights=weights)[0]
        else:
            idx = random.randint(0, n-1)
        
        # Adaptive perturbation based on current radius and iteration
        adaptive_step = 0.03 + best_config[idx, 2] * 0.09  # Larger steps for better exploration
        # Decrease step size as iterations progress
        step_factor = 1.0 - (iteration / 20000.0) * 0.98  # More aggressive cooling
        adaptive_step *= step_factor
        
        test_config[idx, 0] += random.uniform(-adaptive_step, adaptive_step)
        test_config[idx, 1] += random.uniform(-adaptive_step, adaptive_step)
        test_config[idx, 2] += random.uniform(-adaptive_step*0.3, adaptive_step*0.3)  # Slightly larger radius changes
        
        # Apply bounds
        test_config[idx, 0] = np.clip(test_config[idx, 0], test_config[idx, 2], 1 - test_config[idx, 2])
        test_config[idx, 1] = np.clip(test_config[idx, 1], test_config[idx, 2], 1 - test_config[idx, 2])
        test_config[idx, 2] = max(0.001, min(0.5, test_config[idx, 2]))
        
        # If the new configuration is valid, consider it
        if check_validity(test_config):
            new_fitness = evaluate_fitness(test_config)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_config = test_config.copy()
    
    # Stage 2: Systematic radius maximization - enhanced with better convergence and more aggressive search
    refined_config = best_config.copy()
    
    # Try to maximize individual radii systematically with better strategy
    improved = True
    max_iter = 18000  # More iterations for better optimization
    iter_count = 0
    
    # Pre-compute spatial information for faster overlap checks
    from scipy.spatial import cKDTree
    
    # Build global KDTree for efficient neighbor searches
    global_tree = cKDTree(refined_config[:, :2])
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Process circles in random order to avoid bias
        circle_order = list(range(n))
        random.shuffle(circle_order)
        
        # Try increasing each circle's radius with more aggressive approach
        for i in circle_order:
            original_radius = refined_config[i, 2]
            # Try to increase radius up to boundary limits
            max_possible_radius = min(
                refined_config[i, 0], 
                1 - refined_config[i, 0], 
                refined_config[i, 1], 
                1 - refined_config[i, 1]
            )
            
            # Use more aggressive step sizing with better convergence strategy
            base_step = 0.008  # Even larger base step for faster exploration
            step_size = base_step * (1.0 + iter_count / 600.0)  # Faster growth rate
            test_radius = original_radius + step_size
            
            # Try multiple steps with better backtracking strategy
            step_attempts = 0
            max_step_attempts = 40  # More attempts for better convergence
            
            while test_radius <= max_possible_radius and step_attempts < max_step_attempts:
                # Optimized overlap check using spatial indexing - more robust version
                valid = True
                
                # Check overlap with a subset of nearby circles for efficiency
                # Use the global tree for better performance
                distances, indices = global_tree.query(refined_config[i, :2], k=min(25, n))  # More neighbors
                nearby_indices = indices[distances < test_radius + max(refined_config[:, 2]) + 0.01]
                
                # Check overlaps with nearby circles
                for j in nearby_indices:
                    if i != j:
                        dist = np.sqrt(
                            (refined_config[i, 0] - refined_config[j, 0])**2 + 
                            (refined_config[i, 1] - refined_config[j, 1])**2
                        )
                        if dist < test_radius + refined_config[j, 2]:
                            valid = False
                            break
                
                if valid:
                    refined_config[i, 2] = test_radius
                    improved = True
                else:
                    # If we can't increase to test_radius, try smaller increments with more careful backtracking
                    if step_attempts > 0:
                        break
                    else:
                        # Reduce step size more aggressively for next attempt
                        step_size *= 0.1  # Even more aggressive reduction
                        test_radius = original_radius + step_size
                        step_attempts += 1
                        continue
                
                test_radius += step_size
                step_attempts += 1
        
        # Update global tree after significant improvements
        if improved and iter_count % 30 == 0:
            global_tree = cKDTree(refined_config[:, :2])
    
    # Stage 3: Final local refinement with better validation and enhanced strategy
    final_config = refined_config.copy()
    
    # Additional local search on the refined result with enhanced strategy
    for iteration in range(12000):  # More iterations for better optimization
        test_config = final_config.copy()
        idx = random.randint(0, n-1)
        
        # Adaptive perturbation with more sophisticated scaling and better convergence
        current_radius = final_config[idx, 2]
        adaptive_step = max(0.001, current_radius * 0.15)  # Larger initial step
        adaptive_step = adaptive_step * (1.0 - iteration / 12000.0)  # Smaller steps as we refine
        
        # Smaller perturbations for final refinement with better balance
        test_config[idx, 0] += random.uniform(-adaptive_step, adaptive_step)
        test_config[idx, 1] += random.uniform(-adaptive_step, adaptive_step)
        test_config[idx, 2] += random.uniform(-adaptive_step*0.03, adaptive_step*0.03)  # Even smaller radius change
        
        # Apply bounds
        test_config[idx, 0] = np.clip(test_config[idx, 0], test_config[idx, 2], 1 - test_config[idx, 2])
        test_config[idx, 1] = np.clip(test_config[idx, 1], test_config[idx, 2], 1 - test_config[idx, 2])
        test_config[idx, 2] = max(0.001, min(0.5, test_config[idx, 2]))
        
        if check_validity(test_config):
            new_fitness = evaluate_fitness(test_config)
            if new_fitness > evaluate_fitness(final_config):
                final_config = test_config.copy()
    
    # Final optimization pass with improved binary search and spatial indexing
    # Use spatial indexing to make this more efficient
    tree = cKDTree(final_config[:, :2])
    
    for i in range(n):
        x, y, r = final_config[i]
        # Binary search for maximum possible radius at this position
        low, high = r, min(x, y, 1-x, 1-y)
        if high - low > 1e-5:
            # Use more precise convergence threshold
            while high - low > 1e-8:  # Even more precise
                mid = (low + high) / 2
                # Check if this radius works with nearby circles (not all)
                valid = True
                distances, indices = tree.query([x, y], k=min(20, n))
                nearby_indices = indices[distances < 1.0]  # Only check relevant neighbors
                
                for j in nearby_indices:
                    if i != j:
                        dist = np.sqrt((x - final_config[j][0])**2 + (y - final_config[j][1])**2)
                        if dist < mid + final_config[j][2]:
                            valid = False
                            break
                if valid:
                    low = mid
                else:
                    high = mid
            final_config[i][2] = low
    
    # Additional enhancement: Run one more round of greedy optimization with spatial indexing
    # This can squeeze out a few more radius units
    tree = cKDTree(final_config[:, :2])  # Build tree once for efficiency
    
    for _ in range(800):  # More iterations
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            max_rad = 0.47
            max_rad = min(max_rad, final_config[i][0])
            max_rad = min(max_rad, 1 - final_config[i][0])
            max_rad = min(max_rad, final_config[i][1])
            max_rad = min(max_rad, 1 - final_config[i][1])
            
            # Only check nearby circles for efficiency
            distances, indices = tree.query(final_config[i, :2], k=min(20, n))  # More neighbors
            nearby_indices = indices[distances < 1.0]  # Relevant neighbors
            
            for j in nearby_indices:
                if i != j:
                    dist = np.sqrt((final_config[i][0] - final_config[j][0])**2 + 
                                 (final_config[i][1] - final_config[j][1])**2)
                    if dist > 0:
                        max_rad = min(max_rad, dist - final_config[j][2])
            
            if max_rad > final_config[i][2] + 1e-6:
                available_space = max_rad - final_config[i][2]
                step_size = min(0.015, available_space * 0.6)  # More aggressive
                final_config[i][2] = min(max_rad, final_config[i][2] + step_size)
                improved = True
    
    return final_config


# EVOLVE-BLOCK-END
