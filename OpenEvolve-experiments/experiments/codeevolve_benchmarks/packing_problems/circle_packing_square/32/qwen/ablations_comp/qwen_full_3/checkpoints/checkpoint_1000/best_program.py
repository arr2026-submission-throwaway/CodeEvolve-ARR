# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a better starting configuration using optimized hexagonal packing
    # Use a more efficient 7x5 hexagonal pattern with better coverage for 32 circles
    circles = np.zeros((n, 3))
    
    # Create a more optimized hexagonal lattice pattern with 7 rows and 5 columns
    rows = 7
    cols = 5
    spacing_x = 0.85 / cols  # Slightly tighter spacing for better packing
    spacing_y = 0.85 / rows  # Slightly tighter spacing for better packing
    
    circle_idx = 0
    for row in range(rows):
        for col in range(cols):
            if circle_idx >= n:
                break
            # Offset odd rows for hexagonal packing with better spacing
            offset = 0.5 if row % 2 == 1 else 0.0
            x = 0.075 + (col + 0.5 + offset) * spacing_x
            y = 0.075 + (row + 0.5) * spacing_y
            # Keep circles within bounds with stricter limits
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on available space with more aggressive scaling
            max_radius = 0.5 * min(x, 1-x, y, 1-y)
            r = min(0.07, max_radius * 0.9)  # Larger initial radius for better starting point
            circles[circle_idx] = [x, y, r]
            circle_idx += 1
            if circle_idx >= n:
                break
    
    # Fill remaining positions with more strategic random placements
    for i in range(circle_idx, n):
        # Place in regions that are more likely to allow larger radii
        # Use a more informed approach with better distribution
        if random.random() < 0.5:  # Increase probability of center placement
            # Place near center with better distribution using normal
            x = np.random.normal(0.5, 0.15)
            y = np.random.normal(0.5, 0.15)
        else:
            # Place in edge regions with more space but with better distribution
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
        
        # Use more aggressive radius calculation
        max_radius = min(x, 1-x, y, 1-y) * 0.4
        r = max_radius * 0.75  # Larger fraction for more room
        # Ensure bounds are respected
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # Optimized constraint checking - more efficient version
    def check_constraints(circles_array):
        """Check if all constraints are satisfied"""
        # Check containment - circles must be fully within square
        x_coords = circles_array[:, 0]
        y_coords = circles_array[:, 1]
        radii = circles_array[:, 2]
        
        # Vectorized containment check
        if not (np.all(x_coords >= radii) and 
                np.all(x_coords <= 1 - radii) and 
                np.all(y_coords >= radii) and 
                np.all(y_coords <= 1 - radii)):
            return False
        
        # Use more efficient overlap checking with early termination
        n = len(circles_array)
        # For small arrays like 32, brute force is often faster than spatial indexing
        for i in range(n):
            x1, y1, r1 = circles_array[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles_array[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                if dist_sq < (r1+r2)**2 - 1e-10:
                    return False
        return True
    
    # Use a more robust approach with scipy optimization
    # First try a simple heuristic approach
    def simple_heuristic():
        # Start with a better configuration
        best_circles = circles.copy()
        best_sum = np.sum(best_circles[:, 2])
        
        # Try some local optimizations
        for _ in range(100):
            # Random perturbations
            test_circles = best_circles.copy()
            
            # Perturb one circle at a time
            for i in range(n):
                # Slightly adjust position and radius
                test_circles[i, 0] += random.uniform(-0.02, 0.02)
                test_circles[i, 1] += random.uniform(-0.02, 0.02)
                test_circles[i, 2] += random.uniform(-0.01, 0.01)
                
                # Keep within bounds
                test_circles[i, 0] = max(0.01, min(0.99, test_circles[i, 0]))
                test_circles[i, 1] = max(0.01, min(0.99, test_circles[i, 1]))
                test_circles[i, 2] = max(0.001, min(0.49, test_circles[i, 2]))
            
            # Check constraints and calculate new sum
            if check_constraints(test_circles):
                new_sum = np.sum(test_circles[:, 2])
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_circles = test_circles.copy()
        
        return best_circles
    
    def check_constraints(circles_array):
        """Check if all constraints are satisfied"""
        # Check containment - circles must be fully within square
        x_coords = circles_array[:, 0]
        y_coords = circles_array[:, 1]
        radii = circles_array[:, 2]
        
        # Vectorized containment check
        if not (np.all(x_coords >= radii) and 
                np.all(x_coords <= 1 - radii) and 
                np.all(y_coords >= radii) and 
                np.all(y_coords <= 1 - radii)):
            return False
        
        # Check non-overlap using optimized approach
        # Precompute all distances for efficiency
        n = len(circles_array)
        if n > 100:  # Use spatial indexing for large numbers of circles
            try:
                points = circles_array[:, :2]
                tree = cKDTree(points)
                # Query pairs efficiently
                pairs = tree.query_pairs(r=0.0001, output_type='ndarray')
                for i, j in pairs:
                    if i < j:  # Only check each pair once
                        x1, y1, r1 = circles_array[i]
                        x2, y2, r2 = circles_array[j]
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        if dist_sq < (r1 + r2)**2 - 1e-10:
                            return False
            except:
                # Fallback to brute force if spatial indexing fails
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1, r1 = circles_array[i]
                        x2, y2, r2 = circles_array[j]
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        if dist_sq < (r1 + r2)**2 - 1e-10:
                            return False
        else:
            # For smaller arrays, brute force is more efficient
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    if dist_sq < (r1 + r2)**2 - 1e-10:
                        return False
        return True
    
    # Apply simple heuristic to improve initial solution
    improved_circles = simple_heuristic()
    
    # More systematic optimization approach - more efficient and aggressive
    final_circles = improved_circles.copy()
    best_sum = np.sum(final_circles[:, 2])
    
    # Single-phase optimization with better strategies and more aggressive search
    max_iterations = 25000  # More iterations for better search
    for iteration in range(max_iterations):
        candidate = final_circles.copy()
        
        # Strategy: Always mutate one circle with adaptive perturbations
        idx = random.randint(0, n-1)
        current_radius = final_circles[idx, 2]
        
        # Adaptive perturbation based on iteration progress and circle properties
        progress = iteration / max_iterations
        edge_distance = min(final_circles[idx, 0], 1-final_circles[idx, 0], 
                           final_circles[idx, 1], 1-final_circles[idx, 1])
        radius_ratio = current_radius / 0.1
        
        # More aggressive early on, conservative later
        if progress < 0.2:
            # Early phase: very aggressive perturbations
            pos_perturb = 0.05
            rad_perturb = 0.025
        elif progress < 0.5:
            # Mid phase: moderate perturbations
            pos_perturb = 0.025
            rad_perturb = 0.012
        else:
            # Late phase: conservative perturbations
            pos_perturb = 0.01
            rad_perturb = 0.005
        
        # Adjust based on circle properties
        if edge_distance < 0.05 or radius_ratio < 0.3:
            # Near edge or small circle - reduce perturbations
            pos_perturb *= 0.6
            rad_perturb *= 0.6
        
        # Apply perturbations with better distribution
        candidate[idx, 0] += random.uniform(-pos_perturb, pos_perturb)
        candidate[idx, 1] += random.uniform(-pos_perturb, pos_perturb)
        candidate[idx, 2] += random.uniform(-rad_perturb, rad_perturb)
        
        # Keep within bounds with better margin handling
        candidate[idx, 0] = max(candidate[idx, 2], min(1 - candidate[idx, 2], candidate[idx, 0]))
        candidate[idx, 1] = max(candidate[idx, 2], min(1 - candidate[idx, 2], candidate[idx, 1]))
        candidate[idx, 2] = max(0.001, min(0.499, candidate[idx, 2]))
        
        # Check if this improves the solution
        if check_constraints(candidate):
            new_sum = np.sum(candidate[:, 2])
            if new_sum > best_sum:
                best_sum = new_sum
                final_circles = candidate.copy()
    
    # Enhanced global optimization with systematic radius expansion
    # Try to improve by increasing radii of circles that can accommodate larger ones
    for _ in range(2000):  # More iterations for better radius optimization
        improved = False
        # Process circles in order of current radius (largest first) to prioritize big gains
        circle_order = sorted(range(n), key=lambda i: final_circles[i, 2], reverse=True)
        
        for i in circle_order:
            original_radius = final_circles[i, 2]
            # Try to increase radius with different increments, starting with larger ones
            increments = [0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003]
            for incr in increments:
                test_radius = min(0.499, original_radius + incr)
                
                # Skip if no meaningful increase
                if abs(test_radius - original_radius) < 1e-6:
                    continue
                    
                # Check if we can increase radius without violating constraints
                temp_circles = final_circles.copy()
                temp_circles[i, 2] = test_radius
                
                if check_constraints(temp_circles):
                    final_circles = temp_circles
                    improved = True
                    break  # Take the first successful increase
        
        if not improved:
            break
    
    # Enhanced refinement with more aggressive pair optimization
    # Try to optimize pairs of circles that are close together for mutual expansion
    for _ in range(1000):  # More iterations for better pair optimization
        # Find pairs of circles that are very close (potential for mutual expansion)
        close_pairs = []
        # Use spatial indexing for efficiency
        try:
            points = final_circles[:, :2]
            tree = cKDTree(points)
            # Find pairs within a reasonable distance
            pairs = tree.query_pairs(r=0.006, output_type='ndarray')
            for i, j in pairs:
                if i < j:  # Only consider each pair once
                    x1, y1, r1 = final_circles[i]
                    x2, y2, r2 = final_circles[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    # Consider pairs that are nearly touching or slightly overlapping
                    if dist_sq < (r1 + r2 + 0.001)**2 and dist_sq > (r1 + r2)**2:
                        close_pairs.append((i, j))
        except:
            # Fallback to brute force if spatial indexing fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = final_circles[i]
                    x2, y2, r2 = final_circles[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    if dist_sq < (r1 + r2 + 0.001)**2 and dist_sq > (r1 + r2)**2:
                        close_pairs.append((i, j))
        
        if not close_pairs:
            break
            
        # Try to improve by adjusting both circles in close pairs
        # Process pairs in order of closeness to prioritize beneficial changes
        sorted_pairs = sorted(close_pairs, key=lambda p: 
                             (final_circles[p[0], 0] - final_circles[p[1], 0])**2 + 
                             (final_circles[p[0], 1] - final_circles[p[1], 1])**2)
        
        # Select more pairs for better exploitation but limit to avoid overfitting
        selected_pairs = sorted_pairs[:min(20, len(sorted_pairs))]
        for i, j in selected_pairs:
            # Try to increase both radii with adaptive increments based on pair closeness
            distance = np.sqrt((final_circles[i, 0] - final_circles[j, 0])**2 + 
                              (final_circles[i, 1] - final_circles[j, 1])**2)
            # More aggressive when closer, less aggressive when farther apart
            base_incr = 0.02
            # Adjust increment inversely to distance (closer pairs get bigger increases)
            adjustment = max(0.3, min(1.0, 1.0 - distance / (final_circles[i, 2] + final_circles[j, 2])))
            incr = base_incr * adjustment
            
            new_r1 = min(0.499, final_circles[i, 2] + incr)
            new_r2 = min(0.499, final_circles[j, 2] + incr)
            
            temp_circles = final_circles.copy()
            temp_circles[i, 2] = new_r1
            temp_circles[j, 2] = new_r2
            
            if check_constraints(temp_circles):
                final_circles = temp_circles
    
    return final_circles


# EVOLVE-BLOCK-END
