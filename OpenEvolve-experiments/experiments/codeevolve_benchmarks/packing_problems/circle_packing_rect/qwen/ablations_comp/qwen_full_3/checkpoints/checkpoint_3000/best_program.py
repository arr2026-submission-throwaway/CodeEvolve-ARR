# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try multiple aspect ratios to find optimal packing
    # Based on circle packing theory, different ratios work better for different numbers
    aspect_ratios = [(2.0, 1.0), (1.5, 0.5), (1.0, 1.0), (0.5, 1.5), (1.2, 0.8)]
    best_ratio = (2.0, 1.0)  # Try the most promising 2:1 ratio first
    best_sum = 0
    best_circles = None
    
    # Try different aspect ratios and pick the best
    # Focus on ratios that are likely to work well for circle packing
    expanded_aspect_ratios = [(2.0, 1.0), (1.8, 1.0), (1.6, 1.0), (1.4, 1.0), (1.2, 1.0),
                             (1.0, 1.0), (1.0, 1.2), (1.0, 1.4), (1.0, 1.6), (1.0, 1.8),
                             (1.0, 2.0), (1.2, 2.0), (1.5, 2.0), (2.0, 2.0), (2.5, 2.0)]
    
    for width_ratio, height_ratio in expanded_aspect_ratios:
        rect_width = width_ratio * 2.0 / (width_ratio + height_ratio)
        rect_height = height_ratio * 2.0 / (width_ratio + height_ratio)
        
        # Initialize with a better starting configuration using hexagonal packing
        n = 21
        circles = np.zeros((n, 3))
        
        # Use a more mathematically sound hexagonal packing approach
        rows = 5
        cols = 5
        
        # Calculate spacing based on optimal hexagonal packing density
        # Use a more aggressive initial estimate to allow for better optimization
        estimated_radius = min(rect_width, rect_height) * 0.125  # Slightly larger for more room to grow
        spacing_x = 2 * estimated_radius
        spacing_y = 2 * estimated_radius * np.sqrt(3) / 2  # Vertical spacing for hexagonal packing
        
        # Adjust spacing to fit within container dimensions
        actual_cols = max(1, int(rect_width / spacing_x))
        actual_rows = max(1, int(rect_height / spacing_y))
        
        # Generate hexagonal grid with proper centering
        idx = 0
        for i in range(actual_rows):
            for j in range(actual_cols):
                if idx >= n:
                    break
                # Offset odd rows for hexagonal packing
                x_offset = (i % 2) * spacing_x / 2
                x = x_offset + j * spacing_x + estimated_radius
                y = i * spacing_y + estimated_radius
                
                # Ensure we stay within bounds
                if x - estimated_radius >= 0 and x + estimated_radius <= rect_width and \
                   y - estimated_radius >= 0 and y + estimated_radius <= rect_height:
                    circles[idx] = [x, y, estimated_radius]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining circles with smart placement
        for i in range(idx, n):
            # Try corner placements first
            edge_positions = [
                (estimated_radius, estimated_radius),  # bottom-left
                (rect_width - estimated_radius, estimated_radius),  # bottom-right
                (estimated_radius, rect_height - estimated_radius),  # top-left
                (rect_width - estimated_radius, rect_height - estimated_radius),  # top-right
            ]
            
            placed = False
            for x, y in edge_positions:
                if not placed:
                    r = estimated_radius * 0.7  # Slightly smaller radius
                    valid = True
                    # Check overlap with all existing circles
                    for k in range(i):
                        existing_x, existing_y, existing_r = circles[k]
                        dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if dist < (r + existing_r):
                            valid = False
                            break
                    
                    if valid:
                        circles[i] = [x, y, r]
                        placed = True
                        break
            
            if not placed:
                # Fallback to random placement with better validation
                attempts = 0
                while attempts < 50:  # Reduced attempts for efficiency
                    x = random.uniform(estimated_radius, rect_width - estimated_radius)
                    y = random.uniform(estimated_radius, rect_height - estimated_radius)
                    r = estimated_radius * 0.6
                    
                    # Check if this circle overlaps with existing ones
                    valid = True
                    for k in range(i):
                        existing_x, existing_y, existing_r = circles[k]
                        dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if dist < (r + existing_r):
                            valid = False
                            break
                    
                    if valid:
                        circles[i] = [x, y, r]
                        placed = True
                        break
                    attempts += 1
            
            # If still not placed, put in center with very small radius
            if i >= n - 1 or not placed:
                circles[i] = [rect_width/2, rect_height/2, estimated_radius/6]
        
        # Test this configuration
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_ratio = (width_ratio, height_ratio)
            best_circles = circles.copy()
    
    # Use the best configuration found
    rect_width = best_ratio[0] * 2.0 / (best_ratio[0] + best_ratio[1])
    rect_height = best_ratio[1] * 2.0 / (best_ratio[0] + best_ratio[1])
    circles = best_circles
    
    # Refine using a more sophisticated optimization approach
    # Use a hybrid method: first global search then local refinement
    
    # Flatten initial configuration
    initial_params = []
    for i in range(n):
        initial_params.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Better optimization approach with more efficient constraint checking
    def evaluate_constraints(params):
        """Check if all constraints are satisfied"""
        # Early exit for boundary constraints - fastest check
        for i in range(0, len(params), 3):
            x, y, r = params[i], params[i+1], params[i+2]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Check overlap constraints with early termination for performance
        # Use vectorized approach for better efficiency when possible
        n_circles = len(params) // 3
        if n_circles < 2:
            return True
            
        # Convert to numpy arrays for vectorized operations
        coords = np.array([(params[i], params[i+1]) for i in range(0, len(params), 3)])
        radii = np.array([params[i+2] for i in range(0, len(params), 3)])
        
        # Vectorized overlap checking using broadcasting for efficiency
        # Compute pairwise distances using vectorized operations
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances_squared = np.sum(diff**2, axis=2)
        min_distances_squared = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
        
        # Check if any circles overlap (distance < min_distance)
        overlap_mask = distances_squared < min_distances_squared
        np.fill_diagonal(overlap_mask, False)  # Ignore self-overlaps
        
        # Early termination if any overlap found
        if np.any(overlap_mask):
            return False
            
        return True
    
    def objective(params):
        # We want to maximize sum of radii, so minimize negative sum
        total_radius = 0
        for i in range(2, len(params), 3):  # Only sum radii (indices 2,5,8...)
            total_radius += params[i]
        return -total_radius
    
    # Create bounds for optimization
    bounds = []
    for i in range(n):
        bounds.append((0, rect_width))      # x bounds
        bounds.append((0, rect_height))     # y bounds
        bounds.append((0.001, min(rect_width, rect_height)/2))  # radius bounds
    
    # Use a more robust optimization approach with constraint checking
    best_solution = initial_params.copy()
    best_sum = sum(initial_params[2::3])  # Sum of radii
    
    # Enhanced optimization approach with fewer restarts and simpler stages
    # Use fewer restarts but more focused optimization
    for restart in range(10):  # Fewer restarts to save time
        # Reset to initial solution for each restart
        initial_params = best_solution.copy()
        
        # Simplified two-stage optimization approach
        # Stage 1: Aggressive global search
        for iteration in range(2000):
            # Create a new candidate solution
            new_params = initial_params.copy()
            
            # Perturb parameters with balanced approach
            for i in range(0, len(new_params), 3):
                # Adjust x position
                new_params[i] = max(0.001, min(rect_width - 0.001, new_params[i] + random.gauss(0, 0.15)))
                # Adjust y position
                new_params[i+1] = max(0.001, min(rect_height - 0.001, new_params[i+1] + random.gauss(0, 0.15)))
                # Adjust radius
                new_params[i+2] = max(0.001, min(min(rect_width, rect_height)/2, new_params[i+2] + random.gauss(0, 0.08)))
            
            # Validate constraints efficiently
            if evaluate_constraints(new_params):
                new_sum = sum(new_params[2::3])
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_solution = new_params.copy()
            
            # Update for next iteration
            initial_params = new_params.copy()
        
        # Stage 2: Fine-tuning with smaller perturbations
        for iteration in range(1000):
            # Create a new candidate solution
            new_params = initial_params.copy()
            
            # Perturb parameters with smaller steps
            for i in range(0, len(new_params), 3):
                # Adjust x position
                new_params[i] = max(0.001, min(rect_width - 0.001, new_params[i] + random.gauss(0, 0.05)))
                # Adjust y position
                new_params[i+1] = max(0.001, min(rect_height - 0.001, new_params[i+1] + random.gauss(0, 0.05)))
                # Adjust radius
                new_params[i+2] = max(0.001, min(min(rect_width, rect_height)/2, new_params[i+2] + random.gauss(0, 0.02)))
            
            # Validate constraints efficiently
            if evaluate_constraints(new_params):
                new_sum = sum(new_params[2::3])
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_solution = new_params.copy()
            
            # Update for next iteration
            initial_params = new_params.copy()
    
    # Enhanced final refinement with adaptive local search
    # Perform additional local optimization on the best solution found
    local_best = best_solution.copy()
    local_sum = best_sum
    
    # Run rounds of adaptive local search with decreasing step sizes
    # Increase rounds and make more aggressive adjustments in early phases
    for round_num in range(800):  # More rounds for better optimization
        # Make small perturbations to all parameters
        test_params = local_best.copy()
        # Use more aggressive initial phase with gradual decay
        step_size = max(0.0001, 0.03 * (1 - round_num / 800.0))  # Gradually decrease step size
        
        for i in range(0, len(test_params), 3):
            # Small adjustments to all parameters with adaptive step size
            test_params[i] = max(0.001, min(rect_width - 0.001, test_params[i] + random.gauss(0, step_size)))
            test_params[i+1] = max(0.001, min(rect_height - 0.001, test_params[i+1] + random.gauss(0, step_size)))
            test_params[i+2] = max(0.001, min(min(rect_width, rect_height)/2, test_params[i+2] + random.gauss(0, step_size * 0.4)))
        
        if evaluate_constraints(test_params):
            test_sum = sum(test_params[2::3])
            if test_sum > local_sum:
                local_sum = test_sum
                local_best = test_params.copy()
    
    # Simplified final refinement - just do basic local search
    # This is more efficient and often sufficient for this problem
    local_best = best_solution.copy()
    local_sum = best_sum
    
    # Simple local search with fewer iterations
    for iteration in range(500):
        # Make small perturbations to all parameters
        test_params = local_best.copy()
        
        # Apply small random perturbations
        for i in range(0, len(test_params), 3):
            # Small adjustments to all parameters
            test_params[i] = max(0.001, min(rect_width - 0.001, test_params[i] + random.gauss(0, 0.02)))
            test_params[i+1] = max(0.001, min(rect_height - 0.001, test_params[i+1] + random.gauss(0, 0.02)))
            test_params[i+2] = max(0.001, min(min(rect_width, rect_height)/2, test_params[i+2] + random.gauss(0, 0.01)))
        
        if evaluate_constraints(test_params):
            test_sum = sum(test_params[2::3])
            if test_sum > local_sum:
                local_sum = test_sum
                local_best = test_params.copy()
    
    # Return the improved solution
    final_circles = np.zeros((n, 3))
    for i in range(n):
        final_circles[i] = [local_best[i*3], local_best[i*3+1], local_best[i*3+2]]
    
    return final_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
