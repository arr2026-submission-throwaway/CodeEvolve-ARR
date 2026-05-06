# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a better starting configuration using Fibonacci spiral
    # This often produces better initial configurations than grid-based approaches
    golden_ratio = (1 + np.sqrt(5)) / 2.0
    circles = np.zeros((n, 3))
    
    # Generate points using Fibonacci spiral for better distribution
    positions = []
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        x = 0.5 + 0.45 * np.sin(phi) * np.cos(theta)
        y = 0.5 + 0.45 * np.sin(phi) * np.sin(theta)
        positions.append([x, y])
    
    # Set initial positions with reasonable starting radii
    for i in range(n):
        circles[i][0] = positions[i][0]  # x
        circles[i][1] = positions[i][1]  # y
        # Start with a more reasonable initial radius to allow more optimization space
        circles[i][2] = 0.02
    
    # Phase 1: Global optimization using scipy minimize with proper constraints
    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(0, len(vars), 3):
            total_radius += vars[i+2]  # Sum of all radii
        return -total_radius  # Negative because we want to maximize
    
    def constraint_function(vars):
        # Check containment and non-overlap constraints
        cons = []
        
        # Convert vars back to circles array
        temp_circles = []
        for i in range(0, len(vars), 3):
            temp_circles.append([vars[i], vars[i+1], vars[i+2]])
        
        # Containment constraints: each circle must be fully inside unit square
        for i in range(len(temp_circles)):
            x, y, r = temp_circles[i]
            # Circle must be within bounds with margin for numerical stability
            cons.append(x - r - 1e-6)  # x - r >= 1e-6
            cons.append(y - r - 1e-6)  # y - r >= 1e-6
            cons.append(1 - x - r - 1e-6)  # 1 - x - r >= 1e-6
            cons.append(1 - y - r - 1e-6)  # 1 - y - r >= 1e-6
        
        # Non-overlap constraints: distance between centers >= sum of radii
        # Using squared distances for better performance
        for i in range(len(temp_circles)):
            for j in range(i+1, len(temp_circles)):
                x1, y1, r1 = temp_circles[i]
                x2, y2, r2 = temp_circles[j]
                # Use squared distance to avoid sqrt computation when possible
                dx = x1 - x2
                dy = y1 - y2
                dist_squared = dx*dx + dy*dy
                # dist >= r1 + r2 (equivalent to dist^2 >= (r1+r2)^2)
                cons.append(dist_squared - (r1 + r2 + 1e-8)**2)
        
        return np.array(cons)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((1e-6, 1-1e-6))
        # y coordinate bounds  
        bounds.append((1e-6, 1-1e-6))
        # radius bounds
        bounds.append((1e-6, 0.5))
    
    # Run optimization with multiple attempts for better results
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Try multiple optimization runs with different initializations
    for attempt in range(3):  # Reduced attempts but with better approach
        attempt_circles = circles.copy()
        if attempt > 0:
            # Add small random perturbations to positions and radii
            for i in range(n):
                attempt_circles[i][0] += random.uniform(-0.01, 0.01)
                attempt_circles[i][1] += random.uniform(-0.01, 0.01)
                attempt_circles[i][2] *= random.uniform(0.95, 1.05)
        
        try:
            # Start with our attempt solution
            initial_vars = []
            for i in range(n):
                initial_vars.extend([attempt_circles[i][0], attempt_circles[i][1], attempt_circles[i][2]])
            
            # Use scipy.optimize with method='SLSQP' for constrained optimization
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda x: constraint_function(x)},
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
            )
            
            if result.success:
                # Extract optimized values
                temp_circles = attempt_circles.copy()
                for i in range(n):
                    temp_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                
                # Check if this solution is better
                current_sum = np.sum(temp_circles[:, 2])
                if current_sum > best_sum:
                    best_circles = temp_circles.copy()
                    best_sum = current_sum
                    
        except Exception:
            # Continue with next attempt if this one fails
            continue
    
    # Use the best result so far
    circles = best_circles.copy()
    
    # Phase 2: Enhanced local refinement with smarter search strategies
    max_refinement_iterations = 200  # Reduced iterations but with better strategies
    
    for iteration in range(max_refinement_iterations):
        improved = False
        # Process circles in random order for better exploration
        circle_indices = list(range(n))
        random.shuffle(circle_indices)
        
        for i in circle_indices:
            # Store original values
            orig_x, orig_y, orig_r = circles[i][0], circles[i][1], circles[i][2]
            
            # Try more sophisticated perturbation strategies with adaptive step sizes
            best_radius = orig_r
            best_x, best_y = orig_x, orig_y
            
            # Strategy 1: Multi-scale grid-based perturbations with more diverse steps
            perturbations = []
            step_sizes = [0.08, 0.06, 0.04, 0.02, 0.01]
            for step in step_sizes:
                for dx in [-step, -step/2, 0, step/2, step]:
                    for dy in [-step, -step/2, 0, step/2, step]:
                        if abs(dx) + abs(dy) > 0:  # Skip center
                            perturbations.append((dx, dy))
            
            # Strategy 2: Random perturbations with varied ranges
            for _ in range(30):
                perturbations.append((random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)))
            
            # Strategy 3: Boundary-aware moves
            boundary_moves = [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]
            for dx, dy in boundary_moves:
                perturbations.append((dx, dy))
            
            # Strategy 4: Directional expansion moves
            expansion_directions = [(0, 0.02), (0.02, 0), (0, -0.02), (-0.02, 0)]
            for _ in range(10):
                dx, dy = random.choice(expansion_directions)
                perturbations.append((dx, dy))
            
            for dx, dy in perturbations:
                new_x = max(1e-6, min(1-1e-6, orig_x + dx))
                new_y = max(1e-6, min(1-1e-6, orig_y + dy))
                
                # Calculate new maximum radius more efficiently
                max_radius = min(
                    new_x, 1-new_x,
                    new_y, 1-new_y
                )
                
                # Check overlap with all other circles more efficiently
                overlap_found = False
                # Use squared distances for better performance
                for j in range(n):
                    if i != j:
                        dx = new_x - circles[j][0]
                        dy = new_y - circles[j][1]
                        dist_squared = dx*dx + dy*dy
                        # Compare with squared distance to avoid sqrt
                        if dist_squared < (circles[j][2] + 1e-8)**2:
                            overlap_found = True
                            break
                        # Update max_radius with actual distance calculation
                        max_radius = min(max_radius, np.sqrt(dist_squared) - circles[j][2])
                
                # Only accept valid positions without overlaps
                if not overlap_found and max_radius > best_radius:
                    best_radius = max_radius
                    best_x, best_y = new_x, new_y
                    improved = True
            
            # Update if improvement found
            if improved:
                circles[i][0] = best_x
                circles[i][1] = best_y
                circles[i][2] = best_radius
        
        # Early stopping if no improvement
        if not improved:
            break
    
    # Phase 3: Final constraint enforcement with minimal polishing
    # Ensure all constraints are strictly met with efficient final refinement
    for i in range(n):
        # Ensure radius is valid and within bounds
        max_radius = min(
            circles[i][0], 1-circles[i][0],
            circles[i][1], 1-circles[i][1]
        )
        
        # Check overlap with all other circles and adjust radius accordingly
        for j in range(n):
            if i != j:
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                # Allow for small numerical tolerance
                if dist < circles[j][2] + 1e-6:
                    max_radius = min(max_radius, dist - circles[j][2] - 1e-6)
                else:
                    max_radius = min(max_radius, dist - circles[j][2])
        
        circles[i][2] = max(1e-6, max_radius)
    
    # Final lightweight optimization - just a few passes to clean up
    for _ in range(50):
        improved = False
        # Process circles in shuffled order
        circle_indices = list(range(n))
        random.shuffle(circle_indices)
        
        for i in circle_indices:
            orig_x, orig_y, orig_r = circles[i][0], circles[i][1], circles[i][2]
            
            # Try focused moves for final cleanup
            best_radius = orig_r
            best_x, best_y = orig_x, orig_y
            
            # Small adjustments only for final polish
            moves = [
                (0, 0.005), (0, -0.005), (0.005, 0), (-0.005, 0),
                (0.003, 0.003), (-0.003, 0.003), (0.003, -0.003), (-0.003, -0.003)
            ]
            
            for dx, dy in moves:
                new_x = max(1e-6, min(1-1e-6, orig_x + dx))
                new_y = max(1e-6, min(1-1e-6, orig_y + dy))
                
                # Calculate new maximum radius
                max_radius = min(
                    new_x, 1-new_x,
                    new_y, 1-new_y
                )
                
                # Check overlap with all other circles more efficiently
                overlap_found = False
                # Use squared distances for better performance
                for j in range(n):
                    if i != j:
                        dx = new_x - circles[j][0]
                        dy = new_y - circles[j][1]
                        dist_squared = dx*dx + dy*dy
                        if dist_squared < (circles[j][2] + 1e-8)**2:
                            overlap_found = True
                            break
                        dist = np.sqrt(dist_squared)
                        max_radius = min(max_radius, dist - circles[j][2])
                
                if not overlap_found and max_radius > best_radius:
                    best_radius = max_radius
                    best_x, best_y = new_x, new_y
                    improved = True
            
            if improved:
                circles[i][0] = best_x
                circles[i][1] = best_y
                circles[i][2] = best_radius
    
    return circles


# EVOLVE-BLOCK-END
