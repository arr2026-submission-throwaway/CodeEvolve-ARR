# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from scipy.optimize import minimize
import math
from scipy.spatial import cKDTree

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 26
    
    # Better initialization using hexagonal packing principles with improved radii calculation
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Use a more sophisticated hexagonal pattern for better initial configuration
        # Try 5x5 grid with better spacing for maximum density
        rows, cols = 5, 5
        spacing_x = 0.85 / (cols + 0.5)  # Slightly tighter spacing
        spacing_y = 0.85 / (rows + 0.5) * np.sqrt(3)/2  # Hexagonal packing
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) < n:
                    # Offset every other row for hexagonal packing
                    x_offset = 0 if i % 2 == 0 else spacing_x / 2
                    x = (j + 0.5) * spacing_x + x_offset + 0.075  # Better centering
                    y = (i + 0.5) * spacing_y + 0.075  # Better centering
                    positions.append([x, y])
        
        # Fill remaining positions with strategic corner placement
        remaining_needed = n - len(positions)
        if remaining_needed > 0:
            # Add strategic corner positions
            corner_positions = [
                (0.92, 0.92), (0.08, 0.92), (0.92, 0.08), (0.08, 0.08),
                (0.5, 0.92), (0.92, 0.5), (0.5, 0.08), (0.08, 0.5)
            ]
            for i in range(min(remaining_needed, len(corner_positions))):
                positions.append(corner_positions[i])
        
        # Fill any remaining spots with center positions
        while len(positions) < n:
            positions.append([0.5, 0.5])
        
        # Set initial radii with better estimation based on neighbor relationships
        radii = np.full(n, 0.03)  # Start with slightly larger initial radii
        
        # Improve initial radii calculation based on neighbor distances and boundary constraints
        for i in range(n):
            boundary_min = min(positions[i][0], 1-positions[i][0], positions[i][1], 1-positions[i][1])
            
            # Calculate minimum distance to neighbors more accurately
            neighbor_distances = []
            for j in range(n):
                if i != j:
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    neighbor_distances.append(dist)
            
            if neighbor_distances:
                neighbor_min = min(neighbor_distances)
                # Initial radius should be limited by both boundary and neighbor distances
                radii[i] = min(boundary_min, neighbor_min / 2.5, 0.06)  # Slightly smaller cap
            else:
                radii[i] = min(boundary_min, 0.06)
            
            # Ensure reasonable bounds
            radii[i] = max(0.005, min(0.07, radii[i]))
        
        # Construct circles array
        for i in range(n):
            circles[i] = [positions[i][0], positions[i][1], radii[i]]
        
        return circles
    
    circles = initialize_circles()
    
    # Use optimization approach instead of purely local search for better results
    # This follows the approach from top performers that combine optimization with local search
    
    # Constraint functions - more numerically stable implementation
    def create_constraint_functions():
        """Create constraint functions more efficiently"""
        constraints = []
        
        # Containment constraints - with better numerical tolerance
        for i in range(n):
            # r <= x <= 1-r and r <= y <= 1-r
            # x - r >= 0 (equivalent to x >= r)
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2] + 1e-12})
            # 1 - x - r >= 0 (equivalent to x <= 1-r)
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2] + 1e-12})
            # y - r >= 0 (equivalent to y >= r)
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2] + 1e-12})
            # 1 - y - r >= 0 (equivalent to y <= 1-r)
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2] + 1e-12})
            # r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+2] + 1e-12})
        
        # Non-overlap constraints - more numerically stable
        for i in range(n):
            for j in range(i+1, n):
                # Distance squared >= (r1 + r2)^2 (rearranged for better numerical behavior)
                constraints.append({'type': 'ineq', 'fun': lambda c, i=i, j=j:
                                  (c[3*i] - c[3*j])**2 + (c[3*i+1] - c[3*j+1])**2 - (c[3*i+2] + c[3*j+2])**2 + 1e-12})
        
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Flatten for optimization
    initial_guess = circles.flatten()
    
    # Create constraints more efficiently
    cons = create_constraint_functions()
    
    # More robust optimization with better fallback and tighter tolerances
    try:
        # Try multiple optimization methods with better parameters
        result = minimize(objective, initial_guess, method='SLSQP',
                         constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'eps': 1e-12, 'iprint': -1})
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If SLSQP fails, try L-BFGS-B as fallback with better parameters
            result = minimize(objective, initial_guess, method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'iprint': -1})
            if result.success:
                circles = result.x.reshape(-1, 3)
            else:
                # If both fail, try trust-constr as another option with even tighter tolerances
                result = minimize(objective, initial_guess, method='trust-constr',
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'iprint': -1})
                if result.success:
                    circles = result.x.reshape(-1, 3)
    except Exception as e:
        # If optimization fails completely, proceed with local search
        pass
    
    # Enhanced local search refinement after optimization with improved efficiency and better convergence
    def refine_circles(circles):
        # Convert to separate arrays for efficiency
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # More intelligent refinement approach with improved algorithmic structure
        max_iterations = 2000  # Increased for better convergence
        
        # Use spatial indexing for faster overlap detection
        from scipy.spatial import cKDTree
        tree = None
        
        for iteration in range(max_iterations):
            improved = False
            
            # Update spatial index periodically for better performance
            if iteration % 40 == 0:
                tree = cKDTree(positions)
            
            # Phase 1: Prioritized radius maximization with better neighbor consideration
            # Process circles in order of how constrained they are (more constrained first)
            circle_order = list(range(n))
            # Sort by how close they are to boundary constraints (more constrained first)
            circle_order.sort(key=lambda i: min(positions[i][0], 1-positions[i][0], 
                                               positions[i][1], 1-positions[i][1]), reverse=True)
            
            for i in circle_order:
                old_radius = radii[i]
                new_radius = old_radius
                
                # Calculate maximum possible radius
                max_possible_radius = min(
                    positions[i][0], 
                    1 - positions[i][0],
                    positions[i][1],
                    1 - positions[i][1]
                )
                
                # Check overlap with all other circles efficiently using spatial indexing when available
                safe_radius = max_possible_radius
                if tree is not None:
                    # Find nearby circles (within 2*(old_radius + max_radius) distance)
                    neighbors = tree.query_ball_point(positions[i], 2 * (old_radius + max_possible_radius))
                    for j in neighbors:
                        if i != j:
                            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                         (positions[i][1] - positions[j][1])**2)
                            if dist < radii[j] + safe_radius:
                                max_safe_radius = dist - radii[j]
                                safe_radius = min(safe_radius, max_safe_radius)
                else:
                    # Fallback to full check for first iterations
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                         (positions[i][1] - positions[j][1])**2)
                            if dist < radii[j] + safe_radius:
                                max_safe_radius = dist - radii[j]
                                safe_radius = min(safe_radius, max_safe_radius)
                
                # Limit to valid range with more aggressive approach
                new_radius = min(safe_radius, max_possible_radius)
                
                # Only update if there's meaningful improvement
                if new_radius > old_radius + 1e-8:  # Even tighter threshold for better convergence
                    radii[i] = new_radius
                    improved = True
                    
            # Phase 2: Strategic local search with adaptive perturbations and more thorough checks
            if not improved or iteration % 20 == 0:  # Even more frequent local search
                # Try to improve by moving circles more intelligently
                for _ in range(1000):  # More iterations for better exploration
                    idx = random.randint(0, n-1)
                    
                    # Adaptive perturbation based on current radius with better scaling
                    current_radius = radii[idx]
                    if current_radius < 0.01:  # Very small radii get more aggressive moves
                        dx = random.uniform(-0.05, 0.05)
                        dy = random.uniform(-0.05, 0.05)
                        dr = random.uniform(-0.04, 0.04)
                    elif current_radius < 0.03:  # Small radii
                        dx = random.uniform(-0.04, 0.04)
                        dy = random.uniform(-0.04, 0.04)
                        dr = random.uniform(-0.03, 0.03)
                    elif current_radius < 0.06:  # Medium radii
                        dx = random.uniform(-0.035, 0.035)
                        dy = random.uniform(-0.035, 0.035)
                        dr = random.uniform(-0.025, 0.025)
                    else:  # Larger radii get more conservative moves
                        dx = random.uniform(-0.03, 0.03)
                        dy = random.uniform(-0.03, 0.03)
                        dr = random.uniform(-0.02, 0.02)
                    
                    new_x = positions[idx][0] + dx
                    new_y = positions[idx][1] + dy
                    new_r = radii[idx] + dr
                    
                    # Check bounds and validity
                    if (new_x >= new_r and new_x <= 1-new_r and 
                        new_y >= new_r and new_y <= 1-new_r and 
                        new_r > 0):
                        # Check overlaps efficiently with early termination
                        valid = True
                        if tree is not None:
                            # Use spatial indexing for neighbor search with broader radius check
                            neighbors = tree.query_ball_point([new_x, new_y], 2 * (new_r + max(radii)))
                            for j in neighbors:
                                if j != idx:
                                    dist = np.sqrt((new_x - positions[j][0])**2 + (new_y - positions[j][1])**2)
                                    if dist < new_r + radii[j]:
                                        valid = False
                                        break
                        else:
                            # Fallback to direct checking with more neighbors for better accuracy
                            distances = []
                            for j in range(n):
                                if j != idx:
                                    dist = np.sqrt((new_x - positions[j][0])**2 + (new_y - positions[j][1])**2)
                                    distances.append((dist, j))
                            
                            distances.sort()
                            # Check more neighbors for better accuracy
                            for dist, j in distances[:min(15, len(distances))]:  # Increased from 12 to 15
                                if dist < new_r + radii[j]:
                                    valid = False
                                    break
                        
                        if valid:
                            # Update if this improves the total sum
                            temp_radii = radii.copy()
                            temp_radii[idx] = new_r
                            temp_positions = positions.copy()
                            temp_positions[idx] = [new_x, new_y]
                            
                            new_sum = np.sum(temp_radii)
                            if new_sum > np.sum(radii):
                                positions[idx] = [new_x, new_y]
                                radii[idx] = new_r
                                improved = True
        
        # Final optimization pass with more aggressive refinement
        best_positions = positions.copy()
        best_radii = radii.copy()
        
        # Additional focused passes with even tighter constraints and more iterations
        for pass_num in range(6):  # More passes for better convergence
            for _ in range(400):  # More iterations per pass for better refinement
                improved = False
                for i in range(n):
                    old_radius = best_radii[i]
                    new_radius = old_radius
                    
                    # Calculate maximum possible radius
                    max_possible_radius = min(
                        best_positions[i][0], 
                        1 - best_positions[i][0],
                        best_positions[i][1],
                        1 - best_positions[i][1]
                    )
                    
                    # Check overlap with all other circles efficiently using spatial indexing
                    safe_radius = max_possible_radius
                    if tree is not None:
                        neighbors = tree.query_ball_point(best_positions[i], 2 * (old_radius + max_possible_radius))
                        for j in neighbors:
                            if i != j:
                                dist = np.sqrt((best_positions[i][0] - best_positions[j][0])**2 + 
                                             (best_positions[i][1] - best_positions[j][1])**2)
                                if dist < best_radii[j] + safe_radius:
                                    max_safe_radius = dist - best_radii[j]
                                    safe_radius = min(safe_radius, max_safe_radius)
                    else:
                        # Fallback to direct checking with more thorough neighbor consideration
                        for j in range(n):
                            if i != j:
                                dist = np.sqrt((best_positions[i][0] - best_positions[j][0])**2 + 
                                             (best_positions[i][1] - best_positions[j][1])**2)
                                if dist < best_radii[j] + safe_radius:
                                    max_safe_radius = dist - best_radii[j]
                                    safe_radius = min(safe_radius, max_safe_radius)
                    
                    new_radius = min(safe_radius, max_possible_radius)
                    
                    if new_radius > old_radius + 1e-8:  # Even tighter threshold for better convergence
                        best_radii[i] = new_radius
                        improved = True
                
                if not improved:
                    break
        
        # Fill result array
        result = circles.copy()
        for i in range(n):
            result[i] = [best_positions[i][0], best_positions[i][1], best_radii[i]]
            
        return result
    
    # Apply refinement
    circles = refine_circles(circles)
    
    return circles


# EVOLVE-BLOCK-END
