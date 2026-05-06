# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0 or np.allclose(distances, 0):
            return -np.inf
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or very small numbers
        if d_max <= 1e-12:
            return -np.inf
            
        # Return negative because we want to maximize
        # Add small epsilon to prevent numerical issues
        return -d_min / (d_max + 1e-15)
    
    def constraint(x_flat):
        """Constraint function to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Return 0 when constraints are satisfied (points within bounds)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    # Try multiple initial configurations and use the best result
    best_result = None
    best_ratio = -np.inf
    
    np.random.seed(42)
    
    # Try different initial configurations - focus on more promising patterns
    for attempt in range(6):
        # Create diverse initial configurations
        if attempt == 0:
            # Improved 4x4 grid with better spacing and perturbations
            grid_x = np.linspace(0.1, 0.9, 4)
            grid_y = np.linspace(0.1, 0.9, 4)
            initial_points = []
            for i in range(4):
                for j in range(4):
                    initial_points.append([grid_x[i], grid_y[j]])
            initial_points = np.array(initial_points) + np.random.normal(0, 0.02, (16, 2))
        elif attempt == 1:
            # More random but structured - better spread
            initial_points = np.random.uniform(0.05, 0.95, (16, 2))
        elif attempt == 2:
            # Better hexagonal arrangement with proper spacing
            angles = np.linspace(0, 2*np.pi*2, 16)
            radii = np.linspace(0.15, 0.4, 16)
            initial_points = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
            initial_points += 0.5  # Center in unit square
            initial_points += np.random.normal(0, 0.015, (16, 2))
        elif attempt == 3:
            # Spiral arrangement with better distribution
            angles = np.linspace(0, 2*np.pi*3, 16)
            radii = np.linspace(0.1, 0.45, 16)
            initial_points = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
            initial_points += 0.5
            initial_points += np.random.normal(0, 0.015, (16, 2))
        elif attempt == 4:
            # Golden ratio spiral arrangement - more exotic but potentially better
            # This creates a more uniform distribution by using golden angle
            golden_angle = 2.399963229728653  # ~2π(1-φ) where φ is golden ratio
            initial_points = []
            for i in range(16):
                r = 0.4 * np.sqrt(i / 15.0)  # Radial distance
                theta = i * golden_angle     # Angular position
                x = 0.5 + r * np.cos(theta)  # Convert to Cartesian
                y = 0.5 + r * np.sin(theta)
                initial_points.append([x, y])
            initial_points = np.array(initial_points) + np.random.normal(0, 0.01, (16, 2))
        else:
            # Concentrated pattern with better even spacing
            # Generate points more evenly distributed
            initial_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    # Add jitter to avoid perfect grid
                    x += np.random.normal(0, 0.01, 1)[0]
                    y += np.random.normal(0, 0.01, 1)[0]
                    initial_points.append([x, y])
            initial_points = np.array(initial_points)
        
        # Clip to ensure points stay in bounds
        initial_points = np.clip(initial_points, 0, 1)
        
        # Flatten for optimization
        x0 = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint}
        
        # Try multiple optimization methods for robustness
        methods_to_try = ['L-BFGS-B', 'SLSQP', 'trust-constr']
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method=method, 
                    bounds=bounds, 
                    constraints=cons, 
                    options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    
                    # Validate the result
                    distances = pdist(final_points)
                    if len(distances) > 0 and np.max(distances) > 1e-12:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        ratio = min_dist / max_dist
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_result = final_points.copy()
                            
            except Exception:
                continue
    
    # If no good result from local optimization, try differential evolution with better parameters
    if best_result is None:
        try:
            # Create a good initial configuration for DE with more exploration
            # Use a combination approach for better starting point
            grid_x = np.linspace(0.1, 0.9, 4)
            grid_y = np.linspace(0.1, 0.9, 4)
            initial_points = []
            for i in range(4):
                for j in range(4):
                    initial_points.append([grid_x[i], grid_y[j]])
            # Add larger perturbations to get better exploration
            initial_points = np.array(initial_points) + np.random.normal(0, 0.02, (16, 2))
            initial_points = np.clip(initial_points, 0, 1)
            
            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Use differential evolution with even more aggressive parameters for global search
            result_de = differential_evolution(
                objective, 
                bounds, 
                maxiter=250,  # Even more iterations for better search
                popsize=35,   # Even larger population for better exploration
                seed=42,
                tol=1e-13,    # Even tighter tolerance
                recombination=0.97,  # Very high recombination rate
                atol=1e-13,   # Absolute tolerance
                mutation=(0.9, 1.0)  # Very aggressive mutation
            )
            
            if result_de.success:
                final_points = result_de.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                return final_points
                
        except Exception:
            pass
    
    # Final fallback to the best initial configuration with additional refinement
    if best_result is None:
        # Try a more sophisticated fallback approach with multiple refinements
        # Use a known good configuration and refine it with multiple approaches
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        initial_points = []
        for i in range(4):
            for j in range(4):
                initial_points.append([grid_x[i], grid_y[j]])
        initial_points = np.array(initial_points) + np.random.normal(0, 0.01, (16, 2))
        initial_points = np.clip(initial_points, 0, 1)
        
        # Run multiple final optimizations with different methods for robustness
        x0 = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        cons = {'type': 'ineq', 'fun': constraint}
        
        # Try L-BFGS-B first
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                return np.clip(final_points, 0, 1)
        except:
            pass
            
        # Try trust-constr as backup
        try:
            result = minimize(
                objective, 
                x0, 
                method='trust-constr', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 1000, 'gtol': 1e-13, 'xtol': 1e-13}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                return np.clip(final_points, 0, 1)
        except:
            pass
            
        return initial_points
    
    return best_result


# EVOLVE-BLOCK-END
