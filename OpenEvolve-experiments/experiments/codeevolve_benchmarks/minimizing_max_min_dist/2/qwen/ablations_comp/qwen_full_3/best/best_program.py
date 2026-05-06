# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio (since we want to maximize)
        if d_max > 0:
            return -d_min / d_max
        else:
            return 0
    
    # Try multiple initial configurations and pick the best
    best_points = None
    best_ratio = -np.inf
    
    # Configuration 1: Improved hexagonal lattice with better normalization and spacing
    points_hex = []
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            if i % 2 == 0:
                x = j + 0.5
                y = i * np.sqrt(3)/2
            else:
                x = j + 1
                y = i * np.sqrt(3)/2
            points_hex.append([x, y])
    points_hex = np.array(points_hex[:16])
    # Better normalization to fit in [0,1] square properly with more careful scaling
    if points_hex[:, 0].max() != points_hex[:, 0].min():
        points_hex[:, 0] = (points_hex[:, 0] - points_hex[:, 0].min()) / (points_hex[:, 0].max() - points_hex[:, 0].min()) * 0.8 + 0.1
    if points_hex[:, 1].max() != points_hex[:, 1].min():
        points_hex[:, 1] = (points_hex[:, 1] - points_hex[:, 1].min()) / (points_hex[:, 1].max() - points_hex[:, 1].min()) * 0.8 + 0.1
    hex_points = points_hex.copy()
    
    # Configuration 2: Golden spiral with improved parameters and better distribution
    golden_angle = 2 * np.pi * (1 - (1 + np.sqrt(5)) / 2)  # Golden ratio conjugate
    golden_points = []
    for i in range(16):
        r = np.sqrt(i / 15.0) * 0.45  # Slightly larger radius
        theta = i * golden_angle
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        golden_points.append([x, y])
    golden_points = np.array(golden_points)
    golden_points = np.clip(golden_points, 0, 1)
    
    # Configuration 3: Improved circular arrangement with better uniformity and radius control
    angles = np.linspace(0, 2*np.pi, 17)[:-1]  # 16 angles
    radii = 0.35 + 0.25 * np.sin(np.arange(16) * np.pi / 8)  # More varied radius
    circle_points = np.column_stack([
        0.5 + radii * np.cos(angles),
        0.5 + radii * np.sin(angles)
    ])
    circle_points = np.clip(circle_points, 0, 1)
    
    # Configuration 4: Improved grid with better jitter control
    points_grid = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 + np.random.normal(0, 0.03)  # Even smaller jitter for better stability
            y = j + 0.5 + np.random.normal(0, 0.03)
            points_grid.append([x, y])
    grid_points = np.array(points_grid[:16])
    grid_points = np.clip(grid_points, 0, 1)
    
    # Configuration 5: Latin hypercube with better edge avoidance and spacing
    np.random.seed(42)
    latin_points = []
    # Create more evenly distributed samples with better spacing and edge avoidance
    for i in range(16):
        x = np.random.uniform(0.12, 0.88)  # Slightly tighter bounds
        y = np.random.uniform(0.12, 0.88)
        latin_points.append([x, y])
    latin_points = np.array(latin_points)
    
    # Test multiple initial configurations with better optimization strategy
    initial_configs = [hex_points, golden_points, circle_points, grid_points, latin_points]
    
    # More aggressive multi-start optimization with better diversity
    for i, initial_config in enumerate(initial_configs):
        # Try multiple random perturbations of each initial config
        for perturbation_round in range(3):  # Fewer perturbations to save time
            # Add random perturbation with varying intensity
            np.random.seed(42 + i * 10 + perturbation_round)
            perturbed_config = initial_config.copy()
            if perturbation_round == 0:
                # Light perturbation for fine-tuning
                perturbed_config += np.random.normal(0, 0.015, initial_config.shape)
            elif perturbation_round == 1:
                # Medium perturbation for exploration
                perturbed_config += np.random.normal(0, 0.04, initial_config.shape)
            else:
                # Heavy perturbation for global exploration
                perturbed_config += np.random.normal(0, 0.08, initial_config.shape)
            perturbed_config = np.clip(perturbed_config, 0, 1)
            
            # Use multiple optimization methods for better results
            methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell']
            for method in methods:
                try:
                    bounds = [(0, 1) for _ in range(32)]
                    # Use more aggressive optimization parameters
                    if method == 'trust-constr':
                        options = {'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-13}
                    else:
                        options = {'maxiter': 1000, 'ftol': 1e-11}
                    result = minimize(
                        objective, 
                        perturbed_config.flatten(), 
                        method=method,
                        bounds=bounds,
                        options=options
                    )
                    
                    if result.success:
                        # Extract optimized points
                        optimized_points = result.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        
                        # Evaluate the final result
                        distances = pdist(optimized_points)
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                
                except Exception:
                    continue
    
    # If no optimization worked, try differential evolution for global search with better parameters
    if best_points is None:
        try:
            # Use differential evolution for global optimization with more iterations
            def global_objective(params):
                points_test = params.reshape(-1, 2)
                points_test = np.clip(points_test, 0, 1)
                distances = pdist(points_test)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 0
                return -min_dist / max_dist
            
            # Try multiple global optimization approaches
            from scipy.optimize import differential_evolution, shgo
            
            # First try differential evolution with more iterations and better parameters
            result = differential_evolution(
                global_objective, 
                bounds=[(0, 1) for _ in range(32)],
                maxiter=400,
                popsize=60,
                seed=42,
                tol=1e-13,
                mutation=(0.8, 1.2),
                recombination=0.8,
                polish=True
            )
            
            if result.success:
                best_points = result.x.reshape(-1, 2)
                best_points = np.clip(best_points, 0, 1)
                
                # Final evaluation
                distances = pdist(best_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    best_ratio = min_dist / max_dist
            else:
                # Try SHGO as alternative global optimizer with more iterations and better parameters
                try:
                    result_shgo = shgo(global_objective, 
                                     bounds=[(0, 1) for _ in range(32)], 
                                     n=250, iters=25, seed=42, disp=False, sampling_method='sobol')
                    if result_shgo.success:
                        best_points = result_shgo.x.reshape(-1, 2)
                        best_points = np.clip(best_points, 0, 1)
                        
                        # Final evaluation
                        distances = pdist(best_points)
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            best_ratio = min_dist / max_dist
                except:
                    pass
                    
        except Exception:
            pass
    
    # Post-processing: Apply enhanced local refinement
    if best_points is not None:
        try:
            # Enhanced refinement with more sophisticated approach
            refined_points = best_points.copy()
            
            # Run a few rounds of gradient-based optimization for final polishing
            for round_num in range(3):
                try:
                    bounds = [(0, 1) for _ in range(32)]
                    result = minimize(
                        objective, 
                        refined_points.flatten(), 
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 400, 'ftol': 1e-13}
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, 2)
                        refined_points = np.clip(refined_points, 0, 1)
                except:
                    break
            
            # Re-evaluate after refinement
            distances = pdist(refined_points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0 and (min_dist / max_dist) > best_ratio:
                best_points = refined_points
                
        except Exception:
            pass
    
    # Add a specialized strategy: enhanced hybrid approach with superior structure
    if best_points is None:
        try:
            # Create a more refined hybrid approach with better edge distribution and spacing
            hybrid_points = []
            
            # Place 4 corner points with strategic positioning
            corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
            hybrid_points.extend(corners)
            
            # Place 8 points along edges with more precise spacing
            for i in range(8):
                if i < 4:
                    # Along bottom edge - more evenly spaced
                    x = 0.15 + 0.7 * (i / 3)
                    y = 0.15
                else:
                    # Along right edge - more evenly spaced
                    x = 0.85
                    y = 0.15 + 0.7 * ((i-4) / 3)
                hybrid_points.append([x, y])
            
            # Place remaining 4 points in center with even tighter control
            for i in range(4):
                x = 0.5 + np.random.normal(0, 0.015)  # Even tighter center control
                y = 0.5 + np.random.normal(0, 0.015)
                hybrid_points.append([x, y])
            
            hybrid_points = np.array(hybrid_points[:16])
            hybrid_points = np.clip(hybrid_points, 0, 1)
            
            # Evaluate hybrid configuration
            distances = pdist(hybrid_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 1e-12:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = hybrid_points.copy()
        except:
            pass
    
    # If still no success, try one final aggressive refinement with enhanced parameters
    if best_points is None:
        try:
            # Try a more aggressive optimization on the best configuration so far
            np.random.seed(999)
            # Create a more heavily perturbed version of the best configuration
            if best_points is not None:
                perturbed_config = best_points + np.random.normal(0, 0.02, (16, 2))
            else:
                # Fall back to hex_points if needed
                perturbed_config = hex_points + np.random.normal(0, 0.03, (16, 2))
            perturbed_config = np.clip(perturbed_config, 0, 1)
            
            # Run more aggressive optimization on this
            bounds = [(0, 1) for _ in range(32)]
            result = minimize(objective, perturbed_config.flatten(), method='trust-constr',
                            bounds=bounds, options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                
                distances = pdist(final_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 1e-12:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            return final_points
        except:
            pass
    
    # If still no success, return the best hexagonal pattern
    if best_points is None:
        return hex_points
    
    return best_points


# EVOLVE-BLOCK-END
