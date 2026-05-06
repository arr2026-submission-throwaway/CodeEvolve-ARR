# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape x back to points
        points = x.reshape(-1, 3)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Return negative of min/max ratio (we want to maximize this, so minimize negative)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
            
        return -min_dist / max_dist
    
    # Try multiple initialization strategies and select best result
    best_points = None
    best_ratio = -np.inf
    
    # Strategy 1: Dodecahedral-based initialization for better symmetry
    np.random.seed(42)
    # Use vertices of regular dodecahedron (12 vertices) plus 2 additional strategic points
    dodec_vertices = np.array([
        [0, 0, 1], [0, 0, -1], [0.5, 0.5, 0], [0.5, -0.5, 0],
        [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, 0, 0.5], [0.5, 0, -0.5],
        [-0.5, 0, 0.5], [-0.5, 0, -0.5], [0, 0.5, 0.5], [0, 0.5, -0.5],
        [0, -0.5, 0.5], [0, -0.5, -0.5]
    ])
    
    # Take first 12 vertices and add 2 more strategic points
    points_dodec = dodec_vertices[:12].copy()
    # Add two points at strategic positions for better spread
    points_dodec = np.vstack([points_dodec, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    # Normalize to unit sphere
    norms = np.linalg.norm(points_dodec, axis=1, keepdims=True)
    points_dodec = points_dodec / np.max(norms) * 0.8
    # Add perturbation to break perfect symmetry
    points_dodec += np.random.normal(0, 0.05, points_dodec.shape)
    
    # Strategy 2: Enhanced Fibonacci-based distribution on sphere with improved spacing
    points_fib = []
    n = 14
    golden_ratio = (1 + np.sqrt(5)) / 2

    # Generate Fibonacci-like distribution with better uniformity using a more refined approach
    for i in range(n):
        # Use a more sophisticated distribution that avoids clustering
        # Apply a slight adjustment to reduce symmetry and improve spread
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(np.maximum(0, 1 - y * y))  # Ensure non-negative radius

        # Apply golden angle with a small perturbation for better distribution
        phi = ((i % (n - 1)) * golden_ratio + 0.1 * np.sin(i * 0.7)) % (2 * np.pi)

        x = radius * np.cos(phi)
        z = radius * np.sin(phi)

        points_fib.append([x, y, z])

    points_fib = np.array(points_fib)
    # Normalize to unit sphere and scale appropriately
    norms = np.linalg.norm(points_fib, axis=1, keepdims=True)
    points_fib = points_fib / np.maximum(norms, 1e-10) * 0.95
    # Add moderate perturbation to break symmetries with slightly larger variance
    points_fib += np.random.normal(0, 0.04, points_fib.shape)
    
    # Strategy 3: Enhanced random initialization with better spread
    points_random = np.random.uniform(-0.9, 0.9, (14, 3))
    # Add more substantial perturbation to avoid degenerate cases
    points_random += np.random.normal(0, 0.02, points_random.shape)
    
    # Strategy 4: Cube vertex-based initialization
    cube_vertices = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    ])
    points_cube = np.zeros((14, 3))
    points_cube[:8] = cube_vertices
    points_cube[8:] = np.random.rand(6, 3) * 0.6 + 0.2  # Remaining points
    
    # Strategy 5: Enhanced differential evolution for global search with aggressive parameters
    bounds = [(-0.95, 0.95) for _ in range(14 * 3)]
    
    # Run differential evolution with highly aggressive parameters for better global search
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=300,  # More iterations for better exploration
            popsize=45,   # Larger population for better diversity
            mutation=(0.98, 1.0),  # Even higher mutation rate for more exploration
            recombination=0.99,   # Even higher recombination rate
            disp=False,
            atol=1e-15,  # Very tight tolerance for better precision
            rtol=1e-15
        )
        points_de = de_result.x.reshape(-1, 3)
        # Add perturbation to escape local minima with even smaller variance
        points_de += np.random.normal(0, 0.005, points_de.shape)
    except:
        points_de = points_fib  # fallback
    
    # Evaluate all strategies and pick the best
    strategies = [
        ("dodecahedral", points_dodec),
        ("fibonacci", points_fib),
        ("random", points_random), 
        ("cube", points_cube),
        ("differential_evolution", points_de)
    ]
    
    for name, points_init in strategies:
        try:
            # Flatten for optimization
            x0 = points_init.flatten()
            
            # Try multiple local optimization methods for robustness
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            best_local_ratio = -np.inf
            best_local_points = None
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        options={'maxiter': 500, 'ftol': 1e-12},
                        tol=1e-12
                    )
                    
                    if result.success:
                        points_opt = result.x.reshape(-1, 3)
                        distances = pdist(points_opt)
                        if len(distances) > 0:
                            min_dist = np.min(distances)
                            max_dist = np.max(distances)
                            if max_dist > 0:
                                ratio = min_dist / max_dist
                                if ratio > best_local_ratio:
                                    best_local_ratio = ratio
                                    best_local_points = points_opt.copy()
                except:
                    continue
            
            if best_local_points is not None and best_local_ratio > best_ratio:
                best_ratio = best_local_ratio
                best_points = best_local_points.copy()
                
        except:
            continue
    
    # If no good solution found, return fibonacci strategy
    if best_points is None:
        return points_fib
    
    # Final refinement with more aggressive optimization to maximize the ratio
    if best_ratio > 0.1:  # Only refine if we have a reasonable starting point
        try:
            x0 = best_points.flatten()
            # Use more aggressive optimization with highest precision
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-17},
                tol=1e-17
            )
            if result.success:
                final_distances = pdist(result.x.reshape(-1, 3))
                if len(final_distances) > 0:
                    final_min = np.min(final_distances)
                    final_max = np.max(final_distances)
                    if final_max > 0:
                        final_ratio = final_min / final_max
                        if final_ratio > best_ratio:
                            best_points = result.x.reshape(-1, 3)
                            
            # Also try trust-constr as backup optimization method with ultra-precise settings
            result_tc = minimize(
                objective,
                x0,
                method='trust-constr',
                options={'maxiter': 1000, 'xtol': 1e-17, 'gtol': 1e-17}
            )
            if result_tc.success:
                final_distances_tc = pdist(result_tc.x.reshape(-1, 3))
                if len(final_distances_tc) > 0:
                    final_min_tc = np.min(final_distances_tc)
                    final_max_tc = np.max(final_distances_tc)
                    if final_max_tc > 0:
                        final_ratio_tc = final_min_tc / final_max_tc
                        if final_ratio_tc > best_ratio:
                            best_points = result_tc.x.reshape(-1, 3)
                            
            # Also try Nelder-Mead as final backup optimization method
            result_nm = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-16, 'xatol': 1e-16}
            )
            if result_nm.success:
                final_distances_nm = pdist(result_nm.x.reshape(-1, 3))
                if len(final_distances_nm) > 0:
                    final_min_nm = np.min(final_distances_nm)
                    final_max_nm = np.max(final_distances_nm)
                    if final_max_nm > 0:
                        final_ratio_nm = final_min_nm / final_max_nm
                        if final_ratio_nm > best_ratio:
                            best_points = result_nm.x.reshape(-1, 3)
        except:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
