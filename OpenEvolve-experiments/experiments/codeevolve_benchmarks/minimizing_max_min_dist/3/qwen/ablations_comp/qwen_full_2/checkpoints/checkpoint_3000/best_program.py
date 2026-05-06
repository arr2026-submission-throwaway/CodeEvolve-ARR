# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def objective(x):
        # Reshape to 14x3 points
        points = x.reshape(14, 3)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to minimize (since we want to maximize ratio)
        if d_max == 0 or d_max < 1e-12:
            return -1e10
        return -d_min / d_max
    
    # Create better initial configurations with improved geometric distributions
    n = 14
    
    # Configuration 1: Enhanced Fibonacci sphere with better uniformity and perturbation
    points1 = np.zeros((n, 3))
    # Use the most precise golden angle for optimal distribution
    golden_angle = 2.399963229728653  # ~4π/(1+√5)
    for i in range(n):
        # Improved distribution using Fibonacci method with better parameterization
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = golden_angle * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points1[i] = [x, y, z]
    points1 = (points1 + 1) / 2  # Scale to [0,1]^3
    # Add stronger perturbation to break symmetries and improve optimization
    points1 += np.random.normal(0, 0.02, (n, 3))  # Increased noise for better exploration
    points1 = np.clip(points1, 0, 1)
    
    # Configuration 2: Enhanced Icosahedral arrangement with better point distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    # Normalize to unit sphere and scale
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    points2 = (vertices + 1) / 2  # Scale to [0,1]^3
    # Add stronger perturbation to break symmetries
    points2 += np.random.normal(0, 0.03, (12, 3))  # Increased noise for better exploration
    points2 = np.clip(points2, 0, 1)
    if n > 12:
        # Add more points with better geometric distribution
        # Use spiral-based distribution for better uniformity
        additional = np.zeros((n-12, 3))
        for i in range(n-12):
            # Distribute additional points along spiral paths
            t = i / (n - 12 - 1) if n - 12 > 1 else 0
            r = 0.3 + 0.2 * t  # Radial variation
            angle = 2 * np.pi * i * 1.618  # Golden angle for spiral
            additional[i] = [
                0.5 + r * np.cos(angle),
                0.5 + r * np.sin(angle),
                0.5  # Mid-level z-coordinate
            ]
        points2 = np.vstack([points2, additional])
    points2 = points2[:n]
    
    # Configuration 3: Enhanced octahedral arrangement with better spacing and improved distribution
    points3 = np.array([
        [0.5, 0.5, 1], [0.5, 0.5, 0],  # top and bottom faces
        [0.5, 1, 0.5], [0.5, 0, 0.5],  # front and back faces
        [1, 0.5, 0.5], [0, 0.5, 0.5],  # left and right faces
        [0.5, 0.5, 0.5]  # center point
    ])
    # Add stronger perturbation to avoid degenerate cases and improve exploration
    points3 += np.random.normal(0, 0.03, (7, 3))  # Increased noise for better exploration
    points3 = np.clip(points3, 0, 1)
    # Add more points with better geometric distribution
    if n > 7:
        # Add points arranged in a pattern that avoids clustering with better spacing
        additional_points = []
        # Add points along face diagonals and edges with better spacing
        additional_points.extend([[0.2, 0.2, 0.5], [0.2, 0.8, 0.5], [0.8, 0.2, 0.5], [0.8, 0.8, 0.5]])
        additional_points.extend([[0.5, 0.2, 0.2], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2], [0.5, 0.8, 0.8]])
        additional_points.extend([[0.2, 0.5, 0.2], [0.2, 0.5, 0.8], [0.8, 0.5, 0.2], [0.8, 0.5, 0.8]])
        
        # Fill remaining spots with random but constrained points
        remaining_count = n - len(points3) - len(additional_points)
        if remaining_count > 0:
            extra_points = np.random.rand(remaining_count, 3) * 0.6 + 0.2  # Centered in [0.2, 0.8]^3
            additional_points.extend(extra_points.tolist())
        
        points3 = np.vstack([points3, additional_points[:n-7]])
    points3 = points3[:n]
    
    # Configuration 4: Random with better spread (using low-discrepancy sequence when available)
    np.random.seed(42)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, seed=42)
        points4 = sampler.random(n)
    except:
        points4 = np.random.rand(n, 3)
    
    # Configuration 5: Enhanced perturbed Fibonacci with adaptive noise scaling and better tuning
    points5 = points1.copy()
    np.random.seed(123)
    # Use adaptive noise based on point density for better exploration
    noise_scale = 0.01  # Slightly smaller noise for more controlled perturbation
    noise = np.random.normal(0, noise_scale, (n, 3))
    points5 += noise
    points5 = np.clip(points5, 0, 1)
    
    # Configuration 6: Enhanced spiral arrangement with better radial control and perturbation
    points6 = np.zeros((n, 3))
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        # Improved radial variation for better uniformity
        r = 0.35 * (1 - t) + 0.15 * t  # More concentrated at edges
        angle = 2 * np.pi * i * 1.618  # Golden angle for spiral
        points6[i] = [
            0.5 + r * np.cos(angle),
            0.5 + r * np.sin(angle),
            t
        ]
    points6 = np.clip(points6, 0, 1)
    # Add stronger perturbation to improve optimization landscape
    points6 += np.random.normal(0, 0.015, (n, 3))  # Increased noise for better exploration
    points6 = np.clip(points6, 0, 1)
    
    initial_configs = [points1, points2, points3, points4, points5, points6]
    
    best_points = None
    best_ratio = -np.inf
    
    # Try multiple optimization strategies with different starting points
    for i, points in enumerate(initial_configs):
        x0 = points.flatten()
        
        # First try differential evolution for global optimization with more aggressive parameters
        try:
            bounds = [(0, 1)] * (n * 3)
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=250,   # More iterations for better exploration
                popsize=80,   # Larger population for better diversity
                seed=42,
                disp=False,
                atol=1e-17,   # Even tighter tolerance for better convergence
                rtol=1e-17,
                mutation=(0.98, 1.0),  # Better mutation strategy
                recombination=0.998,  # Higher recombination for better mixing
                strategy='best1exp'  # Exponential strategy for better adaptation
            )
            
            if de_result.success:
                final_points = de_result.x.reshape(-1, 3)
                distances = pdist(final_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 1e-12:  # Check for numerical stability
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
        except:
            pass
        
        # Then try local optimization with multiple methods for better convergence
        methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
        for method in methods:
            try:
                if method == 'trust-constr':
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=[(0, 1)] * (n * 3),
                        options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-18, 'disp': False},  # More aggressive tolerances
                        tol=1e-18
                    )
                else:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=[(0, 1)] * (n * 3),
                        options={'maxiter': 1200, 'ftol': 1e-18, 'gtol': 1e-18},  # More aggressive tolerances
                        tol=1e-18
                    )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    distances = pdist(final_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 1e-12:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
            except:
                continue
    
    # If we found a good solution, return it; otherwise return the best initial config
    if best_points is not None:
        return best_points
    
    # Fallback to the best initial configuration
    fallback_ratios = []
    for config in initial_configs:
        distances = pdist(config)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 1e-12:  # Numerical stability check
                ratio = min_dist / max_dist
                fallback_ratios.append(ratio)
            else:
                fallback_ratios.append(-np.inf)
        else:
            fallback_ratios.append(-np.inf)
    
    # Try one more optimization on the best initial config to ensure we don't miss anything
    best_fallback_idx = np.argmax(fallback_ratios)
    best_initial = initial_configs[best_fallback_idx]
    
    # Try to optimize the best initial configuration further with enhanced parameters
    x0 = best_initial.flatten()
    try:
        # Try multiple optimization methods on the fallback for robustness
        for method in ['trust-constr', 'L-BFGS-B', 'SLSQP']:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=[(0, 1)] * (n * 3),
                options={'maxiter': 2000, 'ftol': 1e-19, 'gtol': 1e-19, 'disp': False},
                tol=1e-19
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
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
    
    # If still no good solution, try additional aggressive optimizations
    for method in ['trust-constr']:
        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=[(0, 1)] * (n * 3),
                options={'maxiter': 2500, 'ftol': 1e-20, 'gtol': 1e-20, 'disp': False},
                tol=1e-20
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                distances = pdist(final_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 1e-12:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            return final_points
        except:
            continue
    
    return initial_configs[best_fallback_idx]


# EVOLVE-BLOCK-END
