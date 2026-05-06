# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Return negative of min/max ratio (we want to maximize it)
        if len(distances) == 0:
            return 0
        
        # Filter out zero distances (same points) and handle edge cases
        distances_nonzero = distances[distances > 1e-18]
        if len(distances_nonzero) == 0:
            return 0
            
        min_dist = np.min(distances_nonzero)
        max_dist = np.max(distances)
        
        # Avoid division by zero with a small epsilon
        epsilon = 1e-18
        if max_dist > epsilon:
            return -min_dist / (max_dist + epsilon)
        else:
            return 0
    
    # Multiple initial configurations for better exploration
    np.random.seed(42)
    
    # Configuration 1: Enhanced Fibonacci spiral on sphere with better uniformity and clustering control
    n = 14
    points1 = np.zeros((n, 3))
    golden_angle = 2.399963229728653  # More precise golden angle
    
    for i in range(n):
        # Better distribution using arccos for more even spacing
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        # Add more sophisticated systematic variation to reduce clustering
        systematic_variation = 0.0015 * np.sin(i * 1.3) + 0.001 * np.cos(i * 0.7)
        theta = i * golden_angle + systematic_variation + np.random.normal(0, 0.001)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            points1[i] = [x/norm, y/norm, z/norm]
    
    # Configuration 2: Enhanced Fibonacci with adaptive perturbations and better symmetry breaking
    points2 = points1.copy()
    # Apply more sophisticated perturbations with better control and symmetry breaking
    for i in range(n):
        # Use more structured perturbation based on point position with better variation
        if abs(points2[i, 1]) < 0.3:  # Near poles - smaller perturbations
            magnitude = 0.0025 + 0.001 * np.sin(i * 0.8)
        elif abs(points2[i, 1]) < 0.7:  # Mid-latitudes - medium perturbations
            magnitude = 0.005 + 0.002 * np.cos(i * 0.5)
        else:  # Near equator - larger perturbations
            magnitude = 0.008 + 0.003 * np.sin(i * 1.2)
        
        # Add systematic variation to break symmetry
        systematic_variation = 0.001 * np.sin(i * 1.5) + 0.0005 * np.cos(i * 0.9)
        perturbation = np.random.normal(0, magnitude, 3)
        points2[i] += perturbation + systematic_variation
        # Normalize to unit sphere
        norm = np.linalg.norm(points2[i])
        if norm > 0:
            points2[i] = points2[i] / norm
    
    # Configuration 3: Enhanced icosahedral with better vertex positioning and systematic perturbations
    # Generate vertices of a regular icosahedron with higher precision
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    
    points3 = np.array(ico_vertices)
    # Add 2 more points for better coverage with strategic placement and better perturbations
    polar_points = [[0, 0, 0.995], [0, 0, -0.995]]
    points3 = np.vstack([points3, polar_points])
    
    # Apply systematic perturbations to improve uniformity
    for i in range(14):
        if i < 12:  # Icosahedral vertices
            perturbation_magnitude = 0.0025 + 0.001 * np.sin(i * 0.6)
        else:  # Polar points
            perturbation_magnitude = 0.0015 + 0.0005 * np.cos(i * 0.4)
        
        # Add systematic variation to break symmetry
        systematic_variation = 0.001 * np.sin(i * 1.2) + 0.0005 * np.cos(i * 0.8)
        noise = np.random.normal(0, perturbation_magnitude, 3)
        points3[i] += noise + systematic_variation
        # Normalize to unit sphere
        norm = np.linalg.norm(points3[i])
        if norm > 0:
            points3[i] = points3[i] / norm
    
    # Configuration 4: Enhanced icosahedral with better point distribution and strategic additions
    points4 = points3.copy()
    # Add more strategically positioned additional points using spherical coordinates
    additional_points = []
    for i in range(6):
        # Place points in regions that tend to be underrepresented
        phi = np.pi * (i / 5)  # Spread evenly in phi direction
        theta = i * np.pi / 3  # Spread in theta direction
        x = np.sin(phi) * np.cos(theta) * 0.85
        y = np.sin(phi) * np.sin(theta) * 0.85
        z = np.cos(phi) * 0.85
        additional_points.append([x, y, z])
    
    points4 = np.vstack([points4, additional_points])
    
    # Normalize all points to unit sphere
    for i in range(len(points4)):
        norm = np.linalg.norm(points4[i])
        if norm > 0:
            points4[i] = points4[i] / norm
    
    # Configuration 5: Advanced spherical code with enhanced equidistribution
    points5 = np.zeros((n, 3))
    golden_angle = 2.399963229728653  # More precise golden angle
    for i in range(n):
        phi = np.arccos(1 - 2 * (i / (n - 1)))  # Better distribution
        # Add more sophisticated systematic variation to reduce clustering
        systematic_variation = 0.003 * np.sin(i * 1.5) + 0.002 * np.cos(i * 0.8) + \
                              0.001 * np.sin(i * 2.3) * np.cos(i * 1.1)
        noise_magnitude = 0.001 + 0.0005 * np.sin(i * 0.9)
        theta = i * golden_angle + systematic_variation + np.random.normal(0, noise_magnitude)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            points5[i] = [x/norm, y/norm, z/norm]
    
    # Configuration 6: Stratified random with better spatial distribution and systematic control
    points6 = np.random.rand(n, 3) * 2 - 1  # [-1,1] range
    # Apply stratified sampling with better normalization and systematic variation
    for i in range(n):
        # Add structured variation to reduce clustering
        systematic_variation = 0.0015 * np.sin(i * 1.3) + 0.001 * np.cos(i * 0.7)
        noise = np.random.normal(0, 0.003, 3)
        points6[i] += noise + systematic_variation
        # Normalize to unit sphere
        norm = np.linalg.norm(points6[i])
        if norm > 0:
            points6[i] = points6[i] / norm
    
    # Configuration 7: Optimized icosahedral with enhanced perturbations and better structure
    # Generate vertices of a regular icosahedron with precise positioning
    phi = (1 + np.sqrt(5)) / 2
    vertices = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    
    points7 = np.array(vertices)
    # Normalize to unit sphere
    for i in range(len(points7)):
        norm = np.linalg.norm(points7[i])
        if norm > 0:
            points7[i] = points7[i] / norm
    
    # Add two more points with strategic positioning and better perturbations
    points7 = np.vstack([points7, [[0, 0, 0.985], [0, 0, -0.985]]])
    
    # Apply systematic perturbations with better control
    for i in range(14):
        if i < 12:  # Icosahedral vertices
            perturbation_magnitude = 0.0025 + 0.001 * np.sin(i * 0.6)
        else:  # Polar points
            perturbation_magnitude = 0.0015 + 0.0005 * np.cos(i * 0.4)
        
        # Add systematic variation with more complex pattern
        systematic_variation = 0.001 * np.sin(i * 1.3) + 0.0005 * np.cos(i * 0.8)
        noise = np.random.normal(0, perturbation_magnitude, 3)
        points7[i] += noise + systematic_variation
        norm = np.linalg.norm(points7[i])
        if norm > 0:
            points7[i] = points7[i] / norm
    
    # Test multiple initial configurations - more comprehensive approach
    best_points = None
    best_ratio = -np.inf
    
    configs = [points1, points2, points3, points4, points5, points6, points7]  # Added more diverse configurations
    
    for points in configs:
        try:
            # Try multiple local optimization methods for robustness
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-20}),
                ('TNC', {'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-20}),
                ('SLSQP', {'maxiter': 2000, 'ftol': 1e-20}),
                ('trust-constr', {'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-20})
            ]
            
            for method, options in methods_and_options:
                try:
                    result = minimize(
                        objective,
                        points.flatten(),
                        method=method,
                        bounds=[(-1, 1) for _ in range(42)],
                        options=options
                    )
                    
                    if result.success:
                        test_points = result.x.reshape(-1, 3)
                        # Calculate ratio for this configuration
                        distances = pdist(test_points)
                        min_dist = np.min(distances[distances > 1e-16])  # Exclude zero distances
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = test_points.copy()
                        break  # Success, move to next configuration
                except:
                    continue
        except:
            continue
    
    # If no good configuration found, start with a simple random configuration
    if best_points is None:
        best_points = np.random.rand(n, 3)
    
    # Global optimization with differential evolution for final refinement
    bounds = [(-1, 1) for _ in range(42)]  # Wider bounds for better exploration
    
    try:
        # Use differential evolution for global search with more aggressive settings
        de_configs = [
            {
                'maxiter': 600,
                'popsize': 80,
                'mutation': (0.98, 1),
                'recombination': 0.99,
                'strategy': 'rand1bin'
            },
            {
                'maxiter': 500,
                'popsize': 70,
                'mutation': (0.95, 1),
                'recombination': 0.98,
                'strategy': 'best1bin'
            },
            {
                'maxiter': 400,
                'popsize': 60,
                'mutation': (0.9, 1),
                'recombination': 0.95,
                'strategy': 'best2bin'
            },
            {
                'maxiter': 300,
                'popsize': 50,
                'mutation': (0.85, 1),
                'recombination': 0.92,
                'strategy': 'rand2bin'
            }
        ]
        
        for config in de_configs:
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    **config,
                    seed=42,
                    disp=False,
                    atol=1e-18,
                    rtol=1e-18
                )
                
                if result.success:
                    global_points = result.x.reshape(-1, 3)
                    # Calculate ratio for global solution
                    distances = pdist(global_points)
                    min_dist = np.min(distances[distances > 1e-16])  # Exclude zero distances
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = global_points
            except:
                continue
    except:
        pass
    
    # If we still don't have a good solution, try one more aggressive optimization
    if best_points is None:
        # Use a more focused approach with better initial conditions
        try:
            # Start with the most promising configuration from our tests
            if best_points is not None:
                start_points = best_points.copy()
            else:
                start_points = points1.copy()  # Default to Fibonacci
            
            # Try multiple local optimization methods with stricter tolerances
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 3000, 'ftol': 1e-22, 'gtol': 1e-22}),
                ('TNC', {'maxiter': 3000, 'ftol': 1e-22, 'gtol': 1e-22}),
                ('SLSQP', {'maxiter': 3000, 'ftol': 1e-22}),
                ('trust-constr', {'maxiter': 3000, 'ftol': 1e-22, 'gtol': 1e-22}),
                ('COBYLA', {'maxiter': 3000, 'tol': 1e-21})
            ]
            
            for method, options in methods_and_options:
                try:
                    result = minimize(
                        objective,
                        start_points.flatten(),
                        method=method,
                        bounds=[(-1, 1) for _ in range(42)],
                        options=options
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        distances = pdist(final_points)
                        min_dist = np.min(distances[distances > 1e-16])  # Exclude zero distances
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points
                        break  # Success, move to next method
                except:
                    continue
        except:
            pass
    
    # Additional refinement with hybrid approach - more thorough
    if best_points is not None:
        try:
            # Try several rounds of more intensive local optimization with enhanced strategy
            refined_points = best_points.copy()
            for i in range(15):  # Increased refinement iterations for better convergence
                result = minimize(
                    objective,
                    refined_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-1, 1) for _ in range(42)],
                    options={'maxiter': 4000, 'ftol': 1e-28, 'gtol': 1e-28}  # Even tighter tolerances
                )
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    distances = pdist(refined_points)
                    min_dist = np.min(distances[distances > 1e-22])  # Even tighter threshold
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
                else:
                    break
        except:
            pass
    
    return best_points if best_points is not None else np.random.rand(n, 3)


# EVOLVE-BLOCK-END
