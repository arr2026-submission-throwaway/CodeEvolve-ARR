# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Try multiple rectangle aspect ratios to find optimal configuration
    # Focus on ratios that have shown success in previous experiments
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    best_radius_sum = 0
    best_circles = None
    
    # Try multiple optimization strategies for each aspect ratio
    # Use more aggressive and diverse optimization parameters with increased iterations
    optimization_strategies = [
        {'maxiter': 2000, 'popsize': 300, 'mutation': (0.9995, 1.0), 'recombination': 0.99995},
        {'maxiter': 1800, 'popsize': 280, 'mutation': (0.999, 1.0), 'recombination': 0.9999},
        {'maxiter': 1600, 'popsize': 250, 'mutation': (0.998, 1.0), 'recombination': 0.9998},
        {'maxiter': 1400, 'popsize': 220, 'mutation': (0.995, 1.0), 'recombination': 0.9995},
        {'maxiter': 1700, 'popsize': 300, 'mutation': (0.9995, 1.0), 'recombination': 0.99995}
    ]
    
    for width_ratio in aspect_ratios:
        # Normalize to perimeter = 4
        total_ratio = width_ratio + 1.0  # height = 1.0 for simplicity in calculation
        rect_width = 2 * width_ratio / total_ratio
        rect_height = 2 * 1.0 / total_ratio
        
        # Multiple random restarts for each aspect ratio
        for strategy in optimization_strategies:
            for restart in range(2):  # 2 restarts per strategy
                np.random.seed(42 + restart * 100 + int(width_ratio * 10))
                
                # Define bounds for optimization
                bounds = []
                for i in range(21):
                    bounds.extend([
                        (0.001, rect_width - 0.001),      # x coordinate
                        (0.001, rect_height - 0.001),     # y coordinate  
                        (0.001, min(rect_width, rect_height)/2 - 0.001)  # radius
                    ])
                
                def objective(params):
                    """Minimize negative sum of radii with penalty for constraints"""
                    circles = params.reshape(-1, 3)
                    radii = circles[:, 2]
                    # Add penalty for constraint violations with better scaling
                    penalty = 0
                    positions = circles[:, :2]
                    for i in range(len(circles)):
                        x, y, r = circles[i]
                        # Even stronger penalty for boundary violations with higher-order terms
                        if x - r < 0:
                            penalty += 1e30 * (0 - (x - r))**10
                        if x + r > rect_width:
                            penalty += 1e30 * ((x + r) - rect_width)**10
                        if y - r < 0:
                            penalty += 1e30 * (0 - (y - r))**10
                        if y + r > rect_height:
                            penalty += 1e30 * ((y + r) - rect_height)**10
                    # Add overlap penalty using more sophisticated approach
                    distances = cdist(positions, positions)
                    for i in range(len(circles)):
                        for j in range(i+1, len(circles)):
                            if distances[i, j] < (radii[i] + radii[j]):
                                overlap = (radii[i] + radii[j]) - distances[i, j]
                                penalty += 1e27 * overlap**9
                    return -np.sum(radii) + penalty
                
                # Use differential evolution for global optimization
                try:
                    result = differential_evolution(
                        objective,
                        bounds,
                        maxiter=strategy['maxiter'],
                        popsize=strategy['popsize'],
                        mutation=strategy['mutation'],
                        recombination=strategy['recombination'],
                        seed=42 + restart * 100 + int(width_ratio * 10),
                        disp=False,
                        atol=1e-12,
                        rtol=1e-12
                    )
                    
                    if result.success:
                        # Extract final solution
                        circles = result.x.reshape(-1, 3)
                        
                        # Apply final validation and refinement
                        validated_circles = validate_and_refine(circles, rect_width, rect_height)
                        
                        # Calculate radius sum
                        radius_sum = np.sum(validated_circles[:, 2])
                        
                        if radius_sum > best_radius_sum:
                            best_radius_sum = radius_sum
                            best_circles = validated_circles.copy()
                            
                except Exception as e:
                    continue  # Skip this configuration if optimization fails
    
    # If no good solution found, fall back to a strong heuristic approach
    if best_circles is None:
        best_circles = improved_hexagonal_pack_approach()
    
    return best_circles

def validate_and_refine(circles, width, height):
    """Enhanced validation and refinement approach with more aggressive optimization"""
    # Ensure all circles are within bounds
    validated = circles.copy()
    
    # Apply boundary constraints first
    for i in range(len(validated)):
        x, y, r = validated[i]
        validated[i] = [
            np.clip(x, r, width - r),
            np.clip(y, r, height - r),
            r
        ]
    
    # More sophisticated overlap resolution using iterative improvement with better strategy
    improved = True
    iterations = 0
    max_iterations = 300  # Even more iterations for better convergence
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Check all pairs for overlaps
        distances = cdist(validated[:, :2], validated[:, :2])
        
        # Process overlaps in order of severity (deepest overlap first) for better convergence
        overlap_pairs = []
        for i in range(len(validated)):
            for j in range(i+1, len(validated)):
                if distances[i, j] < (validated[i, 2] + validated[j, 2]):
                    overlap = (validated[i, 2] + validated[j, 2]) - distances[i, j]
                    overlap_pairs.append((i, j, overlap))
        
        # Sort by overlap severity (deepest overlap first)
        overlap_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, overlap_severity in overlap_pairs:
            # Resolve overlap by moving circles apart
            dx = validated[j, 0] - validated[i, 0]
            dy = validated[j, 1] - validated[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0.001:  # Avoid division by zero
                overlap = (validated[i, 2] + validated[j, 2]) - dist
                # Move circles apart with more sophisticated strategy
                move_amount = overlap * 0.9  # Even more aggressive movement
                
                dx /= dist
                dy /= dist
                
                # Apply movement with better weighting - prioritize moving smaller circles
                total_radius = validated[i, 2] + validated[j, 2]
                if total_radius > 0:
                    weight_i = validated[j, 2] / total_radius
                    weight_j = validated[i, 2] / total_radius
                else:
                    weight_i = weight_j = 0.5
                
                # Apply movement with bias towards smaller circles - even more aggressive with better scaling
                validated[i, 0] -= dx * move_amount * weight_i * 2.5
                validated[i, 1] -= dy * move_amount * weight_i * 2.5
                validated[j, 0] += dx * move_amount * weight_j * 2.5
                validated[j, 1] += dy * move_amount * weight_j * 2.5
                
                improved = True
        
        # Reapply boundary constraints after movement
        for i in range(len(validated)):
            x, y, r = validated[i]
            validated[i] = [
                np.clip(x, r, width - r),
                np.clip(y, r, height - r),
                r
            ]
    
    # Local optimization: try to increase radii with even more aggressive approach
    for _ in range(4000):  # More iterations for better local optimization
        improved_local = False
        # Process circles in random order for better exploration
        indices = list(range(len(validated)))
        np.random.shuffle(indices)
        for i in indices:
            # Try to increase radius while maintaining constraints
            orig_x, orig_y, orig_r = validated[i]
            max_possible_r = min(
                orig_x, width - orig_x, 
                orig_y, height - orig_y
            )
            
            # Try to increase radius more aggressively with adaptive factor
            adaptive_factor = 1.9 if orig_r < 0.1 else 1.8
            new_r = min(orig_r * adaptive_factor, max_possible_r * 0.999998)
            
            if new_r > orig_r:
                # Test if this change is valid - check only nearby circles for efficiency
                valid = True
                # Check only nearby circles for efficiency but be thorough
                distances = np.sqrt(((validated[:, :2] - validated[i, :2])**2).sum(axis=1))
                nearby_indices = np.where(distances < (orig_r + max_possible_r * 5))[0]
                
                for j in nearby_indices:
                    if i != j:
                        dx = validated[i, 0] - validated[j, 0]
                        dy = validated[i, 1] - validated[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (new_r + validated[j, 2])**2
                        if dist_sq < min_dist_sq:
                            valid = False
                            break
                
                if valid:
                    validated[i, 2] = new_r
                    improved_local = True
        
        # Stop early if no improvement made
        if not improved_local:
            break
    
    return validated

def improved_hexagonal_pack_approach():
    """Improved fallback approach using better hexagonal packing strategy"""
    # Try a broader range of aspect ratios with more strategic sampling
    # Focus more on ratios around the most promising regions (1.0-2.5) plus some extremes
    # Add more fine-grained sampling around the most successful ratios
    candidates = [
        (0.6, 3.4), (0.65, 3.35), (0.7, 3.3), (0.75, 3.25), (0.8, 3.2),
        (0.85, 3.15), (0.9, 3.1), (0.95, 3.05), (1.0, 3.0), (1.05, 2.95),
        (1.1, 2.9), (1.15, 2.85), (1.2, 2.8), (1.25, 2.75), (1.3, 2.7),
        (1.35, 2.65), (1.4, 2.6), (1.45, 2.55), (1.5, 2.5), (1.55, 2.45),
        (1.6, 2.4), (1.65, 2.35), (1.7, 2.3), (1.75, 2.25), (1.8, 2.2),
        (1.85, 2.15), (1.9, 2.1), (1.95, 2.05), (2.0, 2.0), (2.05, 1.95),
        (2.1, 1.9), (2.15, 1.85), (2.2, 1.8), (2.25, 1.75), (2.3, 1.7),
        (2.35, 1.65), (2.4, 1.6), (2.45, 1.55), (2.5, 1.5), (2.55, 1.45),
        (2.6, 1.4), (2.65, 1.35), (2.7, 1.3), (2.75, 1.25), (2.8, 1.2),
        (2.85, 1.15), (2.9, 1.1), (2.95, 1.05), (3.0, 1.0), (3.2, 0.8),
        (3.4, 0.6), (3.6, 0.4), (3.8, 0.2), (4.0, 0.0), (4.5, 0.0),
        (5.0, 0.0), (6.0, 0.0), (7.0, 0.0), (8.0, 0.0), (10.0, 0.0),
        # Add more extreme ratios that might provide better solutions
        (0.5, 3.5), (0.4, 3.6), (0.3, 3.7), (0.2, 3.8), (0.1, 3.9)
    ]
    
    best_circles = None
    best_radius_sum = 0
    
    for width_ratio, height_ratio in candidates:
        # Normalize to perimeter = 4
        total_ratio = width_ratio + height_ratio
        if total_ratio == 0:
            continue
        width = 2 * width_ratio / total_ratio
        height = 2 * height_ratio / total_ratio
        
        # Hexagonal packing with better parameters - more precise spacing
        circles = []
        
        # Use more precise spacing for better packing
        spacing_x = width / 5.2  # Slightly less spacing to allow for better optimization
        spacing_y = height / 5.2
        
        # Better hexagonal grid with proper offsets
        rows = 5
        cols = 5
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= 21:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Check bounds with tighter margins
                if x > 0.005 and x < width - 0.005 and y > 0.005 and y < height - 0.005:
                    # Radius based on spacing - slightly larger for better packing
                    r = min(spacing_x, spacing_y) * 0.52
                    
                    # Ensure it fits in bounds
                    if x - r > 0 and x + r < width and y - r > 0 and y + r < height:
                        circles.append([x, y, r])
            
            if len(circles) >= 21:
                break
        
        # Fill remaining positions with strategic placement
        while len(circles) < 21:
            # Use a more strategic approach for remaining circles
            # Place some strategically for maximum growth potential
            if len(circles) < 8:
                # Place in corners for larger potential radii
                corner_positions = [
                    (0.1, 0.1), (0.1, height - 0.1), 
                    (width - 0.1, 0.1), (width - 0.1, height - 0.1),
                    (width/2, 0.1), (width/2, height - 0.1),
                    (0.1, height/2), (width - 0.1, height/2)
                ]
                pos_idx = len(circles) % len(corner_positions)
                x, y = corner_positions[pos_idx]
                r = min(0.38, width/6, height/6)  # Even larger initial radii
            elif len(circles) < 16:
                # Place along edges for good growth
                edge_positions = [
                    (0.1, height/2), (width - 0.1, height/2),
                    (width/2, 0.1), (width/2, height - 0.1),
                    (width/4, 0.1), (3*width/4, 0.1),
                    (width/4, height - 0.1), (3*width/4, height - 0.1)
                ]
                pos_idx = len(circles) % len(edge_positions)
                x, y = edge_positions[pos_idx]
                r = min(0.33, width/7, height/7)  # Even larger initial radii
            else:
                # Random placement elsewhere with smaller initial radii
                x = np.random.uniform(0.005, width - 0.005)
                y = np.random.uniform(0.005, height - 0.005)
                r = min(0.30, width/8, height/8)  # Slightly larger
            
            circles.append([x, y, r])
        
        # Convert to numpy array and validate
        result = np.array(circles[:21])
        validated = validate_and_refine(result, width, height)
        radius_sum = np.sum(validated[:, 2])
        
        if radius_sum > best_radius_sum:
            best_radius_sum = radius_sum
            best_circles = validated
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
