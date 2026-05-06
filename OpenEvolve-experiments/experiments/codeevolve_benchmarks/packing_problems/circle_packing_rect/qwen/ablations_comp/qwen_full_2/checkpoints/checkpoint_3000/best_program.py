# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    best_sum = 0
    best_circles = None
    
    # Try multiple rectangle aspect ratios to find optimal configuration
    # Focus on the most promising central region with ultra-dense sampling
    aspect_ratios = []
    # Ultra-dense sampling around the most promising central region (0.95-1.05)
    aspect_ratios.extend(np.linspace(0.95, 1.05, 51))  # Even higher density around 1.0
    # Add some extreme ratios that have shown good performance
    aspect_ratios.extend([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0])
    
    for ratio in aspect_ratios:
        # Calculate width and height such that width + height = 2
        width = 2.0 * ratio / (1 + ratio)
        height = 2.0 / (1 + ratio)
        
        # Skip invalid dimensions
        if width <= 0 or height <= 0:
            continue
            
        # Generate initial configuration using better hexagonal packing
        circles = generate_improved_hexagonal_layout(width, height, 21)
        
        # Optimize using differential evolution for global search
        optimized_circles = optimize_with_de(circles, width, height)
        
        # Further refine with local optimization
        refined_circles = optimize_locally(optimized_circles, width, height)
        
        current_sum = np.sum(refined_circles[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()
    
    # If no good solution found, return a fallback
    if best_circles is None:
        # Fallback to a simple configuration with good aspect ratio
        circles = np.zeros((21, 3))
        width, height = 1.5, 0.5
        for i in range(21):
            circles[i] = [width/2, height/2, 0.1]
        return circles
    
    return best_circles

def generate_improved_hexagonal_layout(width: float, height: float, n: int) -> np.ndarray:
    """Generate initial circle layout using improved hexagonal packing with better density"""
    circles = np.zeros((n, 3))
    
    # Estimate radius based on area with better packing density consideration
    area = width * height
    avg_area_per_circle = area / n
    # Use even smaller initial radius to allow for better optimization later
    estimated_radius = np.sqrt(avg_area_per_circle / np.pi) * 0.88  # Slightly larger for more growth potential
    
    # Better hexagonal packing parameters with optimized spacing
    spacing_x = estimated_radius * 2 * 0.995  # Even tighter spacing for denser packing
    spacing_y = estimated_radius * np.sqrt(3) * 0.995  # Adjusted for better packing
    
    # Calculate grid dimensions with more generous padding
    rows = max(1, int(height / spacing_y) + 12)  # Even more padding for better coverage
    cols = max(1, int(width / spacing_x) + 12)   # Even more padding for better coverage
    
    # Place circles in hexagonal pattern with better boundary checking
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + (i % 2) * 0.5) * spacing_x + estimated_radius
            y = i * spacing_y + estimated_radius
            
            # Ensure within bounds with margin
            if x + estimated_radius <= width * 0.998 and y + estimated_radius <= height * 0.998:
                circles[idx] = [x, y, estimated_radius]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with more strategic placement
    remaining = n - idx
    if remaining > 0:
        # Prioritize edge and corner placements more aggressively with better distribution
        for i in range(remaining):
            if idx >= n:
                break
            # 88% chance for edge/corner placement, 12% for center
            if random.random() < 0.88:
                # Place near edge or corner for better space utilization
                edge_or_corner = random.random()
                if edge_or_corner < 0.7:  # Edge placement
                    edge_type = random.randint(0, 3)
                    if edge_type == 0:  # Top edge
                        x = random.uniform(estimated_radius, width - estimated_radius)
                        y = estimated_radius * 1.02
                    elif edge_type == 1:  # Bottom edge
                        x = random.uniform(estimated_radius, width - estimated_radius)
                        y = height - estimated_radius * 1.02
                    elif edge_type == 2:  # Left edge
                        x = estimated_radius * 1.02
                        y = random.uniform(estimated_radius, height - estimated_radius)
                    else:  # Right edge
                        x = width - estimated_radius * 1.02
                        y = random.uniform(estimated_radius, height - estimated_radius)
                else:  # Corner placement
                    corner_type = random.randint(0, 3)
                    if corner_type == 0:  # Top-left
                        x = estimated_radius * 1.03
                        y = estimated_radius * 1.03
                    elif corner_type == 1:  # Top-right
                        x = width - estimated_radius * 1.03
                        y = estimated_radius * 1.03
                    elif corner_type == 2:  # Bottom-left
                        x = estimated_radius * 1.03
                        y = height - estimated_radius * 1.03
                    else:  # Bottom-right
                        x = width - estimated_radius * 1.03
                        y = height - estimated_radius * 1.03
            else:
                # Place near center with radial distribution
                center_x, center_y = width/2, height/2
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, min(width, height) * 0.2)
                x = center_x + distance * np.cos(angle)
                y = center_y + distance * np.sin(angle)
            
            # Ensure within bounds
            x = np.clip(x, estimated_radius, width - estimated_radius)
            y = np.clip(y, estimated_radius, height - estimated_radius)
            r = estimated_radius * random.uniform(0.9, 1.05)
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

def optimize_with_de(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Use differential evolution for global optimization with enhanced parameters"""
    n = len(initial_circles)
    
    # Flatten initial configuration
    initial_flat = initial_circles.flatten()
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Constraint function with better numerical handling
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints (all circles must be fully within rectangle)
        for i in range(len(circles)):
            x, y, r = circles[i]
            # x - r >= 0, y - r >= 0, width - x - r >= 0, height - y - r >= 0
            constraints.extend([
                x - r,           # x - r >= 0
                y - r,           # y - r >= 0  
                width - x - r,   # width - x - r >= 0
                height - y - r   # height - y - r >= 0
            ])
        
        # Overlap constraints (distance between centers >= sum of radii)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + 1e-12)  # Add epsilon for numerical stability
                # distance - (r1 + r2) >= 0 (so we want distance >= r1 + r2)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
    
    # Use differential evolution for global optimization with better parameters
    try:
        # Run multiple DE instances with different seeds for better exploration
        best_result = None
        best_sum = -float('inf')
        
        # Run multiple DE instances with different seeds - even more for better exploration
        seeds = [42, 123, 456, 789, 999, 1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10000, 11111, 12222, 13333, 14444, 15555, 16666, 17777, 18888, 19999, 20202, 21212, 22222, 23232, 24242, 25252, 26262, 27272, 28282, 29292, 30303, 31313, 32323, 33333, 34343, 35353, 36363, 37373, 38383, 39393, 40404, 41414, 42424, 43434, 44444, 45454, 46464, 47474, 48484, 49494, 50505, 51515, 52525, 53535, 54545, 55555, 56565, 57575, 58585, 59595, 60606, 61616, 62626, 63636, 64646, 65656, 66666, 67676, 68686, 69696, 70707, 71717, 72727, 73737, 74747, 75757, 76767, 77777, 78787, 79797, 80808, 81818, 82828, 83838, 84848, 85858, 86868, 87878, 88888, 89898, 90909, 91919, 92929, 93939, 94949, 95959, 96969, 97979, 98989, 99999]
        for seed_val in seeds:
            result = differential_evolution(
                objective,
                bounds,
                args=(),
                maxiter=300,     # Even more iterations for better convergence
                popsize=80,      # Even larger population for better exploration
                mutation=(0.95, 1.0), # Even more aggressive mutation
                recombination=0.98, # Even higher recombination rate
                seed=seed_val,
                disp=False,
                tol=1e-14        # Even tighter tolerance for better accuracy
            )
            
            if result.success:
                current_sum = -result.fun  # Convert back from minimization
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        
        if best_result is not None:
            optimized_circles = best_result.x.reshape(-1, 3)
            return validate_and_refine(optimized_circles, width, height)
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return validate_and_refine(optimized_circles, width, height)
        
    except Exception:
        pass
    
    # Fallback to local optimization if DE fails
    return optimize_locally(initial_circles, width, height)

def optimize_locally(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Use local optimization with SLSQP for fine-tuning"""
    try:
        n = len(initial_circles)
        initial_flat = initial_circles.flatten()
        
        def objective(params):
            circles = params.reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        def constraint_func(params):
            circles = params.reshape(-1, 3)
            constraints = []
            
            # Boundary constraints
            for i in range(len(circles)):
                x, y, r = circles[i]
                constraints.extend([
                    x - r,           # x - r >= 0
                    y - r,           # y - r >= 0  
                    width - x - r,   # width - x - r >= 0
                    height - y - r   # height - y - r >= 0
                ])
            
            # Overlap constraints
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    constraints.append(distance - (r1 + r2))
            
            return np.array(constraints)
        
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
        
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return validate_and_refine(optimized_circles, width, height)
            
    except Exception:
        pass
    
    return validate_and_refine(initial_circles, width, height)

def validate_and_refine(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all circles are valid and within bounds, then perform enhanced local refinement"""
    validated = circles.copy()
    
    # Apply boundary constraints with better handling
    for i in range(len(validated)):
        x, y, r = validated[i]
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        r = max(r, 0.001)
        validated[i] = [x, y, r]
    
    # Enhanced local refinement with more aggressive optimization
    refined = validated.copy()
    improved = True
    iterations = 0
    
    # Multiple passes with different refinement strategies for better convergence
    for pass_num in range(5):  # More passes for better convergence
        pass_improved = True
        pass_iterations = 0
        
        while pass_improved and pass_iterations < 350:  # More iterations per pass
            pass_improved = False
            pass_iterations += 1
            
            # Process circles in shuffled order for better convergence
            indices = list(range(len(refined)))
            random.shuffle(indices)
            
            for i in indices:
                old_radius = refined[i, 2]
                # Find maximum possible radius
                max_possible_radius = min(
                    refined[i, 0], 
                    width - refined[i, 0],
                    refined[i, 1],
                    height - refined[i, 1]
                )
                
                # Check overlaps with all other circles
                new_radius = max_possible_radius
                valid = True
                
                for j in range(len(refined)):
                    if i != j:
                        x1, y1, r1 = refined[i]
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + 1e-12)  # Numerical stability
                        
                        # Can't overlap with other circles
                        max_radius_for_i = distance - r2
                        if max_radius_for_i < 1e-9:  # Tighter tolerance
                            valid = False
                            break
                        new_radius = min(new_radius, max_radius_for_i)
                
                # Even more aggressive improvement threshold with pass-dependent values
                threshold = 1.05 if pass_num == 0 else (1.04 if pass_num == 1 else (1.03 if pass_num == 2 else (1.02 if pass_num == 3 else 1.015)))
                if valid and new_radius > old_radius * threshold:
                    refined[i, 2] = min(new_radius, max_possible_radius)
                    pass_improved = True
                    improved = True
    
    # Final optimization pass with very fine adjustments
    final_refined = refined.copy()
    for _ in range(200):  # More fine tuning iterations
        any_improved = False
        for i in range(len(final_refined)):
            old_radius = final_refined[i, 2]
            # Find maximum possible radius
            max_possible_radius = min(
                final_refined[i, 0], 
                width - final_refined[i, 0],
                final_refined[i, 1],
                height - final_refined[i, 1]
            )
            
            # Check overlaps with all other circles
            new_radius = max_possible_radius
            valid = True
            
            for j in range(len(final_refined)):
                if i != j:
                    x1, y1, r1 = final_refined[i]
                    x2, y2, r2 = final_refined[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + 1e-12)  # Numerical stability
                    
                    # Can't overlap with other circles
                    max_radius_for_i = distance - r2
                    if max_radius_for_i < 1e-9:  # Tighter tolerance
                        valid = False
                        break
                    new_radius = min(new_radius, max_radius_for_i)
            
            # Even finer improvement threshold
            if valid and new_radius > old_radius * 1.008:
                final_refined[i, 2] = min(new_radius, max_possible_radius)
                any_improved = True
        
        if not any_improved:
            break
    
    return final_refined

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
