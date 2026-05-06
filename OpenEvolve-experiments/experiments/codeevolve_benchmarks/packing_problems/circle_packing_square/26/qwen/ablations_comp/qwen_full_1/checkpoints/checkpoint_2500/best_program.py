# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Use a better initialization strategy with multiple attempts and improved hexagonal packing
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies for better exploration
    for attempt in range(12):  # More attempts for better exploration
        np.random.seed(42 + attempt)
        random.seed(42 + attempt)
        
        circles = np.zeros((n, 3))
        
        # Strategy 1: Improved hexagonal packing with better spacing and more careful positioning
        rows = 5
        cols = 6
        spacing_x = 0.85 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 0.85 / (rows - 1) if rows > 1 else 0.5
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.5
                # Use systematic positioning with minimal randomness for better consistency
                x = 0.075 + (j * spacing_x + x_offset)
                y = 0.075 + (i * spacing_y)
                # Clamp to bounds more carefully
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                # Start with larger initial radius to allow for expansion
                circles[idx] = [x, y, 0.05]  # Even larger initial radius
                idx += 1
        
        # Calculate initial sum
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    # Use the best initialization found
    if best_circles is not None:
        circles = best_circles
    
    # Multi-pass optimization approach for better results
    # Pass 1: Aggressive radius expansion with adaptive growth rates and better early stopping
    max_iter = 1200  # More iterations for better convergence
    last_improvement = 0
    for iteration in range(max_iter):
        improved = False
        
        # Shuffle order to avoid getting stuck in local minima
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if circles[i][2] < 0.0001:
                continue
                
            # Calculate max possible radius for this circle
            max_radius = min(
                circles[i][0], 
                1 - circles[i][0],
                circles[i][1], 
                1 - circles[i][1]
            )
            
            # Check overlap with other circles efficiently using spatial indexing
            min_dist = float('inf')
            
            # Use more efficient overlap checking for early iterations
            if iteration < max_iter * 0.4:
                # For early iterations, do more comprehensive checks
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                        overlap_dist = dist - circles[j][2]
                        min_dist = min(min_dist, overlap_dist)
            else:
                # For later iterations, be more selective with spatial indexing
                from scipy.spatial import cKDTree
                tree = cKDTree(circles[:, :2])
                neighbors = tree.query_ball_point(circles[i, :2], max_radius + 0.1)
                for j in neighbors:
                    if i != j:
                        dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                        overlap_dist = dist - circles[j][2]
                        min_dist = min(min_dist, overlap_dist)
            
            # Can potentially increase radius
            if min_dist > 0:
                max_possible = min(max_radius, min_dist)
                # More sophisticated adaptive growth based on radius and constraint level
                constraint_ratio = min_dist / max_radius if max_radius > 0 else 1.0
                if circles[i][2] < 0.01:
                    growth_factor = 2.5  # Slightly less aggressive for very small radii
                elif circles[i][2] < 0.03:
                    growth_factor = 2.0  # Less aggressive for small radii
                elif circles[i][2] < 0.07:
                    growth_factor = 1.8  # Moderate for medium-small radii
                elif circles[i][2] < 0.15:
                    growth_factor = 1.4  # Moderate for medium radii
                else:
                    growth_factor = 1.2  # Conservative for large radii
                
                # Adjust growth factor based on how constrained the circle is
                if constraint_ratio < 0.05:
                    growth_factor *= 1.5  # More aggressive when less constrained
                elif constraint_ratio < 0.2:
                    growth_factor *= 1.2  # Slightly more aggressive
                elif constraint_ratio < 0.4:
                    growth_factor *= 1.1  # Normal
                else:
                    growth_factor *= 1.0  # Conservative when highly constrained
                
                new_radius = min(max_possible, circles[i][2] * growth_factor)
                if new_radius > circles[i][2]:
                    circles[i][2] = new_radius
                    improved = True
        
        # More patient early stopping with better convergence detection
        if not improved:
            last_improvement += 1
            if last_improvement > 300:  # More patience for convergence
                break
        else:
            last_improvement = 0
    
    # Enhanced local position optimization with better search strategy and spatial indexing
    for _ in range(600):  # Increased iterations for better position optimization
        improved = False
        # Process circles in random order to avoid systematic bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            best_radius = circles[i][2]
            best_pos = [circles[i][0], circles[i][1]]
            
            # Prioritize search based on constraint level
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                    overlap_dist = dist - circles[j][2]
                    min_dist = min(min_dist, overlap_dist)
            
            # Adjust search intensity based on constraint level with more aggressive ranges for better exploration
            if min_dist < 0.001:  # Extremely constrained
                search_steps = [0.08, 0.04, 0.02, 0.01, 0.005]
            elif min_dist < 0.005:  # Highly constrained
                search_steps = [0.06, 0.03, 0.015, 0.008, 0.004]
            elif min_dist < 0.02:  # Moderately constrained
                search_steps = [0.05, 0.025, 0.012, 0.006, 0.003]
            else:  # Less constrained
                search_steps = [0.04, 0.02, 0.01, 0.005]
            
            # Try a more adaptive search around current position with spatial indexing for efficiency
            for step in search_steps:
                # Vary search pattern based on step size
                if step >= 0.02:  # Larger steps - broader search
                    search_space = [-step, -step*0.5, 0, step*0.5, step]
                else:  # Smaller steps - more precise search
                    search_space = [-step, 0, step]
                    
                # Add extra points for more thorough search
                if len(search_space) == 3 and step < 0.01:  # For smallest steps
                    search_space = [-step, -step*0.25, 0, step*0.25, step]
                    
                for dx in search_space:
                    for dy in search_space:
                        new_x = circles[i][0] + dx
                        new_y = circles[i][1] + dy
                        
                        # Check bounds
                        if (new_x >= circles[i][2] and new_x <= 1 - circles[i][2] and
                            new_y >= circles[i][2] and new_y <= 1 - circles[i][2]):
                            
                            # Quick overlap check using spatial indexing for efficiency
                            from scipy.spatial import cKDTree
                            tree = cKDTree(circles[:, :2])
                            neighbors = tree.query_ball_point([new_x, new_y], circles[i][2] + 0.15)
                            
                            valid = True
                            overlap_count = 0
                            for j in neighbors:
                                if i != j:
                                    dist = np.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                                    if dist < circles[i][2] + circles[j][2]:
                                        valid = False
                                        overlap_count += 1
                                        # Early exit if too many overlaps
                                        if overlap_count > 2:  # Reduced threshold for faster processing
                                            break
                            
                            if valid:
                                # Try to increase radius
                                max_radius = min(new_x, 1-new_x, new_y, 1-new_y)
                                min_dist = float('inf')
                                for j in range(n):
                                    if i != j:
                                        dist = np.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                                        min_dist = min(min_dist, dist - circles[j][2])
                                
                                if min_dist > 0:
                                    new_radius = min(max_radius, min_dist)
                                    # Be more aggressive in accepting improvements
                                    if new_radius > best_radius * 1.01:  # Accept slightly better improvements
                                        best_radius = new_radius
                                        best_pos = [new_x, new_y]
                                        improved = True
            
            circles[i][0] = best_pos[0]
            circles[i][1] = best_pos[1]
            circles[i][2] = best_radius
        
        if not improved:
            break
    
    # Pass 3: Enhanced final fine-tuning with better constraint handling and more systematic optimization
    for _ in range(500):  # More iterations for better final tuning
        improved = False
        
        # Process in order of constraint level (most constrained first) for maximum impact
        constraint_levels = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                    overlap_dist = dist - circles[j][2]
                    min_dist = min(min_dist, overlap_dist)
            constraint_levels.append((min_dist, i))
        
        # Sort by constraint level (most constrained first)
        constraint_levels.sort(key=lambda x: x[0])
        
        # Optimize the most constrained ones first (focus on top 15 for better impact)
        for _, i in constraint_levels[:15]:  # Focus on top 15 most constrained
            # Calculate what radius we can have without violating constraints
            max_radius = min(
                circles[i][0], 
                1 - circles[i][0],
                circles[i][1], 
                1 - circles[i][1]
            )
            
            # Check all overlaps more carefully with spatial indexing
            from scipy.spatial import cKDTree
            tree = cKDTree(circles[:, :2])
            neighbors = tree.query_ball_point(circles[i, :2], max_radius + 0.1)
            
            min_overlap_distance = float('inf')
            for j in neighbors:
                if i != j:
                    dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                    overlap_dist = dist - circles[j][2]
                    min_overlap_distance = min(min_overlap_distance, overlap_dist)
            
            # Update radius if beneficial with more aggressive acceptance
            if min_overlap_distance > 0:
                new_radius = min(max_radius, min_overlap_distance)
                if new_radius > circles[i][2] * 1.01:  # Accept meaningful improvements
                    circles[i][2] = new_radius
                    improved = True
        
        # Also do a quick global pass for any remaining improvements
        for i in range(n):
            # Try to slightly adjust position to gain more radius
            best_radius = circles[i][2]
            best_pos = [circles[i][0], circles[i][1]]
            
            # More comprehensive neighborhood search with varying step sizes
            search_deltas = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
            for dx in search_deltas:
                for dy in search_deltas:
                    new_x = circles[i][0] + dx
                    new_y = circles[i][1] + dy
                    
                    # Check bounds
                    if (new_x >= circles[i][2] and new_x <= 1 - circles[i][2] and
                        new_y >= circles[i][2] and new_y <= 1 - circles[i][2]):
                        
                        # Check overlaps
                        valid = True
                        from scipy.spatial import cKDTree
                        tree = cKDTree(circles[:, :2])
                        neighbors = tree.query_ball_point([new_x, new_y], circles[i][2] + 0.1)
                        
                        for j in neighbors:
                            if i != j:
                                dist = np.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                                if dist < circles[i][2] + circles[j][2]:
                                    valid = False
                                    break
                        
                        if valid:
                            # Try to increase radius
                            max_radius = min(new_x, 1-new_x, new_y, 1-new_y)
                            min_dist = float('inf')
                            for j in range(n):
                                if i != j:
                                    dist = np.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                                    min_dist = min(min_dist, dist - circles[j][2])
                            
                            if min_dist > 0:
                                new_radius = min(max_radius, min_dist)
                                if new_radius > best_radius * 1.005:  # Very small improvement threshold
                                    best_radius = new_radius
                                    best_pos = [new_x, new_y]
            
            circles[i][0] = best_pos[0]
            circles[i][1] = best_pos[1]
            circles[i][2] = best_radius
        
        # More aggressive early stopping with better convergence detection
        if not improved:
            if _ > 250:  # Stop if no significant improvement for many iterations
                break
    
    return circles


# EVOLVE-BLOCK-END
