# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math
from shapely.geometry import Polygon

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def hexagon_polygon(center_x, center_y, side_length=1, rotation=0):
    """Create Shapely polygon for a hexagon"""
    vertices = hexagon_vertices(center_x, center_y, rotation, side_length)
    return Polygon(vertices)

def check_hexagon_overlap(hex1_center, hex1_angle, hex2_center, hex2_angle):
    """Proper overlap check using Shapely polygons"""
    hex1 = hexagon_polygon(hex1_center[0], hex1_center[1], 1, hex1_angle)
    hex2 = hexagon_polygon(hex2_center[0], hex2_center[1], 1, hex2_angle)
    return hex1.intersects(hex2)

def check_containment(hex_center, outer_radius):
    """Check if hexagon center is within outer hexagon"""
    distance = math.sqrt(hex_center[0]**2 + hex_center[1]**2)
    return distance <= (outer_radius - 1.0)  # Leave margin for hexagon size

def evaluate_arrangement(inner_hex_data, outer_radius=None):
    """Evaluate if arrangement is valid and compute outer hex side length"""
    # If no outer radius provided, compute it
    if outer_radius is None:
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = (inner_hex_data[i][0], inner_hex_data[i][1])
            # For a unit hexagon, maximum distance from center to any vertex is 1
            dist_to_center = math.sqrt(center[0]**2 + center[1]**2)
            max_dist = max(max_dist, dist_to_center + 1)
        outer_radius = max_dist
    
    # Check for overlaps - use more efficient approach
    num_hexagons = len(inner_hex_data)
    # Early exit if any overlap detected
    for i in range(num_hexagons):
        for j in range(i+1, num_hexagons):
            if check_hexagon_overlap(
                (inner_hex_data[i][0], inner_hex_data[i][1]), inner_hex_data[i][2],
                (inner_hex_data[j][0], inner_hex_data[j][1]), inner_hex_data[j][2]
            ):
                return False, float('inf')
    
    # Check containment for all hexagons
    for i in range(num_hexagons):
        if not check_containment((inner_hex_data[i][0], inner_hex_data[i][1]), outer_radius):
            return False, float('inf')
    
    # Compute required outer hexagon size
    return True, outer_radius

def objective_function(params):
    """Objective function to minimize (negative of 1/outer_hex_side_length)"""
    # Extract parameters
    # First 33 params: 11 hexagons * 3 params (x, y, angle)
    # Last param: outer hex side length
    inner_params = params[:-1]
    outer_side_length = params[-1]
    
    # Reshape inner hex data
    inner_hex_data = inner_params.reshape(-1, 3)
    
    # Check constraints
    if outer_side_length <= 0:
        return 1e10  # Invalid
    
    # Evaluate arrangement properly
    valid, side_length = evaluate_arrangement(inner_hex_data, outer_side_length)
    
    if not valid:
        return 1e10  # Invalid arrangement
    
    # Return negative of 1/outer_hex_side_length (we want to maximize 1/R)
    return -1.0 / side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a more carefully constructed initial configuration based on known optimal patterns
    # Try multiple starting configurations to ensure we don't get stuck in local optima
    candidate_configs = [
        # Configuration 1: Standard hexagonal packing
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 2.0, 0],           # top
            [1.7320508075688772, 1.0, 0],   # top-right
            [1.7320508075688772, -1.0, 0],  # bottom-right
            [0.0, -2.0, 0],          # bottom
            [-1.7320508075688772, -1.0, 0], # bottom-left
            [-1.7320508075688772, 1.0, 0],  # top-left
            [3.4641016151377544, 0.0, 0],   # far right
            [-3.4641016151377544, 0.0, 0],  # far left
            [1.7320508075688772, 3.0, 0],   # upper-right
            [-1.7320508075688772, 3.0, 0],  # upper-left
        ]),
        # Configuration 2: More spread out for better optimization
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 1.95, 0],          # top
            [1.65, 0.95, 0],         # top-right
            [1.65, -0.95, 0],        # bottom-right
            [0.0, -1.95, 0],         # bottom
            [-1.65, -0.95, 0],       # bottom-left
            [-1.65, 0.95, 0],        # top-left
            [3.3, 0.0, 0],           # far right
            [-3.3, 0.0, 0],          # far left
            [1.65, 2.85, 0],         # upper-right
            [-1.65, 2.85, 0],        # upper-left
        ]),
        # Configuration 3: Optimized for tighter packing
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 2.0, 0],           # top
            [1.7320508075688772, 1.0, 0],   # top-right
            [1.7320508075688772, -1.0, 0],  # bottom-right
            [0.0, -2.0, 0],          # bottom
            [-1.7320508075688772, -1.0, 0], # bottom-left
            [-1.7320508075688772, 1.0, 0],  # top-left
            [3.4641016151377544, 0.0, 0],   # far right
            [-3.4641016151377544, 0.0, 0],  # far left
            [1.7320508075688772, 2.99, 0],  # upper-right (slightly closer)
            [-1.7320508075688772, 2.99, 0], # upper-left (slightly closer)
        ]),
        # Configuration 4: Slightly different arrangement to explore more space
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 2.001, 0],         # top (slightly adjusted)
            [1.7320508075688772, 1.001, 0],   # top-right (slightly adjusted)
            [1.7320508075688772, -1.001, 0],  # bottom-right (slightly adjusted)
            [0.0, -2.001, 0],        # bottom (slightly adjusted)
            [-1.7320508075688772, -1.001, 0], # bottom-left (slightly adjusted)
            [-1.7320508075688772, 1.001, 0],  # top-left (slightly adjusted)
            [3.4641016151377544, 0.001, 0],   # far right (slightly adjusted)
            [-3.4641016151377544, 0.001, 0],  # far left (slightly adjusted)
            [1.7320508075688772, 3.001, 0],   # upper-right (slightly adjusted)
            [-1.7320508075688772, 3.001, 0],  # upper-left (slightly adjusted)
        ]),
        # Configuration 5: Asymmetric arrangement to break symmetry
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 2.002, 0],         # top (slightly adjusted)
            [1.7320508075688772, 1.002, 0],   # top-right (slightly adjusted)
            [1.7320508075688772, -1.002, 0],  # bottom-right (slightly adjusted)
            [0.0, -2.002, 0],        # bottom (slightly adjusted)
            [-1.7320508075688772, -1.002, 0], # bottom-left (slightly adjusted)
            [-1.7320508075688772, 1.002, 0],  # top-left (slightly adjusted)
            [3.4641016151377544, 0.002, 0],   # far right (slightly adjusted)
            [-3.4641016151377544, 0.002, 0],  # far left (slightly adjusted)
            [1.7320508075688772, 3.002, 0],   # upper-right (slightly adjusted)
            [-1.7320508075688772, 3.002, 0],  # upper-left (slightly adjusted)
        ])
    ]
    
    # Test all configurations and pick the best valid one with more comprehensive evaluation
    best_valid_config = None
    best_side_length = float('inf')
    best_config_index = -1
    
    # Include more aggressive configurations from evolution history
    additional_configs = [
        # Configuration from evolution history that showed improvement
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 1.998, 0],         # top (more aggressive adjustment)
            [1.732, 1.002, 0],       # top-right (more aggressive adjustment)
            [1.732, -1.002, 0],      # bottom-right (more aggressive adjustment)
            [0.0, -1.998, 0],        # bottom (more aggressive adjustment)
            [-1.732, -1.002, 0],     # bottom-left (more aggressive adjustment)
            [-1.732, 1.002, 0],      # top-left (more aggressive adjustment)
            [3.464, 0.002, 0],       # far right (more aggressive adjustment)
            [-3.464, 0.002, 0],      # far left (more aggressive adjustment)
            [1.732, 2.998, 0],       # upper-right (more aggressive adjustment)
            [-1.732, 2.998, 0],      # upper-left (more aggressive adjustment)
        ]),
        # Even more aggressive configuration
        np.array([
            [0.0, 0.0, 0],           # center
            [0.0, 1.999, 0],         # top (even tighter)
            [1.732, 1.001, 0],       # top-right (tighter)
            [1.732, -1.001, 0],      # bottom-right (tighter)
            [0.0, -1.999, 0],        # bottom (even tighter)
            [-1.732, -1.001, 0],     # bottom-left (tighter)
            [-1.732, 1.001, 0],      # top-left (tighter)
            [3.464, 0.001, 0],       # far right (tighter)
            [-3.464, 0.001, 0],      # far left (tighter)
            [1.732, 2.999, 0],       # upper-right (tighter)
            [-1.732, 2.999, 0],      # upper-left (tighter)
        ])
    ]
    
    all_configs = candidate_configs + additional_configs
    
    for i, config in enumerate(all_configs):
        valid, side_length = evaluate_arrangement(config)
        if valid and side_length < best_side_length:
            best_valid_config = config
            best_side_length = side_length
            best_config_index = i
    
    # If none are valid, use the first one as fallback with more aggressive initial configuration
    if best_valid_config is None:
        best_valid_config = all_configs[0]
        valid, side_length = evaluate_arrangement(best_valid_config)
        if not valid:
            # Fallback to the most aggressive configuration that pushes boundaries
            best_valid_config = np.array([
                [0.0, 0.0, 0],           # center
                [0.0, 1.9995, 0],        # top (very tight)
                [1.73205, 1.0005, 0],    # top-right (very tight)
                [1.73205, -1.0005, 0],   # bottom-right (very tight)
                [0.0, -1.9995, 0],       # bottom (very tight)
                [-1.73205, -1.0005, 0],  # bottom-left (very tight)
                [-1.73205, 1.0005, 0],   # top-left (very tight)
                [3.46410, 0.0005, 0],    # far right (very tight)
                [-3.46410, 0.0005, 0],   # far left (very tight)
                [1.73205, 2.9995, 0],    # upper-right (very tight)
                [-1.73205, 2.9995, 0],   # upper-left (very tight)
            ])
    
    inner_hex_data = best_valid_config
    outer_hex_side_length = best_side_length
    
    # Set up optimization bounds with more aggressive and precise ranges
    bounds = []
    # Positions (x, y) for each hexagon - more aggressive bounds to allow better exploration
    for i in range(11):
        # Use wider bounds to allow more aggressive optimization
        if i in [0]:  # Center hexagon - moderate bounds
            bounds.extend([(-2.0, 2.0), (-2.0, 2.0)])
        elif i in [1, 4]:  # Top and bottom hexagons - vertical spread
            bounds.extend([(-3.0, 3.0), (-3.0, 4.0)])
        elif i in [2, 3, 5, 6]:  # Side hexagons - diagonal spread
            bounds.extend([(-4.0, 4.0), (-4.0, 4.0)])
        elif i in [7, 8]:  # Far left/right hexagons - horizontal spread
            bounds.extend([(-6.0, 6.0), (-3.0, 3.0)])
        else:  # Upper hexagons (9, 10) - vertical spread with horizontal freedom
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0)])
        bounds.extend([(-180, 180)])  # angle bounds
    
    # Outer hexagon size bound - even more precise to allow tight optimization
    bounds.append((outer_hex_side_length * 0.99, outer_hex_side_length * 1.01))
    
    # Flatten initial configuration
    initial_params = inner_hex_data.flatten()
    
    # Use differential evolution for global optimization with highly aggressive parameters
    try:
        # Try multiple optimization runs with very aggressive parameters
        best_result = None
        best_value = float('inf')
        
        # Run several optimization attempts with extremely aggressive parameters
        param_sets = [
            # Very aggressive initial exploration
            {
                'maxiter': 1500,
                'popsize': 200,
                'mutation': (0.999, 1),
                'recombination': 0.9999,
                'tol': 1e-26,
                'seed': 42
            },
            # Medium aggression with more exploitation
            {
                'maxiter': 1200,
                'popsize': 180,
                'mutation': (0.998, 1),
                'recombination': 0.9995,
                'tol': 1e-25,
                'seed': 43
            },
            # Highly focused exploitation
            {
                'maxiter': 1000,
                'popsize': 150,
                'mutation': (0.995, 1),
                'recombination': 0.999,
                'tol': 1e-24,
                'seed': 44
            }
        ]
        
        for params in param_sets:
            result = differential_evolution(
                objective_function,
                bounds,
                args=(),
                seed=params['seed'],
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                mutation=params['mutation'],
                recombination=params['recombination'],
                tol=params['tol'],
                disp=False
            )
            
            if result.success:
                # Check if this result is better than previous best
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result
        
        if best_result is not None:
            inner_params = best_result.x[:-1]
            outer_side_length = best_result.x[-1]
            inner_hex_data_optimized = inner_params.reshape(-1, 3)
            
            # Validate the optimized solution more rigorously
            valid, side_length = evaluate_arrangement(inner_hex_data_optimized, outer_side_length)
            
            if valid:
                # Success case - return the optimized result
                final_positions = inner_hex_data_optimized
                final_size = side_length
            else:
                # If optimization produced invalid result, use original
                final_positions = inner_hex_data
                final_size = outer_hex_side_length
        else:
            # Fallback to original solution if no successful optimization
            final_positions = inner_hex_data
            final_size = outer_hex_side_length
            
    except Exception as e:
        # Fallback to original solution
        final_positions = inner_hex_data
        final_size = outer_hex_side_length
    
    # Apply even more aggressive and precise refinement with targeted adjustments
    if best_config_index >= 0:
        # Apply even more aggressive adjustments based on proven patterns from top performers
        refined_config = best_valid_config.copy()
        
        # Apply ultra-aggressive adjustments that push towards theoretical optimum
        adjustments = [
            # Core aggressive adjustments - maximum impact positions
            (1, 1, -0.08),    # top hexagon DOWN (very aggressive)
            (2, 1, 0.08),     # bottom hexagon UP (very aggressive)
            (3, 0, -0.07),    # top-right LEFT (very aggressive)
            (4, 0, 0.07),     # top-left RIGHT (very aggressive)
            (5, 0, -0.07),    # bottom-right LEFT (very aggressive)
            (6, 0, 0.07),     # bottom-left RIGHT (very aggressive)
            (0, 1, -0.06),    # center slight shift DOWN (very aggressive)
            (9, 1, -0.08),    # upper right cluster DOWN (very aggressive)
            (10, 1, -0.08),   # upper left cluster DOWN (very aggressive)
            # Extreme position adjustments for maximum packing
            (7, 0, -0.05),    # far right hexagon LEFT (very aggressive)
            (8, 0, 0.05),     # far left hexagon RIGHT (very aggressive)
            # Diagonal relationship adjustments
            (3, 1, -0.01),    # top-right vertical adjustment
            (4, 1, 0.01),     # top-left vertical adjustment
            (5, 1, -0.01),    # bottom-right vertical adjustment
            (6, 1, 0.01),     # bottom-left vertical adjustment
            # Center and fine-tuning
            (0, 0, -0.003),   # center slight horizontal shift
            (9, 0, -0.005),   # upper right cluster horizontal shift
            (10, 0, 0.005),   # upper left cluster horizontal shift
        ]
        
        # Apply all adjustments systematically
        for idx, coord_idx, adjustment in adjustments:
            if idx < len(refined_config):
                refined_config[idx][coord_idx] += adjustment
        
        valid, side_length = evaluate_arrangement(refined_config)
        if valid and side_length < best_side_length:
            best_valid_config = refined_config
            best_side_length = side_length
    
    outer_hex_data = np.array([0, 0, 0])
    return best_valid_config, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
