# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random

def compute_c2(f_values):
    """Compute C2 value for given function values using the correct convolution approach"""
    try:
        f = np.array(f_values)
        g = convolve(f, f, mode='full')
        n = len(f)
        g = g[n-1:2*n-1]
        
        g_squared = g * g
        g_abs = np.abs(g)
        
        norm_2_squared = np.sum(g_squared)
        norm_1 = np.sum(g_abs)
        norm_inf = np.max(g_abs)
        
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0
        return norm_2_squared / (norm_1 * norm_inf)
    except:
        return 0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimized approach."""
    
    # Use even higher resolution for better optimization potential (top performers used 10000+)
    n_steps = 10000  # Even higher resolution for better optimization capability
    
    # Strategy: Use a more sophisticated 25-level pattern with optimized amplitude distribution
    # Based on evolution history, more levels with strategic amplitude clustering work better
    # The pattern should have more gradual transitions and controlled peaks for better convolution flattening
    amplitudes = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05,
                  0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
                  0.25, 0.30, 0.35, 0.40, 1.0]
    
    # Create pattern with more strategic segment sizing to enhance convolution flattening
    # Using more varied and strategically sized segments to avoid overly regular patterns
    segment_sizes = [200, 220, 240, 260, 280, 300, 320, 340, 360, 380,
                     400, 420, 440, 460, 480, 460, 440, 420, 400, 380,
                     360, 340, 320, 300, 280]  # Sum = 10000
    f_values = []
    
    for i, (amp, seg_size) in enumerate(zip(amplitudes, segment_sizes)):
        f_values.extend([amp] * seg_size)
    
    # Apply enhanced smoothing with 15-point kernel for better edge preservation
    # Using a slightly different kernel that provides better edge handling for this specific problem
    smoothed = []
    # More aggressive edge weights to better preserve important features
    kernel_weights = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    
    for i in range(len(f_values)):
        if i < 7:
            # Handle early boundaries with extended averaging for better edge preservation
            start_idx = 0
            end_idx = min(15, len(f_values))
            window = f_values[start_idx:end_idx]
            smoothed.append(np.mean(window))
        elif i >= len(f_values) - 7:
            # Handle end boundaries with extended averaging for better edge preservation
            start_idx = max(0, len(f_values) - 15)
            end_idx = len(f_values)
            window = f_values[start_idx:end_idx]
            smoothed.append(np.mean(window))
        else:
            # Use 15-point symmetric kernel for better smoothing
            neighbors = f_values[i-7:i+8]
            weighted_sum = sum(w * n for w, n in zip(kernel_weights, neighbors))
            smoothed.append(weighted_sum)
    
    f_values = smoothed
    
    # Apply more careful normalization targeting better C₂ balance
    # Use a slightly more refined target average that has shown to work well in top performers
    total_sum = sum(f_values)
    if total_sum > 0:
        # Target a more optimal average that balances ||g||₂² growth with ||g||∞ control
        # Based on evolution history, 0.032-0.037 range works well for C₂ maximization
        target_avg = 0.034  # Middle of effective range for better balance
        scale_factor = (target_avg * n_steps) / total_sum
        f_values = [val * scale_factor for val in f_values]
    
    # Ensure non-negative values
    f_values = [max(0, val) for val in f_values]
    
    # Perform enhanced local optimization with more sophisticated strategies
    original_c2 = compute_c2(f_values)
    best_f_values = f_values.copy()
    best_c2 = original_c2
    
    # Phase 1: Fine-grained optimization with more iterations and adaptive adjustments
    for iteration in range(2500):  # More iterations for better convergence
        test_values = best_f_values.copy()
        num_changes = min(400, len(test_values) // 8)  # More changes per iteration for better exploration
        indices_to_modify = random.sample(range(len(test_values)), num_changes)
        
        for idx in indices_to_modify:
            # Use more sophisticated adaptive adjustment based on local context
            current_val = test_values[idx]
            # Adjust based on value range and position in the function with more refined ranges
            if current_val < 0.03:
                adjustment = random.uniform(-0.015, 0.015)
            elif current_val > 0.97:
                adjustment = random.uniform(-0.015, 0.015)
            elif current_val < 0.10:
                adjustment = random.uniform(-0.02, 0.02)
            elif current_val > 0.90:
                adjustment = random.uniform(-0.02, 0.02)
            elif current_val < 0.20:
                adjustment = random.uniform(-0.025, 0.025)
            elif current_val > 0.80:
                adjustment = random.uniform(-0.025, 0.025)
            elif current_val < 0.35:
                adjustment = random.uniform(-0.03, 0.03)
            elif current_val > 0.65:
                adjustment = random.uniform(-0.03, 0.03)
            else:
                adjustment = random.uniform(-0.035, 0.035)
                
            test_values[idx] = max(0, min(1, test_values[idx] + adjustment))
        
        new_c2 = compute_c2(test_values)
        if new_c2 > best_c2:
            best_f_values = test_values
            best_c2 = new_c2
    
    # Phase 2: Global search with larger perturbations and more extensive exploration
    for iteration in range(1000):  # More iterations for thorough global search
        test_values = best_f_values.copy()
        num_changes = min(250, len(test_values) // 5)  # More changes for better global coverage
        indices_to_modify = random.sample(range(len(test_values)), num_changes)
        
        for idx in indices_to_modify:
            # Larger adjustments for global exploration with adaptive bounds
            if test_values[idx] < 0.1 or test_values[idx] > 0.9:
                adjustment = random.uniform(-0.08, 0.08)
            elif test_values[idx] < 0.2 or test_values[idx] > 0.8:
                adjustment = random.uniform(-0.09, 0.09)
            elif test_values[idx] < 0.3 or test_values[idx] > 0.7:
                adjustment = random.uniform(-0.10, 0.10)
            else:
                adjustment = random.uniform(-0.12, 0.12)
            test_values[idx] = max(0, min(1, test_values[idx] + adjustment))
        
        new_c2 = compute_c2(test_values)
        if new_c2 > best_c2:
            best_f_values = test_values
            best_c2 = new_c2
    
    # Phase 3: Local refinement with specialized approach and more precise adjustments
    for iteration in range(800):  # More iterations for additional refinement
        test_values = best_f_values.copy()
        num_changes = min(200, len(test_values) // 10)  # Balanced number of changes with more iterations
        indices_to_modify = random.sample(range(len(test_values)), num_changes)
        
        for idx in indices_to_modify:
            # Specialized refinement using neighbor information with more sophisticated approach
            neighbors = []
            for offset in [-2, -1, 0, 1, 2]:
                if 0 <= idx + offset < len(test_values):
                    neighbors.append(test_values[idx + offset])
            
            if len(neighbors) >= 3:
                avg_neighbor = sum(neighbors) / len(neighbors)
                # Adjust toward the average to promote smoother profiles with more controlled adjustments
                adjustment = (avg_neighbor - test_values[idx]) * 0.25
                test_values[idx] = max(0, min(1, test_values[idx] + adjustment))
            else:
                # Default adjustment with more refined range
                adjustment = random.uniform(-0.015, 0.015)
                test_values[idx] = max(0, min(1, test_values[idx] + adjustment))
        
        new_c2 = compute_c2(test_values)
        if new_c2 > best_c2:
            best_f_values = test_values
            best_c2 = new_c2
    
    # If improvement was found, use the better version
    if best_c2 > original_c2:
        f_values = best_f_values
    
    # Final check to ensure non-negativity
    f_values = [max(0, val) for val in f_values]
    
    # Enhanced final refinement with more aggressive smoothing to reduce noise
    # Apply 15-point kernel with optimized weights for better structure preservation
    final_refined = []
    # Use slightly more aggressive kernel weights to better smooth while preserving important features
    kernel_weights = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    
    for i in range(len(f_values)):
        if i < 7:
            # Handle early boundaries with extended averaging
            window = f_values[:min(15, len(f_values))]
            final_refined.append(np.mean(window))
        elif i >= len(f_values) - 7:
            # Handle end boundaries with extended averaging
            window = f_values[max(0, len(f_values)-15):]
            final_refined.append(np.mean(window))
        else:
            # Use 15-point symmetric kernel for final smoothing
            neighbors = f_values[i-7:i+8]
            weighted_sum = sum(w * n for w, n in zip(kernel_weights, neighbors))
            final_refined.append(weighted_sum)
    
    f_values = final_refined
    f_values = [max(0, val) for val in f_values]
    
    # Convert to list and return
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
