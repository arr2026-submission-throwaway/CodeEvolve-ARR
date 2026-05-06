# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal

def compute_c2(f_values):
    """Compute the C2 value for given step function values using correct piecewise linear integration."""
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = np.convolve(f, f, mode='full')
    
    # Get the middle portion corresponding to convolution over [-1/4, 1/4]
    mid_idx = len(g) // 2
    half_len = len(f)
    start_idx = max(0, mid_idx - half_len + 1)
    end_idx = min(len(g), mid_idx + half_len)
    g_relevant = g[start_idx:end_idx]
    
    if len(g_relevant) <= 1:
        return 0.0
    
    # Compute norms using the specified piecewise linear integration method
    # For interval with heights y1, y2 and width h, contribution is (h/3)(y1² + y1*y2 + y2²)
    g_abs = np.abs(g_relevant)
    
    # ||g||₂² using correct piecewise linear integration approach
    g_norm_2_squared = 0.0
    if len(g_abs) > 1:
        # For consecutive points, integrate as trapezoidal with quadratic approximation
        # This matches the specification in the problem description
        for i in range(len(g_abs) - 1):
            y1, y2 = g_abs[i], g_abs[i+1]
            # Width is 1 for our discretization (step size)
            g_norm_2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # ||g||₁ - sum of absolute values (approximated as sum of values)
    g_norm_1 = np.sum(g_abs)
    
    # ||g||∞ - maximum absolute value
    g_norm_inf = np.max(g_abs)
    
    # Avoid division by zero
    if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
        return 0.0
        
    c2 = g_norm_2_squared / (g_norm_1 * g_norm_inf)
    return c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using advanced optimization."""
    # Use higher resolution for better optimization potential
    n_steps = 5000
    
    # Strategy 1: Enhanced multi-peak pattern with optimized geometry for flat convolution
    # Use more evenly distributed peaks with improved amplitude profile
    advanced_multi_peak = []
    # Create more precisely positioned peaks for better convolution properties
    peak_positions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]  # Even more evenly distributed
    peak_heights = [1.6, 1.4, 1.2, 1.0, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7]  # Optimized amplitudes with more symmetry
    
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        value = 0.0
        # Use optimized peak positions and heights
        for j in range(len(peak_positions)):
            peak_pos = peak_positions[j]
            peak_height = peak_heights[j]
            # Use slightly wider peaks to promote more uniform convolution
            sigma = n_steps / 16
            gaussian = peak_height * np.exp(-((pos - peak_pos)**2) / (2 * (sigma/n_steps)**2))
            value += gaussian
    advanced_multi_peak.append(max(0.0, value))
    candidates = [advanced_multi_peak]
    
    # Strategy 11: Enhanced multi-peak with golden ratio spacing for optimal distribution
    golden_multi_peak = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    num_peaks = 10
    peak_positions = [(i / (num_peaks - 1)) ** phi for i in range(num_peaks)]
    peak_heights = [1.9 * (0.88 ** i) for i in range(num_peaks)]
    
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        value = 0.0
        for j in range(len(peak_positions)):
            peak_pos = peak_positions[j]
            peak_height = peak_heights[j]
            sigma = n_steps / 16
            gaussian = peak_height * np.exp(-((pos - peak_pos)**2) / (2 * (sigma/n_steps)**2))
            value += gaussian
        golden_multi_peak.append(max(0.0, value))
    candidates.append(golden_multi_peak)
    
    # Strategy 12: Fractal-inspired multi-scale pattern
    fractal_pattern = []
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Create fractal-like pattern with self-similar structure
        val = 0.72 + 0.18 * np.sin(pos * np.pi * 10) + 0.12 * np.cos(pos * np.pi * 20) + \
              0.06 * np.sin(pos * np.pi * 40) + 0.03 * np.cos(pos * np.pi * 80)
        fractal_pattern.append(max(0.0, val))
    candidates.append(fractal_pattern)
    
    # Strategy 13: Quantum-like interference pattern
    quantum_pattern = []
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Create pattern that mimics quantum interference
        val = 0.62 + 0.28 * np.sin(pos * np.pi * 14) * np.cos(pos * np.pi * 28) + \
              0.16 * np.sin(pos * np.pi * 42) * np.cos(pos * np.pi * 56) + \
              0.06 * np.sin(pos * np.pi * 70)
        quantum_pattern.append(max(0.0, val))
    candidates.append(quantum_pattern)
    
    # Strategy 14: Multi-scale sinc pattern with enhanced modulation
    enhanced_sinc_multi = []
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Multi-scale sinc pattern with better flatness properties
        sinc_val = np.sinc(12 * (pos - 0.5)) if pos != 0.5 else 1.0
        mod_val = 0.92 + 0.08 * np.sin(32 * np.pi * pos) + 0.06 * np.cos(64 * np.pi * pos) + \
                  0.04 * np.sin(96 * np.pi * pos) + 0.02 * np.cos(128 * np.pi * pos) + \
                  0.01 * np.sin(160 * np.pi * pos) + 0.005 * np.cos(192 * np.pi * pos)
        enhanced_sinc_multi.append(max(0, sinc_val * mod_val))
    candidates.append(enhanced_sinc_multi)
    
    # Strategy 2: Enhanced bell pattern with optimized oscillation
    x = np.linspace(-0.25, 0.25, n_steps)
    # Wider bell with optimized oscillation for better convolution flatness
    bell = np.exp(-((x - 0.0) ** 2) / 0.025)  # Slightly wider bell to spread energy
    # Use frequencies that create destructive interference in convolution
    oscillation = 0.72 + 0.28 * np.sin(14 * np.pi * x) + 0.2 * np.sin(28 * np.pi * x) + 0.15 * np.sin(42 * np.pi * x) + 0.1 * np.sin(56 * np.pi * x) + 0.05 * np.sin(70 * np.pi * x) + 0.02 * np.sin(84 * np.pi * x)
    pattern1 = bell * oscillation
    candidates.append(np.maximum(pattern1, 0).tolist())
    
    # Strategy 3: Multi-scale optimized bell pattern
    multi_scale_bell = []
    center = n_steps // 2
    sigma1 = n_steps / 10  # Slightly wider first sigma for better spread
    sigma2 = n_steps / 15
    sigma3 = n_steps / 20
    for i in range(n_steps):
        x1 = (i - center) / sigma1
        x2 = (i - center) / sigma2
        x3 = (i - center) / sigma3
        # Combine three bell shapes with different spreads
        val1 = 0.75 * np.exp(-0.5 * x1 * x1)
        val2 = 0.25 * np.exp(-0.5 * x2 * x2)
        val3 = 0.05 * np.exp(-0.5 * x3 * x3)
        multi_scale_bell.append(max(0, val1 + val2 + val3))
    candidates.append(multi_scale_bell)
    
    # Strategy 4: Wavelet-inspired with precise frequency control
    precise_wavelet = []
    for i in range(n_steps):
        pos = i / n_steps
        # Precise wavelet pattern with mathematical optimization
        val = 0.52 + 0.30 * np.sin(pos * np.pi * 8) + 0.22 * np.cos(pos * np.pi * 16) + \
              0.16 * np.sin(pos * np.pi * 24) + 0.10 * np.cos(pos * np.pi * 32) + \
              0.07 * np.sin(pos * np.pi * 40)
        precise_wavelet.append(max(0.0, val))
    candidates.append(precise_wavelet)
    
    # Strategy 5: Optimized mathematical flat pattern with better frequency balance
    optimized_flat = []
    for i in range(n_steps):
        pos = i / n_steps
        # Enhanced mathematical pattern with better control over harmonics - more balanced
        val = 0.97 + 0.03 * np.cos(pos * np.pi * 14) + 0.02 * np.sin(pos * np.pi * 28) + \
              0.015 * np.cos(pos * np.pi * 42) + 0.01 * np.sin(pos * np.pi * 56) + \
              0.005 * np.cos(pos * np.pi * 70)
        optimized_flat.append(max(0.0, val))
    candidates.append(optimized_flat)
    
    # Strategy 6: Enhanced sinc-based pattern with better modulation
    enhanced_sinc = []
    for i in range(n_steps):
        pos = i / n_steps
        # Enhanced sinc pattern with more careful modulation
        sinc_val = np.sinc(10 * (pos - 0.5)) if pos != 0.5 else 1.0
        mod_val = 0.88 + 0.12 * np.sin(28 * np.pi * pos) + 0.08 * np.cos(56 * np.pi * pos) + \
                  0.05 * np.sin(84 * np.pi * pos) + 0.02 * np.cos(112 * np.pi * pos) + \
                  0.01 * np.sin(140 * np.pi * pos) + 0.005 * np.cos(168 * np.pi * pos)
        enhanced_sinc.append(max(0, sinc_val * mod_val))
    candidates.append(enhanced_sinc)
    
    # Strategy 7: Advanced anti-correlated pattern with optimized phase relationships
    advanced_anti_correlated = []
    for i in range(n_steps):
        pos = i / n_steps
        # Optimized anti-correlated pattern with better phase alignment - more harmonics
        val = 0.52 + 0.38 * np.sin(pos * np.pi * 12) + 0.25 * np.cos(pos * np.pi * 24) + \
              0.15 * np.sin(pos * np.pi * 36) + 0.1 * np.cos(pos * np.pi * 48) + \
              0.06 * np.sin(pos * np.pi * 60) + 0.03 * np.cos(pos * np.pi * 72)
        advanced_anti_correlated.append(max(0.0, val))
    candidates.append(advanced_anti_correlated)
    
    # Strategy 8: High-frequency mathematical pattern for better flatness
    high_freq_math = []
    for i in range(n_steps):
        pos = i / n_steps
        # Enhanced mathematical pattern with higher frequency components
        val = 0.96 + 0.025 * np.cos(pos * np.pi * 18) + 0.018 * np.sin(pos * np.pi * 36) + \
              0.012 * np.cos(pos * np.pi * 54) + 0.008 * np.sin(pos * np.pi * 72) + \
              0.004 * np.cos(pos * np.pi * 90) + 0.002 * np.sin(pos * np.pi * 108) + \
              0.001 * np.cos(pos * np.pi * 126) + 0.0005 * np.sin(pos * np.pi * 144)
        high_freq_math.append(max(0.0, val))
    candidates.append(high_freq_math)
    
    # Strategy 9: Optimized multi-peak pattern with mathematical precision
    optimized_multi_peak_v2 = []
    num_peaks = 12  # More peaks for better optimization potential
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        value = 0.0
        # Place peaks with mathematical precision using log-spaced distribution
        for j in range(num_peaks):
            # Logarithmic spacing to concentrate peaks near center
            if num_peaks > 1:
                t = j / (num_peaks - 1)
                # Transform to concentrate peaks near center
                peak_pos = 0.5 * (np.tanh(2 * (t - 0.5)) + 1)
            else:
                peak_pos = 0.5
            # Exponential decay for amplitudes to promote flat convolution
            peak_height = 1.8 * (0.8 ** j)
            # Wider peaks for better flatness in convolution
            sigma = n_steps / 18
            gaussian = peak_height * np.exp(-((pos - peak_pos)**2) / (2 * (sigma/n_steps)**2))
            value += gaussian
        optimized_multi_peak_v2.append(max(0.0, value))
    candidates.append(optimized_multi_peak_v2)
    
    # Strategy 10: Enhanced balanced pattern with better transition properties
    enhanced_balanced = []
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Create a pattern with smoother transitions and optimized height ratios
        if pos < 0.25:
            # Gentle ramp up
            enhanced_balanced.append(0.72 + 0.28 * np.sin(pos * np.pi * 2))
        elif pos < 0.75:
            # High plateau
            enhanced_balanced.append(1.25)
        else:
            # Gentle ramp down
            enhanced_balanced.append(0.72 + 0.28 * np.sin(pos * np.pi * 2))
    candidates.append(enhanced_balanced)
    
    # Evaluate all candidates to better explore the solution space
    best_candidate = None
    best_c2 = -1
    
    # Evaluate first few candidates to quickly identify promising ones
    # We'll evaluate the first 18 candidates for speed, then do thorough evaluation of top performers
    for i, candidate in enumerate(candidates[:18]):
        c2 = compute_c2(candidate)
        if c2 > best_c2:
            best_c2 = c2
            best_candidate = candidate
    
    # Now do more thorough evaluation of top candidates to avoid missing better solutions
    # Evaluate top 30 candidates based on initial scoring
    candidate_scores = []
    for i, candidate in enumerate(candidates):
        c2 = compute_c2(candidate)
        candidate_scores.append((c2, i, candidate))
    
    # Sort by score descending and take top 30 candidates
    candidate_scores.sort(reverse=True)
    top_candidates = candidate_scores[:30]
    
    # Evaluate top candidates more carefully
    for c2, i, candidate in top_candidates:
        if c2 > best_c2:
            best_c2 = c2
            best_candidate = candidate
    
    # Apply enhanced local optimization to the best candidate
    if best_candidate is not None:
        best_f = best_candidate.copy()
        best_c2 = compute_c2(best_f)
        
        # Enhanced optimization with more aggressive exploration and better strategy
        # Stage 1: Coarse global exploration with larger perturbations
        for iteration in range(600):  # More iterations for better convergence
            test_f = best_f.copy()
            num_changes = min(300, len(test_f) // 3)  # More changes for exploration
            if iteration > 300:  # Reduce changes in later stages to prevent overshooting
                num_changes = max(35, num_changes // 2)
            for _ in range(num_changes):
                idx = np.random.randint(len(test_f))
                # Adaptive perturbation size based on iteration and value
                if iteration < 150:
                    adjustment = np.random.normal(0, 0.7)  # Larger perturbations early
                elif iteration < 400:
                    adjustment = np.random.normal(0, 0.35)   # Medium perturbations
                else:
                    adjustment = np.random.normal(0, 0.2)  # Smaller perturbations late
                
                test_f[idx] = max(0, test_f[idx] + adjustment)
            
            test_c2 = compute_c2(test_f)
            if test_c2 > best_c2:
                best_f = test_f
                best_c2 = test_c2
        
        # Stage 2: Fine-tuning with medium perturbations
        for iteration in range(500):  # More iterations for better fine-tuning
            test_f = best_f.copy()
            num_changes = min(200, len(test_f) // 5)  # Moderate changes
            for _ in range(num_changes):
                idx = np.random.randint(len(test_f))
                # Medium perturbation for fine-tuning
                adjustment = np.random.normal(0, 0.2)  # Medium perturbations
                test_f[idx] = max(0, test_f[idx] + adjustment)
            
            test_c2 = compute_c2(test_f)
            if test_c2 > best_c2:
                best_f = test_f
                best_c2 = test_c2
        
        # Stage 3: Local refinement with small perturbations
        for iteration in range(400):  # More iterations for final tuning
            test_f = best_f.copy()
            num_changes = min(180, len(test_f) // 6)  # Fewer changes but more precise
            for _ in range(num_changes):
                idx = np.random.randint(len(test_f))
                # Small perturbation for final tuning
                adjustment = np.random.normal(0, 0.1)  # Smaller perturbations
                test_f[idx] = max(0, test_f[idx] + adjustment)
            
            test_c2 = compute_c2(test_f)
            if test_c2 > best_c2:
                best_f = test_f
                best_c2 = test_c2
        
        # Apply more effective smoothing to preserve key features
        smoothed_f = []
        # Use a more moderate smoothing approach to preserve important structure
        window_size = 20  # Slightly larger window for better smoothing
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply a simpler but effective smoothing approach
        for i in range(len(best_f)):
            start = max(0, i - window_size//2)
            end = min(len(best_f), i + window_size//2 + 1)
            avg = np.mean(best_f[start:end])
            smoothed_f.append(avg)
        
        return smoothed_f
    else:
        # Fallback to more sophisticated pattern
        x = np.linspace(-0.25, 0.25, n_steps)
        # Create a pattern that combines multiple frequencies for better convolution properties
        base_pattern = (
            0.7 * np.exp(-x**2 / (2 * 0.08**2)) +
            0.2 * np.sin(12 * np.pi * x) * np.exp(-x**2 / (2 * 0.12**2)) +
            0.1 * np.cos(24 * np.pi * x) * np.exp(-x**2 / (2 * 0.15**2))
        )
        return np.maximum(base_pattern, 0).tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
