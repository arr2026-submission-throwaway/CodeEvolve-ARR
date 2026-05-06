# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List

def compute_c2_stable(f):
    """Compute C2 value with enhanced numerical stability"""
    try:
        # Perform convolution (autoconvolution)
        g = np.convolve(f, f, mode='full')
        # Trim to appropriate size (center portion)
        half_len = len(g) // 2
        g_trimmed = g[half_len - len(f) + 1 : half_len + len(f) - 1]
        
        # Compute norms using the same method as the evaluator
        g_abs = np.abs(g_trimmed)
        
        # Add robust numerical stability checks
        l2_sq = np.sum(g_abs**2)
        l1 = np.sum(g_abs)
        l_inf = np.max(g_abs)
        
        # Avoid division by zero or very small numbers with strict thresholds
        epsilon = 1e-15
        if l1 > epsilon and l_inf > epsilon:
            return l2_sq / (l1 * l_inf)
        else:
            return 0.0
    except Exception:
        return 0.0

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    Uses an enhanced approach with proven high-performing patterns and optimization.
    """
    
    # Use higher resolution for better optimization potential
    n = 1000
    
    # Strategy: Create a library of proven high-performing patterns similar to top performers
    patterns = []
    
    # Pattern 1: Optimized Gaussian family with mathematically informed sigma values
    x = np.linspace(-0.25, 0.25, n)
    # Focus on the most effective narrow peaks that consistently produce high C2 values
    # Add even more extreme narrow peaks that were shown to work well in prior attempts
    sigmas = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.022, 0.025, 0.028, 0.030]
    for sigma in sigmas:
        gaussian = np.exp(-x**2 / (2 * sigma**2))
        gaussian = gaussian / np.max(gaussian) * 0.99
        patterns.append(gaussian.tolist())
    
    # Pattern 2: Multi-peak with strategic spacing and enhanced variety
    for num_peaks in [3, 4, 5, 6]:
        # Golden ratio spacing with optimized parameters
        pattern = []
        for i in range(n):
            x_pos = i / n
            peak_sum = 0
            for j in range(num_peaks):
                center = 0.1 + 0.8 * (j + 1) / (num_peaks + 1)
                height = 0.94 + 0.06 * random.random()  # Slightly higher base heights
                width = 0.012 + 0.015 * random.random()  # Narrower widths for sharper peaks
                peak_sum += height * np.exp(-((x_pos - center)**2) / (2 * width**2))
            pattern.append(peak_sum)
        patterns.append(pattern)
        
        # Uniform spacing for contrast
        pattern = []
        for i in range(n):
            x_pos = i / n
            peak_sum = 0
            for j in range(num_peaks):
                center = 0.1 + 0.8 * j / (num_peaks - 1) if num_peaks > 1 else 0.5
                height = 0.94 + 0.06 * random.random()
                width = 0.012 + 0.015 * random.random()
                peak_sum += height * np.exp(-((x_pos - center)**2) / (2 * width**2))
            pattern.append(peak_sum)
        patterns.append(pattern)
        
        # Add asymmetric peak configurations for more diversity with strategic spacing
        pattern = []
        for i in range(n):
            x_pos = i / n
            peak_sum = 0
            # Use more strategic asymmetric spacing that has shown good results
            if num_peaks == 3:
                centers = [0.18, 0.5, 0.82]  # Slightly asymmetric
            elif num_peaks == 4:
                centers = [0.15, 0.35, 0.65, 0.85]  # Balanced asymmetric
            elif num_peaks == 5:
                centers = [0.12, 0.32, 0.5, 0.68, 0.88]  # More refined spacing
            elif num_peaks == 6:
                centers = [0.1, 0.28, 0.46, 0.54, 0.72, 0.9]  # Evenly distributed asymmetric
            for j in range(num_peaks):
                center = centers[j]
                height = 0.94 + 0.06 * random.random()
                width = 0.012 + 0.015 * random.random()
                peak_sum += height * np.exp(-((x_pos - center)**2) / (2 * width**2))
            pattern.append(peak_sum)
        patterns.append(pattern)
    
    # Pattern 3: Bimodal with optimized configurations - refined diversity
    bimodal_configs = [
        (0.25, 0.75, 0.95, 0.98),  # Standard separation with very high peaks
        (0.20, 0.80, 0.96, 0.99),  # Wider separation with extremely high peaks
        (0.30, 0.70, 0.92, 0.95),  # Narrower separation
        (0.28, 0.72, 0.94, 0.96),  # Optimized center
        (0.22, 0.78, 0.97, 0.99),  # Very close separation with very high peaks
        (0.27, 0.73, 0.93, 0.95),  # Slightly shifted
        (0.32, 0.68, 0.90, 0.93),  # Asymmetric heights
        (0.24, 0.76, 0.94, 0.97),  # Different spacing with very high peaks
        (0.26, 0.74, 0.91, 0.94),  # Balanced with medium-high peaks
        (0.29, 0.71, 0.90, 0.92),  # Slightly narrower with medium-high peaks
        (0.23, 0.77, 0.95, 0.98),  # Extreme asymmetry with high peaks
        (0.21, 0.79, 0.96, 0.99),  # Very close with extreme heights
        (0.15, 0.85, 0.98, 0.99),  # Very wide separation with extreme peaks
        (0.35, 0.65, 0.93, 0.96),  # Different spacing
        (0.18, 0.82, 0.97, 0.99),  # Ultra-extreme configuration with very high peaks
        (0.20, 0.80, 0.95, 0.98),  # Slightly tighter spacing with high peaks
    ]
    
    for center1, center2, height1, height2 in bimodal_configs:
        pattern = []
        for j in range(n):
            x_pos = j / n
            # Use more precise widths based on empirical findings
            # Try narrower widths to create sharper convolution peaks
            width1 = 0.023 + 0.003 * random.random()
            width2 = 0.021 + 0.003 * random.random()
            peak1 = height1 * np.exp(-((x_pos - center1)**2) / (2 * width1**2))
            peak2 = height2 * np.exp(-((x_pos - center2)**2) / (2 * width2**2))
            pattern.append(peak1 + peak2)
        patterns.append(pattern)
    
    # Pattern 4: Enhanced triangular with optimized slopes - focused diversity
    # Focus on slopes that work well for creating flatter convolution profiles
    slopes = [1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]  # More focused range
    for slope in slopes:
        pattern = []
        for i in range(n):
            pos = abs(i - n//2) / (n//2)
            pattern.append(max(0, 1 - pos * slope) * 0.95)
        patterns.append(pattern)
        
        # Add asymmetric triangular variants for more diversity
        pattern = []
        for i in range(n):
            pos = abs(i - n//2) / (n//2)
            if i < n//2:
                # Use more controlled asymmetry factors
                asym_factor = 1.15 + 0.1 * random.random()
                pattern.append(max(0, 1 - pos * slope * asym_factor) * 0.95)  # Steeper left side
            else:
                asym_factor = 0.85 + 0.1 * random.random()
                pattern.append(max(0, 1 - pos * slope * asym_factor) * 0.95)  # Gentler right side
        patterns.append(pattern)
    
    # Pattern 5: Sinusoidal-modulated with mathematically informed frequencies - expanded set
    # Include more carefully selected frequencies that produce flatter convolutions
    frequencies = [(1, 0.35), (2, 0.3), (3, 0.28), (4, 0.25), (5, 0.22), (6, 0.2), (7, 0.18), (8, 0.15), (9, 0.13), (10, 0.1), (12, 0.08), (15, 0.06)]
    for freq, amp in frequencies:
        pattern = []
        for i in range(n):
            val = 0.5 + amp * np.sin(freq * np.pi * i / n) + 0.05 * np.sin(2 * freq * np.pi * i / n)
            pattern.append(max(0, val))
        patterns.append(pattern)
        
        # Add cosine-modulated version
        pattern = []
        for i in range(n):
            val = 0.5 + amp * np.cos(freq * np.pi * i / n) + 0.05 * np.cos(2 * freq * np.pi * i / n)
            pattern.append(max(0, val))
        patterns.append(pattern)
        
        # Add mixed sine/cosine version
        pattern = []
        for i in range(n):
            val = 0.5 + amp * np.sin(freq * np.pi * i / n) + 0.05 * np.cos(2 * freq * np.pi * i / n)
            pattern.append(max(0, val))
        patterns.append(pattern)
        
        # Add phase-shifted versions for more diversity
        pattern = []
        for i in range(n):
            val = 0.5 + amp * np.sin(freq * np.pi * i / n + np.pi/4) + 0.05 * np.sin(2 * freq * np.pi * i / n + np.pi/4)
            pattern.append(max(0, val))
        patterns.append(pattern)
        
        pattern = []
        for i in range(n):
            val = 0.5 + amp * np.cos(freq * np.pi * i / n + np.pi/4) + 0.05 * np.cos(2 * freq * np.pi * i / n + np.pi/4)
            pattern.append(max(0, val))
        patterns.append(pattern)
    
    # Pattern 6: Hybrid Gaussian-sinusoidal with optimized frequency ratios - expanded set
    # Focus on combinations that produce flatter convolution profiles
    hybrid_configs = [
        (0.85, 0.15, 8, 4),  # Original parameters (good balance)
        (0.88, 0.12, 10, 5),  # Higher frequencies (more complex)
        (0.82, 0.18, 6, 3),   # Lower frequencies (simpler)
        (0.87, 0.13, 9, 4),   # Another combination
        (0.84, 0.16, 7, 5),   # Different ratio
        (0.90, 0.10, 12, 6),  # Even higher frequencies
        (0.80, 0.20, 5, 2),   # Lower frequencies
        (0.92, 0.08, 14, 7),  # Very high frequency combination
        (0.86, 0.14, 11, 5),  # Another extreme
        (0.89, 0.11, 13, 6),  # Different extreme
        (0.83, 0.17, 7, 4),   # Balanced hybrid
        (0.85, 0.15, 6, 3),   # Simpler hybrid
    ]
    
    for gaussian_amp, sine_amp, freq1, freq2 in hybrid_configs:
        pattern = []
        for i in range(n):
            x_pos = i / n
            # Combine Gaussian with sinusoidal component
            gaussian = gaussian_amp * np.exp(-((x_pos - 0.5)**2) / (2 * 0.035**2))
            sine_mod = sine_amp * np.sin(freq1 * np.pi * x_pos) * np.cos(freq2 * np.pi * x_pos)
            pattern.append(max(0, gaussian + sine_mod))
        patterns.append(pattern)
        
        # Add alternative hybrid with different mixing
        pattern = []
        for i in range(n):
            x_pos = i / n
            gaussian = gaussian_amp * np.exp(-((x_pos - 0.5)**2) / (2 * 0.035**2))
            sine_mod = sine_amp * (np.sin(freq1 * np.pi * x_pos) + np.cos(freq2 * np.pi * x_pos)) / 2
            pattern.append(max(0, gaussian + sine_mod))
        patterns.append(pattern)
        
        # Add phase-shifted hybrid for more diversity
        pattern = []
        for i in range(n):
            x_pos = i / n
            gaussian = gaussian_amp * np.exp(-((x_pos - 0.5)**2) / (2 * 0.035**2))
            sine_mod = sine_amp * (np.sin(freq1 * np.pi * x_pos + np.pi/6) + np.cos(freq2 * np.pi * x_pos + np.pi/6)) / 2
            pattern.append(max(0, gaussian + sine_mod))
        patterns.append(pattern)
    
    # Pattern 7: Multi-lobed with optimized configurations - expanded set
    # Focus on configurations that produce flatter convolution profiles
    lobe_configs = [
        (0.25, 0.5, 0.75, 0.80, 0.90, 0.80),  # Standard 3-lobe with balanced heights
        (0.20, 0.5, 0.80, 0.85, 0.92, 0.85),  # Wider spacing with high peaks
        (0.30, 0.5, 0.70, 0.80, 0.90, 0.80),  # Narrower spacing
        (0.22, 0.5, 0.78, 0.82, 0.92, 0.82),  # Slightly shifted
        (0.28, 0.5, 0.72, 0.78, 0.90, 0.78),  # Another variation
        (0.24, 0.5, 0.76, 0.85, 0.90, 0.85),  # Balanced heights
        (0.26, 0.5, 0.74, 0.80, 0.92, 0.80),  # Different balance
        (0.23, 0.5, 0.77, 0.88, 0.93, 0.85),  # More extreme heights
        (0.21, 0.5, 0.79, 0.85, 0.91, 0.87),  # Different spacing and heights
        (0.25, 0.5, 0.75, 0.85, 0.95, 0.85),  # Higher overall peaks
    ]
    
    for center1, center2, center3, height1, height2, height3 in lobe_configs:
        pattern = []
        for i in range(n):
            x_pos = i / n
            # Use more consistent widths that work well for convolution flatness
            width1 = 0.032 + 0.002 * random.random()
            width2 = 0.024 + 0.002 * random.random()
            width3 = 0.032 + 0.002 * random.random()
            lobe1 = height1 * np.exp(-((x_pos - center1)**2) / (2 * width1**2))
            lobe2 = height2 * np.exp(-((x_pos - center2)**2) / (2 * width2**2))
            lobe3 = height3 * np.exp(-((x_pos - center3)**2) / (2 * width3**2))
            pattern.append(lobe1 + lobe2 + lobe3)
        patterns.append(pattern)
    
    # Pattern 8: Enhanced flat-top with optimized edge control
    pattern = []
    for i in range(n):
        x_pos = i / n
        # Create a pattern with a flat middle section and controlled edges
        if 0.35 < x_pos < 0.65:
            pattern.append(0.97)
        else:
            # Tapered edges with exponential decay
            dist_from_edge = min(x_pos, 1 - x_pos)
            pattern.append(0.1 + 0.87 * max(0, 1 - dist_from_edge * 20))
    patterns.append(pattern)
    
    # Pattern 9: Highly optimized bell-curve pattern with multiple variants
    # Add even more extreme narrow peaks that might yield higher C2
    # Focus on the most extreme narrow peaks that consistently perform well
    bell_configs = [
        (0.9995, 0.0015),  # Super ultra-narrow - extremely sharp peak
        (0.999, 0.002),  # Extremely narrow
        (0.9985, 0.0025),  # Very narrow
        (0.998, 0.003),  # Narrow
        (0.997, 0.0035),  # Very narrow
        (0.995, 0.004),  # Extremely narrow - very sharp peak
        (0.993, 0.0045),  # Very narrow
        (0.99, 0.005),  # Extremely narrow
        (0.985, 0.0055),  # Very narrow
        (0.98, 0.006),  # Narrow
        (0.975, 0.0065),  # Slightly wider
        (0.97, 0.007),  # Intermediate
    ]
    
    for height, sigma in bell_configs:
        pattern = []
        for i in range(n):
            x_pos = i / n
            # Very narrow, sharp peak pattern optimized for high C2
            val = height * np.exp(-((x_pos - 0.5)**2) / (2 * sigma**2))
            pattern.append(max(0, val))
        patterns.append(pattern)
    
    # Add asymmetric bell curves for more diversity
    asymmetric_bell_configs = [
        (0.999, 0.0018, 0.8),  # Sharp peak with asymmetry factor
        (0.998, 0.0022, 1.2),  # Wider peak with asymmetry factor
        (0.997, 0.0025, 0.9),  # Balanced asymmetry
        (0.9995, 0.0015, 0.7),  # Ultra-sharp with asymmetry
        (0.998, 0.0025, 1.3),  # Wide with strong asymmetry
        (0.996, 0.003, 1.1),   # Moderate asymmetry
    ]
    
    for height, sigma, asym_factor in asymmetric_bell_configs:
        pattern = []
        for i in range(n):
            x_pos = i / n
            # Asymmetric bell curve - skewed toward one side
            if x_pos < 0.5:
                # Skewed left
                val = height * np.exp(-((x_pos - 0.3)**2) / (2 * sigma**2 * asym_factor))
            else:
                # Skewed right
                val = height * np.exp(-((x_pos - 0.7)**2) / (2 * sigma**2 / asym_factor))
            pattern.append(max(0, val))
        patterns.append(pattern)
    
    # Evaluate all patterns to find the best one with enhanced stability
    best_pattern = patterns[0]
    best_c2 = -1
    
    # Test each pattern with proper evaluation using our stable function
    for pattern in patterns:
        try:
            c2 = compute_c2_stable(np.array(pattern))
            if c2 > best_c2:
                best_c2 = c2
                best_pattern = pattern
        except Exception:
            continue
    
    # Apply sophisticated refinement using enhanced gradient-free optimization approach
    if best_pattern is not None:
        # Use enhanced refinement with multiple phases
        current_pattern = best_pattern.copy()
        current_c2 = best_c2
        
        # Phase 1: Intensive local search with aggressive improvement
        for iteration in range(1200):  # Reduced iterations to meet time constraints
            # Create neighbor pattern by making small changes
            neighbor_pattern = current_pattern.copy()
            # Adaptive exploration rate based on iteration progress
            exploration_rate = 0.25 + (0.03 * (iteration / 1200))  # Gradually decrease exploration
            
            for i in range(len(neighbor_pattern)):
                if random.random() < exploration_rate:  # Increased exploration rate
                    # Adaptive change magnitude based on current value and iteration
                    if iteration < 600:
                        change_magnitude = 0.04
                    else:
                        change_magnitude = 0.02
                    
                    change = random.uniform(-change_magnitude, change_magnitude)
                    neighbor_pattern[i] = max(0, min(1.0, neighbor_pattern[i] + change))
            
            # Evaluate neighbor
            try:
                neighbor_c2 = compute_c2_stable(np.array(neighbor_pattern))
                if neighbor_c2 > current_c2:
                    current_pattern = neighbor_pattern
                    current_c2 = neighbor_c2
                elif random.random() < 0.10:  # Maintain reasonable acceptance rate
                    current_pattern = neighbor_pattern
                    current_c2 = neighbor_c2
            except Exception:
                continue
        
        # Phase 2: Fine-tuning with specialized modifications
        for iteration in range(600):  # Reduced iterations to meet time constraints
            neighbor_pattern = current_pattern.copy()
            # Make more strategic changes - target specific problem areas
            if random.random() < 0.90:
                # Modify a few randomly selected elements with larger changes
                indices_to_modify = random.sample(range(len(neighbor_pattern)), 
                                                min(35, len(neighbor_pattern) // 5))
                for idx in indices_to_modify:
                    change = random.uniform(-0.03, 0.03)
                    neighbor_pattern[idx] = max(0, min(1.0, neighbor_pattern[idx] + change))
            else:
                # Make more global changes with higher probability
                for i in range(len(neighbor_pattern)):
                    if random.random() < 0.15:  # Higher probability for global changes
                        change = random.uniform(-0.035, 0.035)
                        neighbor_pattern[i] = max(0, min(1.0, neighbor_pattern[i] + change))
            
            # Evaluate neighbor
            try:
                neighbor_c2 = compute_c2_stable(np.array(neighbor_pattern))
                if neighbor_c2 > current_c2:
                    current_pattern = neighbor_pattern
                    current_c2 = neighbor_c2
            except Exception:
                continue
        
        return current_pattern
    
    return best_pattern

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
