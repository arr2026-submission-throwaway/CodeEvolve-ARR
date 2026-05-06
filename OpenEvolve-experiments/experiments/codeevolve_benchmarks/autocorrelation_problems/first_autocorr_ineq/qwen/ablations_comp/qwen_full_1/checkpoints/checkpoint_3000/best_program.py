# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import convolve
import random

def compute_c1(sequence):
    """Compute C1 value for a given sequence."""
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return float('inf')
    
    n = len(sequence)
    # Use FFT-based convolution for better performance on large sequences
    if n > 100:
        conv_result = np.convolve(sequence, sequence, mode='full')
        max_conv = np.max(conv_result)
    else:
        # For small sequences, direct computation might be faster
        conv_result = np.convolve(sequence, sequence, mode='full')
        max_conv = np.max(conv_result)
    
    sum_sq = sum(sequence) ** 2
    
    if sum_sq == 0:
        return float('inf')
    
    # Add numerical stability check
    if max_conv < 1e-12:
        return float('inf')
    
    return 2 * n * max_conv / sum_sq

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using more sophisticated local search."""
    n = len(sequence)
    if n == 0:
        return None
        
    sum_sequence = sum(sequence)
    if sum_sequence < 0.01:
        return None
    
    # Try multiple local search strategies to find better directions
    best_sequence = sequence.copy()
    best_c1 = compute_c1(sequence)
    
    # Strategy 1: Enhanced random perturbations with smarter sampling and adaptive magnitudes
    for _ in range(100):  # More iterations for better exploration
        test_sequence = sequence.copy()
        # Focus on elements that are more likely to impact convolution peaks
        if n > 30:
            # Sample from critical regions: ends and center where convolution behavior is most sensitive
            critical_indices = []
            if n > 10:
                critical_indices.extend(range(min(5, n//4)))  # First few elements
                critical_indices.extend(range(max(0, n-5), n))  # Last few elements
            if n > 20:
                critical_indices.extend(range(n//2-2, n//2+3))  # Center region
            
            # Remove duplicates and sample appropriately
            critical_indices = list(set(critical_indices))
            num_perturb = max(1, min(len(critical_indices), len(critical_indices) // 2))
            indices = random.sample(critical_indices, num_perturb)
        else:
            # For smaller sequences, sample more broadly but with higher magnitude
            num_perturb = max(1, n // 4)
            indices = random.sample(range(n), num_perturb)
            
        for idx in indices:
            # Add adaptive perturbation based on local context
            base_magnitude = min(2.0, sequence[idx] * 0.8)
            # Use larger perturbations for elements that are likely to be more influential
            if idx < n//4 or idx >= 3*n//4:  # End regions
                factor = random.uniform(-base_magnitude * 1.2, base_magnitude * 1.2)
            else:  # Central regions
                factor = random.uniform(-base_magnitude, base_magnitude)
            test_sequence[idx] = max(0, test_sequence[idx] + factor)
        
        # Check if this improves our result
        test_c1 = compute_c1(test_sequence)
        if test_c1 < best_c1 and test_c1 != float('inf'):
            best_c1 = test_c1
            best_sequence = test_sequence
    
    # Strategy 1b: Enhanced perturbation focused on critical regions (ends and center)
    if n > 30:
        try:
            test_sequence = sequence.copy()
            # Focus on end regions and center where convolution behavior is most critical
            critical_indices = []
            if n > 10:
                critical_indices.extend(range(min(5, n//4)))  # First few elements
                critical_indices.extend(range(max(0, n-5), n))  # Last few elements
            if n > 20:
                critical_indices.extend(range(n//2-2, n//2+3))  # Center region
            
            # Remove duplicates and sample appropriately
            critical_indices = list(set(critical_indices))
            num_perturb = max(1, min(len(critical_indices), len(critical_indices) // 2))
            indices = random.sample(critical_indices, num_perturb)
            
            for idx in indices:
                # Add larger perturbations to critical elements
                base_magnitude = min(1.5, sequence[idx] * 0.8)
                perturbation = random.uniform(-base_magnitude, base_magnitude)
                test_sequence[idx] = max(0, test_sequence[idx] + perturbation)
            
            test_c1 = compute_c1(test_sequence)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = test_sequence
        except Exception:
            pass
    
    # Strategy 2: Enhanced smoothing with more focused kernel designs
    try:
        # Apply focused smoothing techniques that have proven effective
        if n > 5:
            # Approach 1: Gaussian smoothing with optimized width
            smoothed = np.array(sequence)
            kernel_width = min(12, max(3, n // 4))
            kernel = np.exp(-np.arange(-kernel_width, kernel_width+1)**2 / (2 * (kernel_width/2.5)**2))
            kernel = kernel / np.sum(kernel)
            
            # Apply convolution with proper padding
            padded = np.pad(smoothed, kernel_width, mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            
            # Ensure non-negativity and normalize
            smoothed = np.maximum(0, smoothed[:n])
            total_old = sum(sequence)
            total_new = sum(smoothed)
            if total_new > 0:
                smoothed = smoothed * total_old / total_new
            
            # Clip to reasonable bounds
            smoothed = np.clip(smoothed, 0, 1000)
            
            test_c1 = compute_c1(smoothed.tolist())
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = smoothed.tolist()
                
        # Approach 2: Try sharper kernel for more aggressive peak suppression
        if n > 20:
            # Try a sharper kernel for more aggressive peak reduction
            kernel = np.array([0.001, 0.005, 0.02, 0.07, 0.15, 0.25, 0.25, 0.15, 0.07, 0.02, 0.005, 0.001])
            # Pad and apply convolution
            pad_width = len(kernel) // 2
            padded = [0] * pad_width + sequence + [0] * pad_width
            sharper_smoothed = np.convolve(padded, kernel, mode='valid')
            # Ensure non-negativity and normalize
            sharper_smoothed = np.maximum(0, sharper_smoothed)
            total_old = sum(sequence)
            total_new = sum(sharper_smoothed)
            if total_new > 0:
                sharper_smoothed = [x * total_old / total_new for x in sharper_smoothed]
            # Clip to bounds
            sharper_smoothed = [max(0, min(1000, x)) for x in sharper_smoothed]
            
            test_c1 = compute_c1(sharper_smoothed)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = sharper_smoothed
    except Exception:
        pass
    
    # Strategy 3: Enhanced exponential decay testing with strategic base selection
    if n > 10:
        # Focus on bases that have consistently performed well in previous experiments
        # Include both aggressive and moderate decay rates
        bases = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
        for base in bases:
            exp_pattern = [base**i for i in range(n)]
            total_exp = sum(exp_pattern)
            if total_exp > 0:
                exp_scaled = [x * sum_sequence / total_exp for x in exp_pattern]
                test_c1 = compute_c1(exp_scaled)
                if test_c1 < best_c1 and test_c1 != float('inf'):
                    best_c1 = test_c1
                    best_sequence = exp_scaled
    
    # Strategy 4: Enhanced power-law decay testing with focused alpha values
    if n > 10:
        try:
            # Focus on alpha values that have shown consistent effectiveness
            # These are based on empirical evidence from previous optimizations
            alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]
            for alpha in alphas:
                power_pattern = [1.0 / ((i+1)**alpha) for i in range(n)]
                total_power = sum(power_pattern)
                if total_power > 0:
                    power_scaled = [x * sum_sequence / total_power for x in power_pattern]
                    test_c1 = compute_c1(power_scaled)
                    if test_c1 < best_c1 and test_c1 != float('inf'):
                        best_c1 = test_c1
                        best_sequence = power_scaled
        except Exception:
            pass
    
    # Strategy 5: Enhanced hybrid approach with proven blend combinations
    if n > 20:
        # Try focused hybrid combinations that have shown effectiveness
        hybrid_patterns = []
        
        # Pattern 1: Standard geometric-linear blend with proven weights
        hybrid1 = []
        for i in range(n):
            geo_val = 0.9**i
            linear_val = 1.0 - i/(n-1) if n > 1 else 0
            hybrid1.append(0.2 * geo_val + 0.7 * linear_val + 0.1 * (0.95**i))
        hybrid_patterns.append(('geo_linear_1', hybrid1))
        
        # Pattern 2: Alternative blend with different emphasis
        hybrid2 = []
        for i in range(n):
            geo_val = 0.92**i
            linear_val = 1.0 - i/(n-1) if n > 1 else 0
            hybrid2.append(0.15 * geo_val + 0.75 * linear_val + 0.1 * (0.98**i))
        hybrid_patterns.append(('geo_linear_2', hybrid2))
        
        # Pattern 3: More geometric emphasis
        hybrid3 = []
        for i in range(n):
            geo_val = 0.95**i
            linear_val = 1.0 - i/(n-1) if n > 1 else 0
            hybrid3.append(0.3 * geo_val + 0.6 * linear_val + 0.1 * (0.9**i))
        hybrid_patterns.append(('geo_linear_3', hybrid3))
        
        # Evaluate hybrid patterns
        for name, pattern in hybrid_patterns:
            total_hybrid = sum(pattern)
            if total_hybrid > 0:
                hybrid_scaled = [x * sum_sequence / total_hybrid for x in pattern]
                test_c1 = compute_c1(hybrid_scaled)
                if test_c1 < best_c1 and test_c1 != float('inf'):
                    best_c1 = test_c1
                    best_sequence = hybrid_scaled
    
    # Strategy 6: Enhanced sinusoidal modulation with strategic parameters
    if n > 30:
        try:
            # Try focused sinusoidal modulation with proven amplitude/frequency combinations
            # These have shown effectiveness in reducing convolution peaks
            modulations = [
                (0.2, n//6),   # Moderate amplitude, medium frequency
                (0.25, n//8),  # Slightly larger amplitude, lower frequency
                (0.15, n//10), # Smaller amplitude, higher frequency
                (0.3, n//12),  # Larger amplitude, lowest frequency
            ]
            
            for amp, freq in modulations:
                if freq > 0:
                    modulated = [sequence[i] * (1 + amp * np.sin(i * 2 * np.pi / freq)) for i in range(n)]
                    modulated = [max(0, x) for x in modulated]
                    total_mod = sum(modulated)
                    if total_mod > 0:
                        modulated = [x * sum_sequence / total_mod for x in modulated]
                        test_c1 = compute_c1(modulated)
                        if test_c1 < best_c1 and test_c1 != float('inf'):
                            best_c1 = test_c1
                            best_sequence = modulated
        except Exception:
            pass
    
    # Strategy 7: Focused peak reduction with proven kernel designs
    try:
        if n > 15:
            # Approach 1: Standard Gaussian smoothing (most reliable)
            kernel_width = min(12, max(3, n // 4))
            kernel = np.exp(-np.arange(-kernel_width, kernel_width+1)**2 / (2 * (kernel_width/2.5)**2))
            kernel = kernel / np.sum(kernel)
            
            # Apply convolution with padding
            padded = np.pad(sequence, kernel_width, mode='edge')
            reduced_peak = np.convolve(padded, kernel, mode='valid')
            
            # Ensure non-negativity and normalize
            reduced_peak = np.maximum(0, reduced_peak[:n])
            total_old = sum(sequence)
            total_new = sum(reduced_peak)
            if total_new > 0:
                reduced_peak = [x * total_old / total_new for x in reduced_peak]
            
            test_c1 = compute_c1(reduced_peak)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = reduced_peak
                
    except Exception:
        pass
    
    return best_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None
        
    # Instead of complex LP solving, use a more effective heuristic
    # Try to construct a sequence that minimizes the maximum convolution value
    # One effective approach: try to make the convolution as flat as possible
    
    # Create a simple, effective sequence based on known good patterns
    # Start with a simple pattern that has been shown to work well
    try:
        # Use a pattern that balances high values with low convolution peaks
        # Try a sequence with a few large values and many small ones
        new_seq = np.array(f_sequence)
        
        # Apply a simple smoothing operation to reduce convolution peaks
        # This is a more principled approach than pure random noise
        smoothed = np.convolve(new_seq, np.ones(3)/3, mode='same')
        
        # Add some random variation but keep it controlled
        noise = np.random.normal(0, 0.005, n)
        new_seq = smoothed + noise
        
        # Ensure non-negativity and normalize
        new_seq = np.maximum(new_seq, 0)
        if np.sum(new_seq) > 1e-10:
            new_seq = new_seq * np.sum(f_sequence) / np.sum(new_seq)
        
        return new_seq.tolist()
        
    except Exception:
        # Fallback to a simple modification
        try:
            # Try to create a better sequence by modifying existing one
            # Use a Gaussian-like pattern which tends to work well
            indices = np.arange(n)
            # Create a bell-shaped sequence
            sigma = n / 6.0  # Adjust width based on sequence length
            gaussian = np.exp(-0.5 * ((indices - (n-1)/2) / sigma)**2)
            # Scale to maintain roughly same sum
            scaled_gaussian = gaussian * np.sum(f_sequence) / np.sum(gaussian)
            return scaled_gaussian.tolist()
        except:
            return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Try multiple random initializations to find good starting points
    best_sequence = None
    best_c1 = float('inf')
    
    # Try different initialization strategies with focus on proven effective patterns
    init_strategies = [
        # Uniform distribution
        lambda n: [1.0] * n,
        # Geometric decay (focused on effective bases)
        lambda n: [0.9**i for i in range(n)],
        # Linear decay
        lambda n: [max(0, 1.0 - i/(n-1)) for i in range(n)],
        # Spike at beginning
        lambda n: [1.0 if i == 0 else 0.0 for i in range(n)],
        # Random with variance control
        lambda n: [random.uniform(0.1, 15.0) for _ in range(n)],
        # Power law decay (focused on effective alphas)
        lambda n: [1.0 / ((i+1)**0.6) for i in range(n)],
        # Gaussian-like pattern
        lambda n: [np.exp(-((i - n/2)**2) / (2 * (n/6)**2)) for i in range(n)],
        # Logarithmic decay
        lambda n: [1.0/np.log(i+2) for i in range(n)] if n > 1 else [1.0] * n,
        # Sine wave pattern
        lambda n: [0.5 * (1 + np.sin(i * 2 * np.pi / (n//4))) for i in range(n)],
        # Inverted exponential
        lambda n: [0.95**(n-1-i) for i in range(n)],
        # Step function pattern
        lambda n: [1.0 if i < n//2 else 0.3 for i in range(n)],
        # Parabolic decay
        lambda n: [1.0 - (i/(n-1))**2 for i in range(n)],
        # Exponential with base 0.98
        lambda n: [0.98**i for i in range(n)],
        # Heavy-tailed power law
        lambda n: [1.0 / ((i+1)**1.3) for i in range(n)],
        # Modified exponential decay
        lambda n: [0.92**i for i in range(n)],
        # Fast decay exponential
        lambda n: [0.85**i for i in range(n)],
        # Concave pattern
        lambda n: [(n-i)/n for i in range(n)],
        # Anti-correlated pattern
        lambda n: [1.0 if i % 3 == 0 else 0.2 if i % 3 == 1 else 0.8 for i in range(n)],
    ]
    
    for attempt in range(35):  # Reduce attempts slightly to increase speed
        # Use more focused sequence length ranges to prioritize efficiency
        rand_val = random.random()
        if rand_val < 0.4:
            n = random.randint(20, 400)  # Emphasis on smaller ranges for efficiency
        elif rand_val < 0.7:
            n = random.randint(400, 800)  # Medium sequences
        else:
            n = random.randint(800, 1500)  # Larger sequences for breakthroughs
        
        # Choose initialization strategy
        strategy_idx = attempt % len(init_strategies)
        sequence = init_strategies[strategy_idx](n)
        
        # Apply optimization multiple times for better convergence
        current_sequence = sequence.copy()
        # Dynamic optimization steps based on sequence size
        num_steps = max(8, min(20, n // 25))  # Fewer steps for larger sequences
        for _ in range(num_steps):
            h_function = get_good_direction_to_move_into(current_sequence)
            if h_function is not None:
                current_sequence = h_function
            else:
                break
                
        # Evaluate
        c1 = compute_c1(current_sequence)
        if c1 < best_c1 and c1 != float('inf'):
            best_c1 = c1
            best_sequence = current_sequence.copy()
    
    # If no good sequence found, start with a better baseline
    if best_sequence is None:
        # Try a more carefully constructed sequence
        n = 350
        # Use a combination that often works well: geometric with slight modification
        best_sequence = [0.92**i * (1.0 + 0.015 * random.random()) for i in range(n)]
        # Normalize to have reasonable sum
        s = sum(best_sequence)
        if s > 0:
            best_sequence = [x/s * 70 for x in best_sequence]
    
    # Final refinement with fewer passes to preserve time
    for _ in range(15):  # Fewer refinement passes to save time
        refined = get_good_direction_to_move_into(best_sequence)
        if refined is not None:
            c1_refined = compute_c1(refined)
            if c1_refined < best_c1 and c1_refined != float('inf'):
                best_c1 = c1_refined
                best_sequence = refined
        else:
            break
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
