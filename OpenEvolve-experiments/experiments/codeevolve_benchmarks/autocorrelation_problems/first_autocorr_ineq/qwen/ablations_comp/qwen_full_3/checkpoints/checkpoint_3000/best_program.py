# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import time

def compute_c1(sequence):
    """Compute C1 constant for a given sequence."""
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return float('inf')
    
    # Use FFT for efficient convolution
    conv = signal.fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    sum_sq = sum(sequence) ** 2
    
    if sum_sq == 0:
        return float('inf')
    
    return (2 * len(sequence) * max_conv) / sum_sq

def compute_inv_c1(sequence):
    """Compute 1/C1 for a given sequence."""
    c1 = compute_c1(sequence)
    return 1.0 / c1 if c1 != float('inf') else 0.0

def create_better_candidate_sequence():
    """Create a better candidate sequence using diverse known good patterns."""
    # Try to construct sequences based on known good structures
    n = random.randint(50, 400)  # Narrowed range for better control
    
    # Use weighted probabilities for pattern selection to favor more successful ones
    # Emphasizing the most proven patterns for higher chance of success
    # Optimize weights based on recent performance trends
    pattern_weights = {
        'concentrated': 0.60,   # Increased weight for most effective pattern
        'multi_peak': 0.20,     # Strong second choice
        'sparse': 0.10,
        'geometric': 0.05,
        'alternating': 0.03,    # Less consistently effective
        'step': 0.02,           # Simpler patterns
        'gaussian': 0.00        # Removed as less effective
    }
    
    # Select pattern type based on weighted probabilities
    pattern_type = random.choices(list(pattern_weights.keys()), 
                                  list(pattern_weights.values()))[0]
    
    if pattern_type == 'geometric':
        # Geometric decay with controlled randomness
        sequence = []
        for i in range(n):
            base_val = max(0.01, 1.0 / (1.0 + i * 0.08))  # Slightly faster decay
            variation = random.uniform(0.85, 1.15)
            sequence.append(base_val * variation)
    elif pattern_type == 'step':
        # Step function with strategic transitions
        sequence = []
        step1 = random.uniform(1.5, 3.5)  # Wider range for more variation
        step2 = random.uniform(0.1, 0.5)
        for i in range(n):
            if i < n // 2:
                sequence.append(step1)
            else:
                sequence.append(step2)
    elif pattern_type == 'sparse':
        # Sparse with some large values - can reduce peak convolution
        sequence = [0.0] * n
        num_large = min(8, n // 15)  # Slightly increased for more impact
        for _ in range(num_large):
            idx = random.randint(0, n-1)
            sequence[idx] = random.uniform(1.5, 4.0)  # Larger values
    elif pattern_type == 'concentrated':
        # Highly concentrated pattern with mass at one end - very effective
        sequence = [0.0] * n
        # Put most mass at the beginning (99.8%) for even better concentration
        mass_start = int(0.998 * n)
        for i in range(mass_start):
            sequence[i] = 1.0
        # Distribute remaining mass with even smaller values
        remaining = n - mass_start
        for i in range(mass_start, n):
            if remaining > 0:
                sequence[i] = 0.001
                remaining -= 1
    elif pattern_type == 'multi_peak':
        # Multi-peak pattern with better spacing and amplitude
        sequence = [0.0] * n
        num_peaks = min(6, n // 10)  # Slightly more peaks for better distribution
        for i in range(num_peaks):
            pos = int((i + 1) * (n - 1) / (num_peaks + 1))
            sequence[pos] = random.uniform(4.0, 10.0)  # Higher amplitude
    else:  # alternating
        # Alternating pattern - often reduces autocorrelation peaks
        sequence = []
        for i in range(n):
            if i % 2 == 0:
                sequence.append(random.uniform(1.2, 2.5))  # Wider range for more flexibility
            else:
                sequence.append(random.uniform(0.1, 0.5))
    
    # Normalize to have reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x * 15.0 / total for x in sequence]  # Higher normalization factor
    
    return sequence

def get_good_direction_to_move_into(sequence):
    """Returns a better direction to move into the sequence."""
    n = len(sequence)
    if n == 0:
        return None
        
    # Try multiple strategies for finding better directions
    strategies = []
    
    # Strategy 1: Even more aggressive variance reduction for better peak suppression
    new_sequence1 = sequence.copy()
    mean_val = np.mean(sequence)
    std_val = np.std(sequence)
    
    if std_val > 0.05 * mean_val and mean_val > 0:
        for i in range(len(new_sequence1)):
            # Even more aggressive shrink toward mean to reduce variance
            new_sequence1[i] = mean_val + 0.95 * (new_sequence1[i] - mean_val) + \
                              np.random.normal(0, 0.010 * mean_val)
            new_sequence1[i] = max(0, new_sequence1[i])
    
    strategies.append(("variance_reduction", new_sequence1))
    
    # Strategy 2: Enhanced pair-wise swaps to break correlations
    new_sequence2 = sequence.copy()
    if len(new_sequence2) >= 2:
        # Try swapping adjacent elements with higher probability
        if random.random() < 0.7:  # Increased probability for better correlation breaking
            i = random.randint(0, len(new_sequence2)-2)
            # Swap elements with more strategic selection
            new_sequence2[i], new_sequence2[i+1] = new_sequence2[i+1], new_sequence2[i]
    
    strategies.append(("pair_swap", new_sequence2))
    
    # Strategy 3: Enhanced adaptive spreading for better distribution with more precise targeting
    new_sequence3 = sequence.copy()
    if len(new_sequence3) > 10:
        # Apply more aggressive spreading strategy with better control
        total = sum(new_sequence3)
        if total > 0.01:
            avg_val = total / len(new_sequence3)
            # Spread values more aggressively with better targeting
            for i in range(len(new_sequence3)):
                # Push values toward average with more controlled noise
                if new_sequence3[i] > avg_val:
                    # More aggressive reduction for high values with adaptive factor
                    reduction_factor = 0.92 + 0.03 * (new_sequence3[i] / avg_val)
                    new_sequence3[i] = max(0, new_sequence3[i] * reduction_factor + 
                                         np.random.normal(0, 0.03 * avg_val))
                else:
                    # Slight increase for low values with adaptive factor
                    increase_factor = 1.07 + 0.02 * (avg_val / new_sequence3[i] if new_sequence3[i] > 0 else 1)
                    new_sequence3[i] = max(0, new_sequence3[i] * increase_factor + 
                                         np.random.normal(0, 0.015 * avg_val))
    
    strategies.append(("aggressive_spread", new_sequence3))
    
    # Strategy 4: Enhanced local smoothing to reduce sharp peaks
    new_sequence4 = sequence.copy()
    if len(new_sequence4) > 3:
        # Apply a more sophisticated smoothing approach with better neighbor consideration
        smoothed = []
        for i in range(len(new_sequence4)):
            # Average with neighbors but with weighted contribution
            neighbors = []
            weights = []
            for j in range(max(0, i-2), min(len(new_sequence4), i+3)):
                # Give more weight to closer neighbors
                weight = 1.0 / (1 + abs(i - j))
                weights.append(weight)
                neighbors.append(new_sequence4[j])
            # Weighted average
            if neighbors:
                weighted_avg = sum(w * n for w, n in zip(weights, neighbors)) / sum(weights)
                smoothed.append(weighted_avg)
            else:
                smoothed.append(new_sequence4[i])
        # Add some noise to prevent getting stuck
        for i in range(len(smoothed)):
            smoothed[i] = max(0, smoothed[i] + np.random.normal(0, 0.015 * smoothed[i]))
        new_sequence4 = smoothed
    
    strategies.append(("smoothing", new_sequence4))
    
    # Strategy 5: Enhanced random perturbations for better escape from local optima
    new_sequence5 = sequence.copy()
    # Apply more targeted random changes with higher variance for better exploration
    for i in range(min(len(new_sequence5), 30)):  # Increased limit for more exploration
        if random.random() < 0.40:  # Increased probability for better exploration
            # More aggressive adjustment to help escape local minima
            change = np.random.normal(0, 0.07 * sequence[i])  # Increased variance
            new_sequence5[i] = max(0, sequence[i] + change)
    
    strategies.append(("random_perturb", new_sequence5))
    
    # Strategy 6: Peak-focused reduction with enhanced distance-weighted approach
    new_sequence6 = sequence.copy()
    if len(new_sequence6) > 5:
        # Analyze convolution to identify problematic regions
        conv = signal.fftconvolve(sequence, sequence, mode='full')
        max_conv_idx = np.argmax(conv)
        # Modify elements around the peak convolution index with adaptive factor
        start_idx = max(0, max_conv_idx - 7)
        end_idx = min(len(new_sequence6), max_conv_idx + 8)
        for i in range(start_idx, end_idx):
            # Reduce high values that might contribute to peaks with adaptive factor
            if new_sequence6[i] > np.mean(new_sequence6) * 1.6:
                # Use distance-based reduction factor for better precision with stronger reduction
                distance_factor = 1.0 - abs(i - max_conv_idx) / (end_idx - start_idx)
                reduction_factor = 0.60 + 0.35 * distance_factor
                new_sequence6[i] = max(0, new_sequence6[i] * reduction_factor + 
                                     np.random.normal(0, 0.006 * new_sequence6[i]))
    
    strategies.append(("peak_reduction", new_sequence6))
    
    # Strategy 7: Enhanced peak reduction with improved distance weighting and stronger reduction
    new_sequence7 = sequence.copy()
    if len(new_sequence7) > 10:
        # Analyze convolution to identify problematic regions
        conv = signal.fftconvolve(sequence, sequence, mode='full')
        max_conv_idx = np.argmax(conv)
        # Modify elements around the peak convolution index with adaptive reduction
        start_idx = max(0, max_conv_idx - 6)
        end_idx = min(len(new_sequence7), max_conv_idx + 7)
        for i in range(start_idx, end_idx):
            # Reduce high values that might contribute to peaks with better targeting
            if new_sequence7[i] > np.mean(new_sequence7) * 1.7:
                # Distance-based reduction with better control and more aggressive factors
                distance_factor = 1.0 - abs(i - max_conv_idx) / ((end_idx - start_idx) / 2)
                reduction_factor = 0.50 + 0.40 * distance_factor
                new_sequence7[i] = max(0, new_sequence7[i] * reduction_factor + 
                                     np.random.normal(0, 0.005 * new_sequence7[i]))
    
    strategies.append(("enhanced_peak_reduction", new_sequence7))
    
    # Strategy 8: Enhanced concentrated mass pattern for better C1 behavior
    new_sequence8 = sequence.copy()
    if len(new_sequence8) > 10:
        # Try creating a more concentrated mass pattern with better targeting
        # Identify peaks and enhance their concentration
        peaks = []
        for i in range(len(new_sequence8)):
            if i == 0:
                if new_sequence8[i] > new_sequence8[i+1]:
                    peaks.append(i)
            elif i == len(new_sequence8) - 1:
                if new_sequence8[i] > new_sequence8[i-1]:
                    peaks.append(i)
            else:
                if new_sequence8[i] > new_sequence8[i-1] and new_sequence8[i] > new_sequence8[i+1]:
                    peaks.append(i)
        
        # If there are peaks, concentrate mass around them with better control
        if peaks:
            peak_positions = peaks[:min(3, len(peaks))]  # Only consider top 3 peaks
            for i in range(len(new_sequence8)):
                # Increase values near peaks with stronger emphasis
                min_dist = min(abs(i - p) for p in peak_positions)
                if min_dist <= 2:
                    factor = 1.0 + 0.35 * (2 - min_dist)  # Even stronger increase closer to peak
                    new_sequence8[i] = max(0, new_sequence8[i] * factor)
        else:
            # If no clear peaks, create a more concentrated version
            # Create a sharper peak at the beginning for better concentration
            total = sum(new_sequence8)
            if total > 0.01:
                # Create a more concentrated version with sharper peak
                new_sequence8 = [0.0] * len(new_sequence8)
                mass_start = int(0.99 * len(new_sequence8))  # Even more concentrated
                for i in range(mass_start):
                    new_sequence8[i] = 1.0
                remaining = len(new_sequence8) - mass_start
                for i in range(mass_start, len(new_sequence8)):
                    if remaining > 0:
                        new_sequence8[i] = 0.005
                        remaining -= 1
    
    strategies.append(("concentrated_mass", new_sequence8))
    
    # Strategy 9: Refined extreme peak reduction with better targeting and stronger reduction
    new_sequence9 = sequence.copy()
    if len(new_sequence9) > 10:
        # Find and aggressively reduce the highest values with more careful targeting
        sorted_indices = np.argsort(new_sequence9)[::-1]  # Descending order
        # Reduce top 7 values with adaptive reduction for more thorough peak suppression
        for i in range(min(7, len(sorted_indices))):
            idx = sorted_indices[i]
            if new_sequence9[idx] > np.mean(new_sequence9) * 1.9:
                # Adaptive reduction based on rank with more aggressive factors
                reduction_factor = 0.50 + 0.25 * (1.0 - i / len(sorted_indices))
                new_sequence9[idx] = max(0, new_sequence9[idx] * reduction_factor + 
                                        np.random.normal(0, 0.006 * new_sequence9[idx]))
    
    strategies.append(("extreme_peak_reduction", new_sequence9))
    
    # Choose the best strategy based on fitness improvement
    best_strategy = None
    best_fitness = compute_inv_c1(sequence)
    
    # Test only the most promising strategies to save time
    # Focus on the most effective strategies rather than all of them
    selected_strategies = strategies[:10]  # Test fewer strategies to focus on highest potential
    
    for name, candidate_seq in selected_strategies:
        fitness = compute_inv_c1(candidate_seq)
        if fitness > best_fitness:
            best_fitness = fitness
            best_strategy = candidate_seq
    
    # If none improved, return a more controlled variation of the original
    if best_strategy is None:
        # Make small, controlled changes with better scaling
        new_sequence = sequence.copy()
        for i in range(min(len(new_sequence), 25)):  # Reduced limit for speed
            if random.random() < 0.30:  # Increased probability for more exploration
                # More conservative adjustment with better scaling
                change = np.random.normal(0, 0.02 * sequence[i])  # Slightly smaller perturbation
                new_sequence[i] = max(0, sequence[i] + change)
        best_strategy = new_sequence
    
    # Normalize to preserve total mass
    total = sum(best_strategy)
    if total > 0.01:
        scale_factor = sum(sequence) / total if total > 0 else 1.0
        best_strategy = [x * scale_factor for x in best_strategy]
        
    return best_strategy

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Multiple attempts to find better solutions
    best_inv_c1 = 0
    best_result = None
    
    # Try several different approaches
    for attempt in range(30):  # Increased attempts for better exploration
        # Start with a better initial sequence
        best_sequence = create_better_candidate_sequence()
        
        # Try multiple iterations of improvement
        current_best_inv_c1 = compute_inv_c1(best_sequence)
        current_best = best_sequence.copy()
        
        # More extensive local search for each attempt
        for iteration in range(300):  # Increased iterations for better search
            # Try to improve the current sequence
            improved_sequence = get_good_direction_to_move_into(best_sequence)
            
            if improved_sequence is not None:
                # Check if improvement is significant
                new_inv_c1 = compute_inv_c1(improved_sequence)
                if new_inv_c1 > current_best_inv_c1:
                    best_sequence = improved_sequence
                    current_best_inv_c1 = new_inv_c1
                    current_best = best_sequence.copy()
        
        # Keep track of the best overall
        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_result = current_best.copy()
    
    # Ensure minimum sum constraint
    if best_result is not None:
        total_sum = sum(best_result)
        if total_sum < 0.01:
            # If too small, make it larger
            best_result = [x * 10 for x in best_result]
    else:
        # Fallback to a simple uniform sequence
        best_result = [1.0] * 100
    
    # Additional refinement pass with enhanced strategies
    if best_result is not None:
        # Run more optimization passes to further refine
        for _ in range(35):  # Increased from 30 to 35 for better final optimization
            refined_sequence = get_good_direction_to_move_into(best_result)
            if refined_sequence is not None:
                refined_fitness = compute_inv_c1(refined_sequence)
                if refined_fitness > best_inv_c1:
                    best_result = refined_sequence
                    best_inv_c1 = refined_fitness
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
