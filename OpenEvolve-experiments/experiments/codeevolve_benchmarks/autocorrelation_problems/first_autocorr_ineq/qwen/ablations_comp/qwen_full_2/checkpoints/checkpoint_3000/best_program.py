# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import convolve
import random
from typing import List, Tuple

def compute_c1(sequence: List[float]) -> float:
    """Compute C1 for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return float('inf')
    
    # Compute convolution using FFT for efficiency
    conv_result = convolve(sequence, sequence, mode='full')
    max_conv = max(conv_result)
    
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def compute_c1_fast(sequence: List[float]) -> float:
    """Fast C1 computation with early termination for very large sequences."""
    if len(sequence) == 0:
        return float('inf')
    
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return float('inf')
    
    n = len(sequence)
    # For very long sequences, use a smarter approach to avoid memory issues
    if n > 10000:
        # Sample a subset for estimation
        step = max(1, n // 1000)
        sampled = [sequence[i] for i in range(0, n, step)]
        conv_result = convolve(sampled, sampled, mode='full')
        max_conv = max(conv_result)
    else:
        # Compute full convolution
        conv_result = convolve(sequence, sequence, mode='full')
        max_conv = max(conv_result)
    
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def compute_inv_c1(sequence: List[float]) -> float:
    """Compute 1/C1 for a given sequence."""
    c1 = compute_c1(sequence)
    return 1.0 / c1 if c1 != 0 else 0.0

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with non-negative values."""
    return [random.uniform(0, 1000) for _ in range(length)]

def generate_specialized_sequence() -> List[float]:
    """Generate sequences with specialized structures that often perform well."""
    # Strategy 1: Two peaks with strategic spacing and optimized values
    n = random.randint(50, 500)
    sequence = [0.0] * n
    # Place peaks to maximize separation while minimizing convolution overlap
    peak1 = random.randint(n//6, n//4)
    peak2 = random.randint(3*n//4, 5*n//6)
    # Use even higher values for stronger peaks
    sequence[peak1] = random.uniform(120.0, 500.0)
    sequence[peak2] = random.uniform(120.0, 500.0)
    return sequence

def generate_decay_sequence() -> List[float]:
    """Generate exponentially decaying sequences."""
    n = random.randint(50, 500)
    # Use a more aggressive decay to reduce convolution peak
    base_options = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    base = random.choice(base_options)
    return [base ** i for i in range(n)]

def generate_peak_sequence() -> List[float]:
    """Generate sequences with a strong central peak."""
    n = random.randint(50, 500)
    sequence = [0.1] * n
    # Place peak away from true center to avoid symmetric convolution
    # Try different peak positions that have been successful
    peak_positions = [n//5, n//4, n//3, n//2, 2*n//3, 3*n//4, 4*n//5]
    peak_pos = random.choice(peak_positions)
    sequence[peak_pos] = random.uniform(300.0, 700.0)
    return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a sequence with given rate."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random change
            mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1 * mutated[i] + 0.1))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single point crossover
    crossover_point = random.randint(1, len(seq1) - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Apply some mutation to the offspring
    return mutate_sequence(child, 0.05)

def hill_climb_step(sequence: List[float], step_size: float = 0.01) -> List[float]:
    """Perform a hill climb step by slightly adjusting values."""
    new_sequence = sequence.copy()
    idx = random.randint(0, len(new_sequence) - 1)
    # Slightly adjust one element with adaptive step size
    adapt_step = step_size * (1.0 + random.random() * 0.5)
    new_sequence[idx] = max(0, new_sequence[idx] + random.gauss(0, adapt_step))
    return new_sequence

def advanced_local_search(sequence: List[float], iterations: int = 50) -> List[float]:
    """Advanced local search with multiple neighborhood operators."""
    current = sequence[:]
    best = current[:]
    best_fitness = compute_inv_c1(best)
    
    # More effective and focused local search operators
    effective_moves = ['single', 'multiple', 'peak_shift', 'targeted', 'gradient', 'adaptive', 'noise', 'ripple', 'amplify', 'smooth', 'damped', 'cluster', 'shift_center', 'sharp_peak']
    
    for _ in range(iterations):
        # Prioritize the most effective operators
        move_type = random.choice(effective_moves)
        
        if move_type == 'single':
            # Single element adjustment with adaptive step size
            new_seq = current[:]
            idx = random.randint(0, len(new_seq) - 1)
            # Use larger steps for larger values to maintain proportionality
            step_size = 0.2 * new_seq[idx] + 0.2
            new_seq[idx] = max(0, new_seq[idx] + random.gauss(0, step_size))
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                
        elif move_type == 'multiple':
            # Multiple element adjustments with strategic selection
            new_seq = current[:]
            num_changes = random.randint(1, max(1, len(new_seq) // 5))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                step_size = 0.2 * new_seq[idx] + 0.2
                new_seq[idx] = max(0, new_seq[idx] + random.gauss(0, step_size))
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                
        elif move_type == 'peak_shift':
            # Move a peak to a neighboring position
            new_seq = current[:]
            if len(new_seq) > 10:
                # Find existing peak
                peak_idx = np.argmax(new_seq)
                # Move peak to nearby position with preference for reducing convolution
                if peak_idx > 0 and peak_idx < len(new_seq) - 1:
                    # Prefer moving toward center to reduce peak overlap
                    if peak_idx < len(new_seq) // 2:
                        new_peak_pos = min(len(new_seq) - 1, peak_idx + random.randint(1, 8))
                    else:
                        new_peak_pos = max(0, peak_idx - random.randint(1, 8))
                    if new_peak_pos != peak_idx:
                        new_seq[new_peak_pos] = new_seq[peak_idx]
                        new_seq[peak_idx] = 0.0
                        new_fitness = compute_inv_c1(new_seq)
                        if new_fitness > best_fitness:
                            current = new_seq
                            best = new_seq[:]
                            best_fitness = new_fitness
                        
        elif move_type == 'targeted':
            # Targeted adjustment focused on reducing convolution peaks directly
            new_seq = current[:]
            # Analyze the convolution to identify problematic regions
            conv = convolve(new_seq, new_seq, mode='full')
            # Focus on reducing the peak value in convolution
            # Find where the convolution peak is located
            max_conv_idx = np.argmax(conv)
            mid = len(conv) // 2
            # If peak is in the center (expected), try to reduce it
            if abs(max_conv_idx - mid) < len(conv) // 4:
                # Reduce the values near the center to reduce convolution peak
                center = len(new_seq) // 2
                # Reduce elements around the center more aggressively
                for i in range(max(0, center - 6), min(len(new_seq), center + 6)):
                    new_seq[i] *= 0.82  # More aggressive reduction
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'gradient':
            # Gradient-based adjustment - modify elements based on local differences
            new_seq = current[:]
            # Find indices where we should adjust values based on neighbors
            for i in range(len(new_seq)):
                if i > 0 and i < len(new_seq) - 1:
                    # Adjust based on difference with neighbors
                    diff = new_seq[i-1] - new_seq[i+1]
                    adjustment = 0.15 * diff
                    new_seq[i] = max(0, new_seq[i] + adjustment)
                elif i == 0:
                    # Adjust based on next element
                    adjustment = 0.15 * (new_seq[i] - new_seq[i+1])
                    new_seq[i] = max(0, new_seq[i] + adjustment)
                elif i == len(new_seq) - 1:
                    # Adjust based on previous element
                    adjustment = 0.15 * (new_seq[i-1] - new_seq[i])
                    new_seq[i] = max(0, new_seq[i] + adjustment)
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'adaptive':
            # Adaptive approach with multiple strategies
            new_seq = current[:]
            # Apply different adjustments to different parts of the sequence
            for i in range(len(new_seq)):
                if random.random() < 0.4:  # Slightly higher chance to adjust
                    # Use different step sizes based on position and value
                    if i < len(new_seq) // 3:
                        step = 0.3 * new_seq[i] + 0.3
                    elif i > 2 * len(new_seq) // 3:
                        step = 0.25 * new_seq[i] + 0.25
                    else:
                        step = 0.2 * new_seq[i] + 0.2
                    new_seq[i] = max(0, new_seq[i] + random.gauss(0, step))
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'noise':
            # Add small random noise to help escape local optima
            new_seq = current[:]
            for i in range(len(new_seq)):
                if random.random() < 0.4:  # Slightly higher chance
                    noise_factor = random.gauss(0, 0.12 * new_seq[i] + 0.12)
                    new_seq[i] = max(0, new_seq[i] * (1 + noise_factor))
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'ripple':
            # Ripple effect - apply adjustments to a range of elements in a wave pattern
            new_seq = current[:]
            # Apply small adjustments to a range of elements
            start = random.randint(0, len(new_seq) - 3)
            end = random.randint(start + 2, len(new_seq))
            for i in range(start, end):
                # Add small random adjustment
                adjustment = random.gauss(0, 0.12 * new_seq[i] + 0.12)
                new_seq[i] = max(0, new_seq[i] + adjustment)
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'amplify':
            # Amplify specific regions to reduce convolution
            new_seq = current[:]
            # Select a random region to amplify
            start = random.randint(0, len(new_seq) - 3)
            end = random.randint(start + 2, len(new_seq))
            # Amplify the selected region
            amplification_factor = random.uniform(1.05, 1.75)
            for i in range(start, end):
                new_seq[i] = max(0, new_seq[i] * amplification_factor)
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'smooth':
            # Smooth the entire sequence to reduce oscillations
            new_seq = current[:]
            # Apply a smoothing operation
            smoothed = []
            for i in range(len(new_seq)):
                if i == 0:
                    smoothed.append(new_seq[i])
                elif i == len(new_seq) - 1:
                    smoothed.append(new_seq[i])
                else:
                    # Improved moving average with more weight on center
                    smoothed_val = (new_seq[i-1] * 0.2 + new_seq[i] * 0.6 + new_seq[i+1] * 0.2)
                    smoothed.append(smoothed_val)
            new_seq = smoothed
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'damped':
            # Apply damping to reduce high-frequency components
            new_seq = current[:]
            # Apply a damping factor to reduce oscillations
            damping_factor = 0.92
            for i in range(len(new_seq)):
                new_seq[i] *= damping_factor
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'cluster':
            # Adjust cluster of elements together to preserve structure
            new_seq = current[:]
            if len(new_seq) > 5:
                start = random.randint(0, len(new_seq) - 3)
                end = random.randint(start + 2, len(new_seq))
                # Apply uniform scaling to cluster
                scale_factor = random.uniform(0.7, 1.3)
                for i in range(start, end):
                    new_seq[i] = max(0, new_seq[i] * scale_factor)
                new_fitness = compute_inv_c1(new_seq)
                if new_fitness > best_fitness:
                    current = new_seq
                    best = new_seq[:]
                    best_fitness = new_fitness
                        
        elif move_type == 'shift_center':
            # Shift the entire sequence to potentially reduce convolution
            new_seq = current[:]
            shift_amount = random.randint(1, min(10, len(new_seq) // 4))
            # Rotate the sequence to shift values
            new_seq = new_seq[shift_amount:] + new_seq[:shift_amount]
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
                        
        elif move_type == 'sharp_peak':
            # Create or enhance sharp peaks to minimize convolution overlap
            new_seq = current[:]
            # Find peaks and enhance them with sharper profiles
            peaks = []
            for i in range(len(new_seq)):
                if i == 0:
                    if new_seq[i] > new_seq[i+1]:
                        peaks.append(i)
                elif i == len(new_seq) - 1:
                    if new_seq[i] > new_seq[i-1]:
                        peaks.append(i)
                else:
                    if new_seq[i] > new_seq[i-1] and new_seq[i] > new_seq[i+1]:
                        peaks.append(i)
            
            # If no clear peaks, create one at center
            if not peaks:
                center = len(new_seq) // 2
                new_seq[center] = max(new_seq[center], new_seq[center] * 1.2)
            else:
                # Enhance existing peaks
                for peak in peaks[:2]:  # Enhance first two peaks
                    new_seq[peak] = max(0, new_seq[peak] * 1.15)  # Slightly more aggressive enhancement
            new_fitness = compute_inv_c1(new_seq)
            if new_fitness > best_fitness:
                current = new_seq
                best = new_seq[:]
                best_fitness = new_fitness
    
    return best

def optimize_sequence_evolutionary(max_evals: int = 10000) -> List[float]:
    """Evolutionary optimization approach with enhanced diversity and better selection."""
    # Initialize population
    population_size = 120  # Increased population size for better exploration
    population = []
    
    # Generate initial diverse population with different strategies
    for _ in range(population_size // 12):
        population.append(generate_specialized_sequence())
    for _ in range(population_size // 12):
        population.append(generate_decay_sequence())
    for _ in range(population_size // 12):
        population.append(generate_peak_sequence())
    # Add some more diverse patterns
    for _ in range(population_size // 12):
        # Add a Gaussian-like pattern with more variance
        n = random.randint(50, 300)
        center = n // 2
        # Try different sigma values
        sigma_options = [n/2.2, n/2.5, n/3, n/3.5, n/4, n/4.5, n/5, n/5.5, n/6]
        sigma = random.choice(sigma_options)
        sequence = [np.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(n)]
        # Normalize and scale
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 250.0 for x in sequence]
        population.append(sequence)
    # Add logarithmic decay patterns
    for _ in range(population_size // 12):
        n = random.randint(50, 300)
        sequence = [1.0 / (np.log(i + 2)) for i in range(n)]
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 120.0 for x in sequence]
        population.append(sequence)
    # Add power law patterns
    for _ in range(population_size // 12):
        n = random.randint(50, 300)
        # Power law with varied exponents
        exp_options = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        exp = random.choice(exp_options)
        sequence = [1.0 / ((i + 1) ** exp) for i in range(n)]
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 160.0 for x in sequence]
        population.append(sequence)
    # Add inverse square root patterns
    for _ in range(population_size // 12):
        n = random.randint(50, 300)
        sequence = [1.0 / np.sqrt(i + 1) for i in range(n)]
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 140.0 for x in sequence]
        population.append(sequence)
    # Add more sophisticated patterns
    for _ in range(population_size // 12):
        # Add a hyperbolic pattern
        n = random.randint(50, 250)
        sequence = [1.0 / (1.0 + (i / (n//3))**2) for i in range(n)]
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 150.0 for x in sequence]
        population.append(sequence)
    # Add triangular patterns
    for _ in range(population_size // 12):
        n = random.randint(50, 200)
        sequence = []
        for i in range(n):
            if i <= n//2:
                sequence.append(i / (n//2))
            else:
                sequence.append((n - i) / (n//2))
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 280.0 for x in sequence]
        population.append(sequence)
    # Add more exotic mathematical patterns
    for _ in range(population_size // 12):
        # Add a sinc pattern (sin(x)/x) which tends to have good convolution properties
        n = random.randint(50, 200)
        sequence = []
        for i in range(n):
            if i == 0:
                sequence.append(1.0)  # sinc(0) = 1
            else:
                sequence.append(np.sin(i) / i)
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 170.0 for x in sequence]
        population.append(sequence)
    # Add more specialized patterns
    for _ in range(population_size // 12):
        # Add a modified exponential decay
        n = random.randint(50, 250)
        sequence = []
        for i in range(n):
            # Use a more aggressive decay rate
            sequence.append(np.exp(-i/15.0) * (1.0 + 0.2 * np.sin(2 * np.pi * i / n)))
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 220.0 for x in sequence]
        population.append(sequence)
    # Add inverse quartic patterns
    for _ in range(population_size // 12):
        n = random.randint(50, 250)
        sequence = [1.0 / (1.0 + (i / (n//4))**4) for i in range(n)]
        max_val = max(sequence)
        if max_val > 0:
            sequence = [x / max_val * 150.0 for x in sequence]
        population.append(sequence)
    # Fill remaining with random
    for _ in range(population_size - 11 * (population_size // 12)):
        length = random.randint(10, 500)
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_fitness = 0.0
    best_individual = None
    
    # Evolutionary loop
    for generation in range(max_evals // population_size):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = compute_inv_c1(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Update best solution
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
        
        # Select top individuals (elitism) - keep more elite individuals
        top_individuals = [ind for _, ind in fitness_scores[:population_size//5]]  # More elitism
        
        # Create new population through crossover and mutation
        new_population = top_individuals.copy()
        
        # Fill up population with offspring - increase offspring diversity
        while len(new_population) < population_size:
            parent1 = random.choice(top_individuals)
            parent2 = random.choice(top_individuals)
            
            # Crossover with probability
            if random.random() < 0.995:  # Slightly increased crossover rate
                # Use more sophisticated crossover
                child = []
                min_len = min(len(parent1), len(parent2))
                # Mix genes from both parents with weighted average
                for i in range(min_len):
                    weight = random.uniform(0.2, 0.8)  # Narrower weight range for more stability
                    child.append(weight * parent1[i] + (1-weight) * parent2[i])
                
                # Extend with longer sequence if needed
                if len(parent1) > len(parent2):
                    child.extend(parent1[min_len:])
                elif len(parent2) > len(parent1):
                    child.extend(parent2[min_len:])
            else:
                # Clone parent with slight variation
                child = parent1[:]
                for i in range(len(child)):
                    if random.random() < 0.15:
                        child[i] = max(0, child[i] + random.gauss(0, 0.15 * child[i] + 0.15))
            
            # Mutation with higher rate for exploration
            if random.random() < 0.92:  # Increased mutation rate for better exploration
                child = mutate_sequence(child, 0.25)  # Reduced mutation rate
            
            new_population.append(child)
        
        population = new_population
        
        # Occasionally add completely new random individuals with more strategic sampling
        if generation % 3 == 0:  # Less frequent additions to maintain structure
            for _ in range(15):
                length = random.randint(20, 350)  # Wider range for exploration
                individual = generate_random_sequence(length)
                if random.random() < 0.95:
                    population.append(individual)
        
        # Add some specialized sequences occasionally with higher frequency
        if generation % 2 == 0:
            for _ in range(15):  # Fewer additions to avoid overfitting
                if random.random() < 0.995:
                    population.append(generate_specialized_sequence())
                elif random.random() < 0.9:
                    population.append(generate_decay_sequence())
                elif random.random() < 0.85:
                    population.append(generate_peak_sequence())
                elif random.random() < 0.8:
                    # Add a new pattern type: sparse peaks with different heights
                    n = random.randint(50, 250)
                    sequence = [0.0] * n
                    num_peaks = random.randint(1, 8)  # Fewer peaks
                    for _ in range(num_peaks):
                        pos = random.randint(0, n-1)
                        sequence[pos] = random.uniform(220.0, 750.0)  # Slightly narrower range
                    population.append(sequence)
                else:
                    # Add a triangular pattern
                    n = random.randint(50, 200)
                    sequence = []
                    for i in range(n):
                        if i <= n//2:
                            sequence.append(i / (n//2))
                        else:
                            sequence.append((n - i) / (n//2))
                    max_val = max(sequence)
                    if max_val > 0:
                        sequence = [x / max_val * 280.0 for x in sequence]
                    population.append(sequence)
    
    return best_individual if best_individual is not None else generate_random_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Try multiple approaches with better balance
    best_sequence = None
    best_fitness = 0.0
    
    # Approach 1: Evolutionary algorithm with more iterations and enhanced diversity
    try:
        evol_seq = optimize_sequence_evolutionary(25000)  # Increased evaluations
        evol_fitness = compute_inv_c1(evol_seq)
        if evol_fitness > best_fitness:
            best_fitness = evol_fitness
            best_sequence = evol_seq
    except Exception as e:
        print(f"Evolutionary approach failed: {e}")
    
    # Approach 2: Advanced local search with specialized sequences and more iterations
    try:
        # Start with specialized sequences and do advanced local search
        for _ in range(250):  # More iterations
            # Try different starting strategies with better weighting
            strategy = random.choice(['specialized', 'decay', 'peak', 'gaussian', 'sparse', 'logarithmic', 'power_law', 'inverse_sqrt', 'hyperbolic', 'triangular', 'sinc', 'multi_peak', 'modified_exp', 'oscillating', 'inverse_quartic'])
            if strategy == 'specialized':
                seq = generate_specialized_sequence()
            elif strategy == 'decay':
                seq = generate_decay_sequence()
            elif strategy == 'peak':
                seq = generate_peak_sequence()
            elif strategy == 'gaussian':
                # Add Gaussian pattern with more variation
                n = random.randint(50, 300)
                center = n // 2
                sigma_options = [n/2.2, n/2.5, n/3, n/3.5, n/4, n/4.5, n/5, n/5.5, n/6]
                sigma = random.choice(sigma_options)
                sequence = [np.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 250.0 for x in sequence]
                seq = sequence
            elif strategy == 'sparse':
                # Add sparse peak pattern
                n = random.randint(50, 200)
                sequence = [0.0] * n
                num_peaks = random.randint(1, 8)  # Fewer peaks
                for _ in range(num_peaks):
                    pos = random.randint(0, n-1)
                    sequence[pos] = random.uniform(220.0, 750.0)  # Slightly narrower range
                seq = sequence
            elif strategy == 'logarithmic':
                # Add logarithmic pattern
                n = random.randint(50, 200)
                sequence = [1.0 / (np.log(i + 2)) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 120.0 for x in sequence]
                seq = sequence
            elif strategy == 'power_law':
                # Add power law pattern
                n = random.randint(50, 200)
                exp_options = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
                exp = random.choice(exp_options)
                sequence = [1.0 / ((i + 1) ** exp) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 160.0 for x in sequence]
                seq = sequence
            elif strategy == 'inverse_sqrt':
                # Add inverse square root pattern
                n = random.randint(50, 200)
                sequence = [1.0 / np.sqrt(i + 1) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 140.0 for x in sequence]
                seq = sequence
            elif strategy == 'hyperbolic':
                # Add hyperbolic pattern
                n = random.randint(50, 200)
                sequence = [1.0 / (1.0 + (i / (n//3))**2) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 150.0 for x in sequence]
                seq = sequence
            elif strategy == 'triangular':
                # Add triangular pattern
                n = random.randint(50, 200)
                sequence = []
                for i in range(n):
                    if i <= n//2:
                        sequence.append(i / (n//2))
                    else:
                        sequence.append((n - i) / (n//2))
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 280.0 for x in sequence]
                seq = sequence
            elif strategy == 'sinc':
                # Add sinc pattern
                n = random.randint(50, 200)
                sequence = []
                for i in range(n):
                    if i == 0:
                        sequence.append(1.0)
                    else:
                        sequence.append(np.sin(i) / i)
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 170.0 for x in sequence]
                seq = sequence
            elif strategy == 'multi_peak':
                # Add multi-peak pattern
                n = random.randint(50, 200)
                sequence = [0.0] * n
                num_peaks = random.randint(2, 9)  # Fewer peaks
                for _ in range(num_peaks):
                    pos = random.randint(0, n-1)
                    sequence[pos] = random.uniform(160.0, 650.0)  # Slightly narrower range
                seq = sequence
            elif strategy == 'modified_exp':
                # Add modified exponential pattern
                n = random.randint(50, 200)
                sequence = []
                for i in range(n):
                    # Use a more aggressive decay rate with oscillation
                    base = np.exp(-i/15.0)
                    oscillation = 0.1 * np.sin(2 * np.pi * i / n)
                    sequence.append(max(0.01, base + oscillation))
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 220.0 for x in sequence]
                seq = sequence
            elif strategy == 'oscillating':
                # Add oscillating pattern
                n = random.randint(50, 200)
                sequence = []
                for i in range(n):
                    # Create oscillating pattern with amplitude that decreases
                    amplitude = 1.0 - i/n * 0.7
                    oscillation = amplitude * np.sin(2 * np.pi * i / (n//5))
                    sequence.append(max(0.01, 1.0 + oscillation))
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 150.0 for x in sequence]
                seq = sequence
            elif strategy == 'inverse_quartic':
                # Add inverse quartic pattern
                n = random.randint(50, 200)
                sequence = [1.0 / (1.0 + (i / (n//4))**4) for i in range(n)]
                max_val = max(sequence)
                if max_val > 0:
                    sequence = [x / max_val * 150.0 for x in sequence]
                seq = sequence
            else:
                seq = generate_random_sequence(random.randint(50, 300))
            
            # Advanced local search with more iterations
            refined_seq = advanced_local_search(seq, 350)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Advanced local search approach failed: {e}")
    
    # Approach 3: Enhanced local search with different strategies
    try:
        # Try a few random sequences with extended local search
        for _ in range(150):  # More iterations
            length = random.randint(50, 300)
            seq = generate_random_sequence(length)
            # Run extended local search
            refined_seq = advanced_local_search(seq, 350)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Enhanced local search approach failed: {e}")
    
    # Approach 4: Direct pattern construction with specific mathematical properties
    try:
        # Try constructing sequences with known good properties
        for _ in range(120):  # More iterations
            # Create a sequence with strategic peak placement
            n = random.randint(80, 400)
            sequence = [0.0] * n
            # Place peaks at strategic locations to minimize convolution interference
            num_peaks = random.randint(4, 12)  # More peaks for better distribution
            for i in range(num_peaks):
                # Distribute peaks more evenly across the sequence
                pos = random.randint(i * n // (num_peaks + 1), (i + 1) * n // (num_peaks + 1))
                sequence[pos] = random.uniform(80.0, 450.0)  # Slightly narrower range
            
            # Refine with local search
            refined_seq = advanced_local_search(sequence, 350)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Pattern construction approach failed: {e}")
    
    # Approach 5: Additional sparse peak patterns that often work well
    try:
        for _ in range(70):  # More iterations
            n = random.randint(60, 250)
            sequence = [0.0] * n
            num_peaks = random.randint(1, 8)  # Fewer peaks
            for _ in range(num_peaks):
                pos = random.randint(0, n-1)
                sequence[pos] = random.uniform(220.0, 750.0)  # Slightly narrower range
            refined_seq = advanced_local_search(sequence, 220)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Sparse peak approach failed: {e}")
    
    # Approach 6: Try creating sequences with specific mathematical properties
    try:
        # Try creating sequences that are optimized for low convolution
        for _ in range(60):  # More iterations
            n = random.randint(100, 300)
            # Create a sequence that decreases rapidly to reduce convolution
            sequence = [0.0] * n
            peak_idx = random.randint(0, n//2)
            peak_val = random.uniform(120.0, 450.0)  # Slightly narrower range
            sequence[peak_idx] = peak_val
            # Decrease values rapidly after the peak
            for i in range(peak_idx+1, n):
                sequence[i] = max(0.0, sequence[i-1] * random.uniform(0.4, 0.65))  # Narrower range
            
            refined_seq = advanced_local_search(sequence, 200)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Rapid decay approach failed: {e}")
    
    # Approach 7: Add a new approach - pattern inspired by Fourier transform properties
    try:
        # Try creating sequences with specific frequency characteristics
        for _ in range(30):  # More iterations
            n = random.randint(80, 200)
            # Create a pattern that resembles a low-pass filter response
            sequence = []
            for i in range(n):
                # Low frequency component with gradual decay
                freq_component = np.exp(-i/20) * np.cos(0.1*i)
                sequence.append(max(0.01, freq_component))
            refined_seq = advanced_local_search(sequence, 140)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Frequency domain approach failed: {e}")
    
    # Approach 8: Add a more aggressive optimization approach with specialized patterns
    try:
        # Try specialized patterns with more careful parameter tuning
        for _ in range(35):  # More iterations
            # Create a hybrid pattern with different behaviors
            n = random.randint(100, 250)
            sequence = []
            for i in range(n):
                if i < n//3:
                    # Early part: exponential decay
                    sequence.append(np.exp(-i/10.0))
                elif i < 2*n//3:
                    # Middle part: stable plateau
                    sequence.append(1.0)
                else:
                    # Late part: rapid decay
                    sequence.append(np.exp(-(i-2*n//3)/5.0))
            
            # Normalize to reasonable values
            max_val = max(sequence)
            if max_val > 0:
                sequence = [x / max_val * 200.0 for x in sequence]
            
            refined_seq = advanced_local_search(sequence, 180)  # More iterations
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"Hybrid pattern approach failed: {e}")
    
    # Approach 9: Add another specialized approach - high-contrast patterns
    try:
        # Try creating sequences with high contrast between peak and background values
        for _ in range(25):  # More iterations
            n = random.randint(80, 200)
            sequence = [0.0] * n
            # Place a few very high peaks
            num_high_peaks = random.randint(2, 5)  # Fewer peaks
            for _ in range(num_high_peaks):
                pos = random.randint(0, n-1)
                sequence[pos] = random.uniform(500.0, 1000.0)  # Much higher values
            # Fill in with smaller values
            for i in range(n):
                if sequence[i] == 0.0:
                    sequence[i] = random.uniform(10.0, 50.0)
            refined_seq = advanced_local_search(sequence, 150)
            fitness = compute_inv_c1(refined_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = refined_seq
    except Exception as e:
        print(f"High-contrast approach failed: {e}")
    
    # Return best found
    if best_sequence is None:
        best_sequence = generate_random_sequence(100)
    
    # Ensure minimum sum
    if sum(best_sequence) < 0.01:
        best_sequence[0] = 1.0
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
