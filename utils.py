import torch
import einops
import numpy as np
from scipy.stats import entropy

import torch
import einops
import numpy as np
from scipy.stats import entropy

def analyze_neuron_frequencies(model, p=113, device='cuda'):
    """
    Analyzes the frequency components of neurons in a transformer model with continuous metrics.
    
    Args:
        model: The transformer model to analyze
        p (int): The modulus for arithmetic (default 113)
        device (str): Device to run computation on (default 'cuda')
    
    Returns:
        dict: Contains frequency analysis results and continuous metrics
    """
    # Generate all possible inputs
    all_data = torch.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
    
    # Create cache and get activations
    cache = {}
    model.cache_all(cache)
    _ = model(all_data)  # Run forward pass to fill cache
    
    # Extract neuron activations at final position
    neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
    d_mlp = neuron_acts.shape[1]
    
    # Create Fourier basis
    fourier_basis = []
    fourier_basis.append(torch.ones(p)/np.sqrt(p))
    
    for i in range(1, p//2 + 1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis[-2] /= fourier_basis[-2].norm()
        fourier_basis[-1] /= fourier_basis[-1].norm()
    
    fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
    
    # Center the neurons
    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    
    # Convert to Fourier basis
    def fft2d(mat):
        shape = mat.shape
        mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
        fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
        return fourier_mat.reshape(shape)
    
    fourier_neuron_acts = fft2d(neuron_acts_centered)
    fourier_neuron_acts_square = fourier_neuron_acts.reshape(p, p, d_mlp)
    
    # Analyze frequency components
    def extract_freq_2d(tensor, freq):
        index_1d = [0, 2*freq-1, 2*freq]
        return tensor[[[i]*3 for i in index_1d], [index_1d]*3]
    
    # Initialize matrix to store frequency strengths
    freq_strength_matrix = np.zeros((d_mlp, p//2))
    neuron_freqs = []
    neuron_frac_explained = []
    
    for ni in range(d_mlp):
        best_frac_explained = 0
        best_freq = 0  # Initialize to 0 instead of -1
        for freq in range(1, p//2):
            frac_explained = (
                extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq).pow(2).sum()/
                fourier_neuron_acts_square[:, :, ni].pow(2).sum()
            ).item()
            freq_strength_matrix[ni, freq] = frac_explained
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)  # Will always be >= 0
        neuron_frac_explained.append(best_frac_explained)
    
    neuron_freqs = np.array(neuron_freqs)    
    neuron_frac_explained = np.array(neuron_frac_explained)
    
    
    # A + B: [1, 2, 3, 4, 5, 3, 2, 4, 1, 5, 1]
    # A / B: [6, 7, 8, 9, 10, 8, 7, 9, 6, 10, 6]
    
    # I want to see if the key frequencies d

    
    # Calculate continuous metrics
    # 1. Overall specialization score (how much neurons prefer their top frequency)
    specialization_score = np.mean(neuron_frac_explained)
    
    # 2. Frequency distribution and its entropy
    freq_dist = np.bincount(neuron_freqs, minlength=p//2)[1:]  # Skip freq 0, safe now since all freqs >= 0
    total_neurons_with_freq = freq_dist.sum()
    if total_neurons_with_freq > 0:  # Avoid division by zero
        freq_dist = freq_dist / total_neurons_with_freq  # Normalize
    frequency_entropy = entropy(freq_dist + 1e-10)  # Add small constant to avoid log(0)
    
    # 3. Average frequency strength distribution per neuron
    avg_freq_distribution = np.mean(freq_strength_matrix, axis=0)[1:]  # Skip freq 0
    
    # 4. Gini coefficient of frequency distribution (measure of inequality)
    sorted_freq_dist = np.sort(freq_dist)
    n = len(sorted_freq_dist)
    index = np.arange(1, n + 1)
    gini = ((np.sum((2 * index - n - 1) * sorted_freq_dist)) / 
            (n * np.sum(sorted_freq_dist) + 1e-10))  # Add small constant to avoid division by zero
    
    # 5. Additional metrics for early training
    num_specialized_neurons = np.sum(neuron_frac_explained > 0.5)  # Count neurons with clear frequency preference
    max_freq_strength = np.max(neuron_frac_explained)  # Strongest frequency preference found

    # 6. TODO: I want to see whether some of the key frequencies vanish and reappear in a different neuron. Introduce this metric for me
    def calculate_frequency_transitions(current_freqs, current_strengths, prev_freqs=None, prev_strengths=None):
        if prev_freqs is None or prev_strengths is None:
            return {
                'freq_transitions': 0,
                'strength_preservation': 0.0,
                'active_freqs': set(current_freqs)
            }
        
        # Track frequencies that disappeared and reappeared
        prev_freq_to_neurons = {f: set(np.where(prev_freqs == f)[0]) 
                              for f in np.unique(prev_freqs)}
        curr_freq_to_neurons = {f: set(np.where(current_freqs == f)[0]) 
                              for f in np.unique(current_freqs)}
        
        # Calculate transitions
        transitions = 0
        strength_preservation = 0.0
        
        for freq in set(prev_freq_to_neurons.keys()) & set(curr_freq_to_neurons.keys()):
            prev_neurons = prev_freq_to_neurons[freq]
            curr_neurons = curr_freq_to_neurons[freq]
            
            if prev_neurons != curr_neurons:
                transitions += 1
                
                # Calculate strength preservation for this frequency
                prev_strength = max(prev_strengths[list(prev_neurons)])
                curr_strength = max(current_strengths[list(curr_neurons)])
                strength_preservation += min(prev_strength, curr_strength) / max(prev_strength, curr_strength)
        
        if transitions > 0:
            strength_preservation /= transitions
            
        return {
            'freq_transitions': transitions,
            'strength_preservation': float(strength_preservation),
            'active_freqs': set(current_freqs)
        }
    
    # Store the transition metrics in the results
    transition_metrics = calculate_frequency_transitions(
        neuron_freqs, 
        neuron_frac_explained
    )

    
    # Clear the cache to free memory
    model.remove_all_hooks()
    
    
    
    res = {
        'analysis/specialization_score': float(specialization_score),  # Convert to Python float for safer serialization
        'analysis/frequency_entropy': float(frequency_entropy),
        'analysis/gini_coefficient': float(gini),
        'analysis/mean_explained_variance': float(np.mean(neuron_frac_explained)),
        # 'frequency_distribution': freq_dist,
        # 'average_frequency_strength': avg_freq_distribution,
        # 'neuron_freqs': neuron_freqs,
        # 'neuron_frac_explained': neuron_frac_explained,
        'analysis/num_key_freqs': len(np.unique(neuron_freqs)),
        # 'frequency_strength_matrix': freq_strength_matrix,
        'analysis/num_specialized_neurons': int(num_specialized_neurons),
        'analysis/max_freq_strength': float(max_freq_strength),
        'analysis/frequency_transitions': transition_metrics['freq_transitions'],
        'analysis/strength_preservation': transition_metrics['strength_preservation'],
        'analysis/active_frequencies': len(transition_metrics['active_freqs'])

    }
    
    # Add previous state tracking as attributes of the function
    # TODO: This saves key frequency at epoch 10. I want to save key frequency at epoch 0.
    if not hasattr(analyze_neuron_frequencies, 'prev_freqs'):
        analyze_neuron_frequencies.prev_freqs = None
        analyze_neuron_frequencies.prev_strengths = None
    
        # Update previous state for next call
        analyze_neuron_frequencies.prev_freqs = neuron_freqs.copy()
        analyze_neuron_frequencies.prev_strengths = neuron_frac_explained.copy()
    
    return res

    

def plot_frequency_evolution(frequency_data_list, save_path=None):
    """
    Plots the evolution of frequency metrics over training.
    
    Args:
        frequency_data_list: List of frequency analysis results over training
        save_path: Optional path to save the plots
    """
    epochs = range(len(frequency_data_list))
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Specialization Score
    spec_scores = [d['specialization_score'] for d in frequency_data_list]
    axes[0, 0].plot(epochs, spec_scores)
    axes[0, 0].set_title('Neuron Specialization Score')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Score')
    
    # 2. Frequency Entropy
    entropies = [d['frequency_entropy'] for d in frequency_data_list]
    axes[0, 1].plot(epochs, entropies)
    axes[0, 1].set_title('Frequency Distribution Entropy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Entropy')
    
    # 3. Gini Coefficient
    ginis = [d['gini_coefficient'] for d in frequency_data_list]
    axes[1, 0].plot(epochs, ginis)
    axes[1, 0].set_title('Frequency Distribution Gini Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gini Coefficient')
    
    # 4. Frequency Distribution Heatmap
    freq_dists = np.stack([d['frequency_distribution'] for d in frequency_data_list])
    im = axes[1, 1].imshow(freq_dists.T, aspect='auto', interpolation='nearest')
    axes[1, 1].set_title('Frequency Distribution Evolution')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Frequency')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()