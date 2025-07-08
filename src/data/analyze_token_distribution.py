#!/usr/bin/env python3
"""
Analyze token distribution in training data to understand class imbalance
"""

import pickle
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

def load_sequence_file(file_path: Path) -> Dict:
    """Load a single sequence pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_chord_distribution(data_dir: Path, split: str = "train", max_files: int = None) -> Dict:
    """Analyze chord token distribution in the dataset."""
    
    split_dir = data_dir / split
    pkl_files = list(split_dir.glob("sequence_*.pkl"))
    
    if max_files:
        pkl_files = pkl_files[:max_files]
    
    print(f"Analyzing {len(pkl_files)} files from {split} split...")
    
    # Counters for different aspects
    chord_token_counts = Counter()
    chord_position_counts = Counter()  # Count chords by position in sequence
    sequence_lengths = []
    total_chord_positions = 0
    
    # Load tokenizer info
    tokenizer_file = split_dir / "tokenizer_info.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, 'r') as f:
            tokenizer_info = json.load(f)
        chord_token_start = tokenizer_info['melody_vocab_size'] + 1
    else:
        # Fallback - assume melody vocab is 178
        chord_token_start = 179
        tokenizer_info = {"melody_vocab_size": 178}
    
    for i, file_path in enumerate(pkl_files):
        if i % 100 == 0:
            print(f"  Processing file {i+1}/{len(pkl_files)}")
        
        try:
            sequence_data = load_sequence_file(file_path)
            
            # Handle FrameSequence objects
            if hasattr(sequence_data, 'chord_tokens'):
                chord_tokens = sequence_data.chord_tokens
                melody_tokens = sequence_data.melody_tokens
                sequence_length = len(chord_tokens)
            else:
                # Handle dictionary format
                chord_tokens = sequence_data.get('chord_tokens', [])
                melody_tokens = sequence_data.get('melody_tokens', [])
                sequence_length = len(chord_tokens) if chord_tokens else 0
            
            if sequence_length == 0:
                continue
                
            sequence_lengths.append(sequence_length)
            
            # Extract chord tokens (all positions in chord_tokens array)
            for pos, token in enumerate(chord_tokens):
                if token >= chord_token_start:  # Valid chord token
                    chord_token_counts[token] += 1
                    chord_position_counts[pos] += 1
                    total_chord_positions += 1
                        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    # Calculate statistics
    total_sequences = len(pkl_files)
    avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
    unique_chord_tokens = len(chord_token_counts)
    
    # Get most/least common chords
    most_common_chords = chord_token_counts.most_common(10)
    least_common_chords = chord_token_counts.most_common()[-10:]
    
    # Calculate entropy and diversity metrics
    if total_chord_positions > 0:
        chord_probs = np.array(list(chord_token_counts.values())) / total_chord_positions
        chord_entropy = -np.sum(chord_probs * np.log(chord_probs + 1e-10))
        
        # Calculate dominance ratio
        top_chord_count = most_common_chords[0][1] if most_common_chords else 0
        dominance_ratio = top_chord_count / total_chord_positions
        
        # Calculate Gini coefficient (inequality measure)
        sorted_counts = sorted(chord_token_counts.values())
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    else:
        chord_entropy = 0
        dominance_ratio = 0
        gini = 0
    
    results = {
        'dataset_stats': {
            'split': split,
            'total_sequences': total_sequences,
            'total_chord_positions': total_chord_positions,
            'avg_sequence_length': avg_sequence_length,
            'unique_chord_tokens': unique_chord_tokens,
            'chord_token_start': chord_token_start
        },
        'distribution_stats': {
            'chord_entropy': chord_entropy,
            'dominance_ratio': dominance_ratio,
            'gini_coefficient': gini,
            'most_common_chords': most_common_chords,
            'least_common_chords': least_common_chords
        },
        'token_counts': dict(chord_token_counts),
        'position_distribution': dict(chord_position_counts),
        'tokenizer_info': tokenizer_info
    }
    
    return results

def calculate_class_weights(token_counts: Dict[int, int], method: str = "inverse_frequency") -> Dict[int, float]:
    """Calculate class weights for balanced training."""
    
    total_samples = sum(token_counts.values())
    unique_classes = len(token_counts)
    
    if method == "inverse_frequency":
        # Weight = total_samples / (n_classes * class_count)
        weights = {}
        for token, count in token_counts.items():
            weights[token] = total_samples / (unique_classes * count)
            
    elif method == "sqrt_inverse":
        # Weight = sqrt(total_samples / class_count)
        weights = {}
        for token, count in token_counts.items():
            weights[token] = np.sqrt(total_samples / count)
            
    elif method == "log_inverse":
        # Weight = log(total_samples / class_count)
        weights = {}
        for token, count in token_counts.items():
            weights[token] = np.log(total_samples / count)
            
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to have mean of 1.0
    weight_values = list(weights.values())
    mean_weight = np.mean(weight_values)
    normalized_weights = {token: weight / mean_weight for token, weight in weights.items()}
    
    return normalized_weights

def plot_distribution_analysis(results: Dict, output_dir: Path):
    """Create visualization plots for the distribution analysis."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Check if we have any data to plot
    if not results['token_counts']:
        print("No chord tokens found - skipping visualizations")
        return
    
    # 1. Top 50 most common chord tokens
    plt.figure(figsize=(15, 8))
    most_common = Counter(results['token_counts']).most_common(50)
    tokens, counts = zip(*most_common)
    
    plt.subplot(2, 2, 1)
    plt.bar(range(len(tokens)), counts)
    plt.xlabel('Chord Token Rank')
    plt.ylabel('Frequency')
    plt.title('Top 50 Most Common Chord Tokens')
    plt.yscale('log')
    
    # 2. Distribution histogram
    plt.subplot(2, 2, 2)
    count_values = list(results['token_counts'].values())
    plt.hist(count_values, bins=50, alpha=0.7)
    plt.xlabel('Token Frequency')
    plt.ylabel('Number of Tokens')
    plt.title('Distribution of Token Frequencies')
    plt.yscale('log')
    plt.xscale('log')
    
    # 3. Cumulative distribution
    plt.subplot(2, 2, 3)
    sorted_counts = sorted(count_values, reverse=True)
    cumsum = np.cumsum(sorted_counts)
    plt.plot(cumsum / cumsum[-1])
    plt.xlabel('Token Rank')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Token Usage')
    
    # 4. Top 20 vs bottom 20 comparison
    plt.subplot(2, 2, 4)
    top_20 = Counter(results['token_counts']).most_common(20)
    bottom_20 = Counter(results['token_counts']).most_common()[-20:]
    
    top_counts = [count for _, count in top_20]
    bottom_counts = [count for _, count in bottom_20]
    
    x = np.arange(20)
    width = 0.35
    
    plt.bar(x - width/2, top_counts, width, label='Top 20', alpha=0.8)
    plt.bar(x + width/2, bottom_counts, width, label='Bottom 20', alpha=0.8)
    plt.xlabel('Rank within group')
    plt.ylabel('Frequency')
    plt.title('Top 20 vs Bottom 20 Token Frequencies')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chord_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Class weights visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate different weighting schemes
    inv_freq_weights = calculate_class_weights(results['token_counts'], "inverse_frequency")
    sqrt_weights = calculate_class_weights(results['token_counts'], "sqrt_inverse")
    
    # Plot weights for top 50 most common tokens
    top_50_tokens = [token for token, _ in Counter(results['token_counts']).most_common(50)]
    inv_weights_top50 = [inv_freq_weights[token] for token in top_50_tokens]
    sqrt_weights_top50 = [sqrt_weights[token] for token in top_50_tokens]
    
    x = np.arange(len(top_50_tokens))
    width = 0.35
    
    plt.bar(x - width/2, inv_weights_top50, width, label='Inverse Frequency', alpha=0.8)
    plt.bar(x + width/2, sqrt_weights_top50, width, label='Sqrt Inverse', alpha=0.8)
    
    plt.xlabel('Token Rank (Most Common)')
    plt.ylabel('Class Weight')
    plt.title('Class Weights for Top 50 Most Common Tokens')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze chord token distribution")
    parser.add_argument("--data_dir", type=str, default="data/interim", help="Data directory")
    parser.add_argument("--split", type=str, default="train", help="Data split to analyze")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for results")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze distribution
    print("🔍 Analyzing chord token distribution...")
    results = analyze_chord_distribution(data_dir, args.split, args.max_files)
    
    # Print summary
    stats = results['dataset_stats']
    dist_stats = results['distribution_stats']
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Split: {stats['split']}")
    print(f"   Total sequences: {stats['total_sequences']:,}")
    print(f"   Total chord positions: {stats['total_chord_positions']:,}")
    print(f"   Average sequence length: {stats['avg_sequence_length']:.1f}")
    print(f"   Unique chord tokens: {stats['unique_chord_tokens']:,}")
    
    print(f"\n📈 Distribution Statistics:")
    print(f"   Chord entropy: {dist_stats['chord_entropy']:.3f}")
    print(f"   Dominance ratio: {dist_stats['dominance_ratio']:.3f}")
    print(f"   Gini coefficient: {dist_stats['gini_coefficient']:.3f}")
    
    print(f"\n🎯 Most Common Chords:")
    for i, (token, count) in enumerate(dist_stats['most_common_chords'][:5]):
        percentage = (count / stats['total_chord_positions']) * 100
        print(f"   {i+1}. Token {token}: {count:,} ({percentage:.2f}%)")
    
    # Diagnose class imbalance severity
    dominance = dist_stats['dominance_ratio']
    gini = dist_stats['gini_coefficient']
    
    print(f"\n⚠️  Class Imbalance Assessment:")
    if dominance > 0.5:
        print(f"   SEVERE: Top token dominates {dominance:.1%} of data")
    elif dominance > 0.2:
        print(f"   MODERATE: Top token represents {dominance:.1%} of data")
    else:
        print(f"   MILD: Reasonable distribution (top token {dominance:.1%})")
    
    if gini > 0.8:
        print(f"   HIGH INEQUALITY: Gini coefficient {gini:.3f}")
    elif gini > 0.6:
        print(f"   MODERATE INEQUALITY: Gini coefficient {gini:.3f}")
    else:
        print(f"   LOW INEQUALITY: Gini coefficient {gini:.3f}")
    
    # Calculate and save class weights
    print(f"\n⚖️  Calculating class weights...")
    
    weight_methods = ["inverse_frequency", "sqrt_inverse", "log_inverse"]
    class_weights = {}
    
    for method in weight_methods:
        try:
            weights = calculate_class_weights(results['token_counts'], method)
            class_weights[method] = weights
            
            # Print statistics about weights
            weight_values = list(weights.values())
            print(f"   {method}: min={min(weight_values):.3f}, max={max(weight_values):.3f}, mean={np.mean(weight_values):.3f}")
            
        except Exception as e:
            print(f"   Failed to calculate {method}: {e}")
    
    # Save all results
    output_file = output_dir / f"chord_distribution_analysis_{args.split}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert dictionary keys to strings for JSON compatibility
    def convert_dict_keys(d):
        if isinstance(d, dict):
            return {str(k): convert_dict_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict_keys(item) for item in d]
        else:
            return d
    
    save_results = {
        'analysis_results': results,
        'class_weights': class_weights,
        'recommendations': {
            'use_weighted_loss': dominance > 0.3 or gini > 0.7,
            'recommended_weight_method': 'sqrt_inverse' if dominance > 0.5 else 'inverse_frequency',
            'use_focal_loss': dominance > 0.4,
            'label_smoothing_alpha': 0.1 if dominance > 0.3 else 0.05
        }
    }
    
    # Deep convert numpy types and dict keys
    import json
    save_results = convert_dict_keys(save_results)
    json_str = json.dumps(save_results, default=convert_numpy, indent=2)
    
    with open(output_file, 'w') as f:
        f.write(json_str)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Create visualizations
    print(f"📊 Creating visualizations...")
    plot_distribution_analysis(results, output_dir)
    print(f"📊 Plots saved to {output_dir}")
    
    # Final recommendations
    recommendations = save_results['recommendations']
    print(f"\n🎯 Training Recommendations:")
    
    if not results['token_counts']:
        print(f"   ⚠️  No chord tokens found in analysis")
        print(f"   ⚠️  Check data preprocessing and tokenization")
        print(f"   ⚠️  Verify chord_token_start value ({tokenizer_info.get('chord_token_start', 'unknown')})")
        return
    
    if recommendations['use_weighted_loss']:
        print(f"   ✓ Use weighted CrossEntropyLoss with method: {recommendations['recommended_weight_method']}")
    else:
        print(f"   ✓ Standard CrossEntropyLoss should work")
        
    if recommendations['use_focal_loss']:
        print(f"   ✓ Consider Focal Loss (alpha=0.25, gamma=2.0)")
        
    print(f"   ✓ Use label smoothing with alpha={recommendations['label_smoothing_alpha']}")
    print(f"   ✓ Lower learning rate (e.g., 5e-4 instead of 1e-3)")
    print(f"   ✓ Add diversity penalties during training")

if __name__ == "__main__":
    main() 