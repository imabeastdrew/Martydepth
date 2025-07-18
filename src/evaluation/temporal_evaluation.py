#!/usr/bin/env python3
"""
Temporal evaluation system for measuring harmonic quality over time.

This module implements step-by-step generation and evaluation to track
how harmonic quality changes during sequence generation, matching the
methodology from research papers.

Supports three evaluation scenarios:
1. Primed with ground truth - Model starts with correct context (several beats)
2. Cold start - Model starts with minimal context (more challenging)
3. Perturbed midway - Melody gets transposed by tritone at beat 17
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import check_harmony
from src.config.tokenization_config import SILENCE_TOKEN, CHORD_TOKEN_START, PAD_TOKEN
from src.data.preprocess_frames import MIDITokenizer

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Create a global tokenizer instance for efficiency
_melody_tokenizer = None

def get_melody_tokenizer():
    """Get a singleton MIDITokenizer instance for efficiency."""
    global _melody_tokenizer
    if _melody_tokenizer is None:
        _melody_tokenizer = MIDITokenizer()
    return _melody_tokenizer


def transpose_melody_token(melody_token: int, semitones: int = 6) -> int:
    """
    Transpose a melody token by the specified number of semitones.
    
    Args:
        melody_token: Original melody token
        semitones: Number of semitones to transpose (6 = tritone)
        
    Returns:
        Transposed melody token, or original if not transposable
    """
    # Skip non-musical tokens
    if melody_token == PAD_TOKEN or melody_token == SILENCE_TOKEN:
        return melody_token
    
    # Use singleton MIDITokenizer for efficiency
    melody_tokenizer = get_melody_tokenizer()
    midi_note, is_onset = melody_tokenizer.decode_token(melody_token)
    
    # Skip if not a valid melody note
    if midi_note is None:
        return melody_token
    
    # Transpose the MIDI note
    transposed_midi = midi_note + semitones
    
    # Ensure we stay within valid MIDI range (0-127)
    transposed_midi = max(0, min(127, transposed_midi))
    
    # Re-encode to token
    try:
        onset_token, hold_token = melody_tokenizer.encode_midi_note(transposed_midi)
        # Return the appropriate token type (onset or hold)
        return onset_token if is_onset else hold_token
    except:
        # If encoding fails, return original token
        return melody_token


def calculate_harmony_at_timestep(melody_token: int, chord_token: int, tokenizer_info: Dict) -> Optional[float]:
    """
    Calculate harmony ratio for a single melody-chord pair at one timestep.
    
    Args:
        melody_token: Single melody token at current timestep
        chord_token: Single chord token at current timestep  
        tokenizer_info: Tokenizer information dictionary
        
    Returns:
        float: 1.0 if in harmony, 0.0 if not in harmony, None if silence/invalid
    """
    # Skip PAD tokens
    if melody_token == PAD_TOKEN or chord_token == PAD_TOKEN:
        return None
    
    # Skip silence tokens
    if melody_token == SILENCE_TOKEN or chord_token == SILENCE_TOKEN:
        return None
    
    # Skip if chord token is not in valid range
    if chord_token < CHORD_TOKEN_START:
        return None
    
    # Get melody pitch using singleton MIDITokenizer
    melody_tokenizer = get_melody_tokenizer()
    midi_note, is_onset = melody_tokenizer.decode_token(melody_token)
    
    # Skip if not a valid melody note
    if midi_note is None:
        return None
    
    # Get chord information
    chord_token_str = str(chord_token)
    token_to_chord = tokenizer_info.get("token_to_chord", {})
    
    if chord_token_str not in token_to_chord:
        return None
    
    chord_info = token_to_chord[chord_token_str]
    
    # Check harmony using existing logic
    if check_harmony(midi_note, chord_info):
        return 1.0
    else:
        return 0.0


def extract_scenario_sequences(input_tokens: torch.Tensor, target_tokens: torch.Tensor, 
                             scenario: str, perturbation_beat: int = 17) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract sequences for different evaluation scenarios following research paper methodology.
    
    Args:
        input_tokens: Input interleaved tokens [chord_0, melody_0, chord_1, melody_1, ...]
        target_tokens: Target interleaved tokens for comparison
        scenario: One of 'primed', 'cold_start', 'perturbed'
        perturbation_beat: Beat at which to inject perturbation (default: 17 from paper)
        
    Returns:
        Tuple of (context_tokens, reference_tokens) for the scenario
    """
    batch_size, seq_len = input_tokens.shape
    
    if scenario == 'primed':
        # Start with several beats of ground truth (similar to RLDuet)
        # Use first 8 beats (16 tokens) as context
        context_length = min(16, seq_len)
        context_tokens = input_tokens[:, :context_length].clone()
        reference_tokens = target_tokens.clone()
        
    elif scenario == 'cold_start':
        # True cold start - minimal context to avoid completely invalid input
        # Start with just the first chord token (no melody context)
        context_tokens = input_tokens[:, :1].clone()  # Just first chord
        reference_tokens = target_tokens.clone()
        
    elif scenario == 'perturbed':
        # Start with minimal context like cold start, but apply perturbation during generation
        # Models should generate from the beginning, with melody transposition applied from beat 17 onwards
        context_tokens = input_tokens[:, :1].clone()  # Just first chord, same as cold start
        reference_tokens = target_tokens.clone()
        print(f"Online model: Perturbed scenario will apply melody transposition (+6 semitones) from beat {perturbation_beat} onwards during generation")
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return context_tokens, reference_tokens


def generate_online_temporal(model, dataloader, tokenizer_info: Dict, device: torch.device,
                           scenarios: List[str] = ['primed', 'cold_start', 'perturbed'],
                           max_beats: int = 32, temperature: float = 1.0, top_k: int = 50,
                           perturbation_beat: int = 17) -> Dict:
    """
    Generate sequences step-by-step for online model while tracking harmony quality at each beat.
    
    Args:
        model: Trained online transformer model
        dataloader: DataLoader containing test sequences
        tokenizer_info: Tokenizer information dictionary
        device: Device to run inference on
        scenarios: List of scenarios to evaluate
        max_beats: Maximum number of beats to evaluate
        temperature: Sampling temperature
        top_k: Top-k filtering for sampling
        perturbation_beat: Beat at which to apply perturbation (default: 17 from paper)
        
    Returns:
        Dict with timestep-by-timestep metrics for each scenario
    """
    model.eval()
    
    # Get token ranges for filtering
    chord_token_start = tokenizer_info['chord_token_start']
    total_vocab_size = tokenizer_info['total_vocab_size']
    
    # Initialize results storage
    results = {scenario: {'harmony_scores': [], 'beat_numbers': []} for scenario in scenarios}
    
    print(f"Running temporal evaluation for online model with scenarios: {scenarios}")
    print(f"Perturbation will be applied at beat {perturbation_beat} (melody transposition +6 semitones)")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Temporal Online Generation")):
            # Get data from online dataloader format
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            
            # Debug: Print sequence info for first batch
            if batch_idx == 0:
                print(f"Input tokens shape: {input_tokens.shape}")
                print(f"Target tokens shape: {target_tokens.shape}")
                print(f"Interleaved format: [chord_0, melody_0, chord_1, melody_1, ...]")
            
            # Extract melody from the interleaved input format
            melody_sequences = input_tokens[:, 1::2]  # Every other token starting from index 1
            batch_size, melody_seq_len = melody_sequences.shape
            
            # For interleaved format, the actual sequence length in beats is melody_seq_len
            # But we need to be careful about the relationship between tokens and beats
            max_possible_beats = min(melody_seq_len, input_tokens.shape[1] // 2)
            eval_beats = min(max_beats, max_possible_beats)
            
            if batch_idx == 0:
                print(f"Evaluating {eval_beats} beats (from {max_possible_beats} possible beats, input tokens: {input_tokens.shape[1]})")
                print(f"Melody sequence length: {melody_seq_len}")
                print(f"Required tokens for perturbation at beat {perturbation_beat}: {(perturbation_beat + 1) * 2}")
            
            # Evaluate each scenario
            for scenario in scenarios:
                # Get scenario-specific context
                context_tokens, reference_tokens = extract_scenario_sequences(
                    input_tokens, target_tokens, scenario, perturbation_beat
                )
                
                # Initialize generation sequence
                generated_tokens = context_tokens.clone()
                scenario_harmony_scores = []
                
                # For perturbed scenario, continue applying transposition to new melody tokens
                apply_transposition = (scenario == 'perturbed')
                
                # Generate step by step
                for beat in range(eval_beats):
                    token_idx = beat * 2  # Convert beat to token index (chord position)
                    
                    # For cold start, we need to handle the first beat differently
                    # because we start with only one chord token
                    if scenario == 'cold_start' and beat == 0:
                        # For cold start beat 0, we need to generate the first melody token
                        # to complete the first chord-melody pair
                        if generated_tokens.shape[1] == 1:  # Only have first chord
                            # Get the first melody token from the original sequence
                            if beat < melody_seq_len:
                                melody_token = melody_sequences[:, beat]
                                
                                # Add the melody token to complete the first pair
                                generated_tokens = torch.cat([generated_tokens, melody_token.unsqueeze(1)], dim=1)
                    
                    # Check if we have enough tokens for this beat
                    if token_idx + 1 >= generated_tokens.shape[1]:
                        # We need to generate more tokens
                        if token_idx >= generated_tokens.shape[1]:
                            # Need to generate chord token
                            current_length = generated_tokens.shape[1]
                            model_input = generated_tokens[:, :current_length]
                            
                            # Get model predictions
                            logits = model(model_input)
                            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                            
                            # Filter to chord tokens only
                            chord_logits = next_token_logits[:, chord_token_start:total_vocab_size]
                            
                            # Apply temperature and top-k sampling
                            chord_logits = chord_logits / temperature
                            if top_k > 0:
                                top_k_vals, top_k_indices = torch.topk(chord_logits, 
                                                                     min(top_k, chord_logits.size(-1)), 
                                                                     dim=-1)
                                filtered_logits = torch.full_like(chord_logits, float('-inf'))
                                filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
                                chord_logits = filtered_logits
                            
                            # Sample chord tokens
                            chord_probs = torch.softmax(chord_logits, dim=-1)
                            sampled_indices = torch.multinomial(chord_probs, 1).squeeze(-1)
                            sampled_chord_tokens = sampled_indices + chord_token_start
                            
                            # Add the generated chord token
                            generated_tokens = torch.cat([generated_tokens, sampled_chord_tokens.unsqueeze(1)], dim=1)
                        
                        # Now add the melody token if needed
                        if token_idx + 1 >= generated_tokens.shape[1]:
                            if beat < melody_seq_len:
                                melody_token = melody_sequences[:, beat]
                                
                                # Apply transposition if in perturbed scenario and beyond perturbation point
                                if apply_transposition and beat >= perturbation_beat:
                                    melody_token = transpose_melody_token(melody_token.item(), semitones=6)
                                    melody_token = torch.tensor([melody_token], device=device)
                                
                                # Add the melody token
                                generated_tokens = torch.cat([generated_tokens, melody_token.unsqueeze(1)], dim=1)
                            else:
                                # If we're beyond the melody sequence, break
                                break
                    
                    # Calculate harmony at this beat (now we should have both tokens)
                    if token_idx + 1 < generated_tokens.shape[1]:
                        chord_token = generated_tokens[0, token_idx].item()
                        melody_token = generated_tokens[0, token_idx + 1].item()
                        
                        harmony_score = calculate_harmony_at_timestep(
                            melody_token, chord_token, tokenizer_info
                        )
                        
                        if harmony_score is not None:
                            scenario_harmony_scores.append(harmony_score)
                        else:
                            scenario_harmony_scores.append(0.0)  # Default for silence/invalid
                    else:
                        scenario_harmony_scores.append(0.0)
                
                # Store results for this scenario
                results[scenario]['harmony_scores'].append(scenario_harmony_scores)
                results[scenario]['beat_numbers'] = list(range(len(scenario_harmony_scores)))
            
            # Process all test sequences for comprehensive evaluation
            # Removed batch limit to use entire test set
    
    # Average across all test sequences with improved handling
    for scenario in scenarios:
        if results[scenario]['harmony_scores']:
            # Convert to numpy array and average, handling variable lengths
            scores_list = results[scenario]['harmony_scores']
            if scores_list:
                # Find the minimum length to avoid padding issues
                min_length = min(len(seq) for seq in scores_list)
                if min_length > 0:
                    # Truncate all sequences to minimum length for fair comparison
                    truncated_scores = [seq[:min_length] for seq in scores_list]
                    scores_array = np.array(truncated_scores)
                    results[scenario]['mean_harmony'] = np.mean(scores_array, axis=0)
                    results[scenario]['std_harmony'] = np.std(scores_array, axis=0)
                    results[scenario]['beat_numbers'] = list(range(len(results[scenario]['mean_harmony'])))
                    
                    # Log statistics for debugging
                    print(f"  Online {scenario}: {len(scores_list)} sequences, {min_length} beats, "
                          f"mean harmony: {np.mean(results[scenario]['mean_harmony']):.3f}")
                else:
                    results[scenario]['mean_harmony'] = []
                    results[scenario]['std_harmony'] = []
                    results[scenario]['beat_numbers'] = []
            else:
                results[scenario]['mean_harmony'] = []
                results[scenario]['std_harmony'] = []
                results[scenario]['beat_numbers'] = []
        else:
            results[scenario]['mean_harmony'] = []
            results[scenario]['std_harmony'] = []
            results[scenario]['beat_numbers'] = []
    
    return results


def generate_offline_temporal(model, dataloader, tokenizer_info: Dict, device: torch.device,
                            scenarios: List[str] = ['primed', 'cold_start', 'perturbed'],
                            max_beats: int = 32, temperature: float = 1.0, top_k: int = 50,
                            perturbation_beat: int = 17) -> Dict:
    """
    Generate chord sequences step-by-step for offline model while tracking harmony quality.
    
    Args:
        model: Trained offline transformer model
        dataloader: DataLoader containing test sequences
        tokenizer_info: Tokenizer information dictionary
        device: Device to run inference on
        scenarios: List of scenarios to evaluate
        max_beats: Maximum number of beats to evaluate
        temperature: Sampling temperature
        top_k: Top-k filtering for sampling
        perturbation_beat: Beat at which to apply perturbation (default: 17 from paper)
        
    Returns:
        Dict with timestep-by-timestep metrics for each scenario
    """
    model.eval()
    
    # Get token ranges for filtering
    chord_token_start = tokenizer_info['chord_token_start']
    total_vocab_size = tokenizer_info['total_vocab_size']
    pad_token_id = tokenizer_info.get('pad_token_id', PAD_TOKEN)
    
    # Initialize results storage
    results = {scenario: {'harmony_scores': [], 'beat_numbers': []} for scenario in scenarios}
    
    print(f"Running temporal evaluation for offline model with scenarios: {scenarios}")
    print(f"Perturbation will be applied at beat {perturbation_beat} (melody transposition +6 semitones)")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Temporal Offline Generation")):
            melody_tokens = batch['melody_tokens'].to(device)
            ground_truth_chord_tokens = batch['chord_target'].to(device)
            
            # Debug: Print sequence info for first batch
            if batch_idx == 0:
                print(f"Melody tokens shape: {melody_tokens.shape}")
                print(f"Ground truth chord tokens shape: {ground_truth_chord_tokens.shape}")
                print(f"Direct format: melody[i] paired with chord[i]")
            
            batch_size, seq_length = melody_tokens.shape
            eval_beats = min(max_beats, seq_length)
            
            if batch_idx == 0:
                print(f"Evaluating {eval_beats} beats (from {seq_length} possible beats)")
            
            # Evaluate each scenario
            for scenario in scenarios:
                scenario_harmony_scores = []
                
                # Create melody sequence for this scenario
                melody_for_scenario = melody_tokens.clone()
                
                # Apply perturbation to melody if needed
                if scenario == 'perturbed':
                    # Transpose melody from perturbation beat onwards
                    for beat in range(perturbation_beat, seq_length):
                        if beat < melody_for_scenario.shape[1]:
                            original_token = melody_for_scenario[0, beat].item()
                            transposed_token = transpose_melody_token(original_token, semitones=6)
                            melody_for_scenario[0, beat] = transposed_token
                    print(f"Offline model: Applied melody transposition (+6 semitones) from beat {perturbation_beat} onwards")
                
                # Initialize chord generation based on scenario
                if scenario == 'primed':
                    # Start with first 8 beats of ground truth
                    context_length = min(8, seq_length)
                    generated_chords = ground_truth_chord_tokens[:, :context_length].clone()
                elif scenario == 'cold_start':
                    # Start with PAD token (minimal context)
                    generated_chords = torch.full((batch_size, 1), pad_token_id, device=device)
                elif scenario == 'perturbed':
                    # Start with PAD token like cold start - models should generate from beginning
                    # with melody transposition applied from beat 17 onwards
                    generated_chords = torch.full((batch_size, 1), pad_token_id, device=device)
                else:
                    continue
                
                # Generate step by step
                for beat in range(eval_beats):
                    # If we need to generate this chord
                    if beat >= generated_chords.shape[1]:
                        # Get model predictions using T5 interface
                        logits = model(
                            melody_tokens=melody_for_scenario,
                            chord_tokens=generated_chords
                        )[:, -1, :]  # [batch_size, vocab_size]
                        
                        # Filter to chord tokens only
                        chord_logits = logits[:, chord_token_start:total_vocab_size]
                        
                        # Apply temperature and top-k sampling
                        chord_logits = chord_logits / temperature
                        if top_k > 0:
                            top_k_vals, top_k_indices = torch.topk(chord_logits, 
                                                                 min(top_k, chord_logits.size(-1)), 
                                                                 dim=-1)
                            filtered_logits = torch.full_like(chord_logits, float('-inf'))
                            filtered_logits.scatter_(-1, top_k_indices, top_k_vals)
                            chord_logits = filtered_logits
                        
                        # Sample chord tokens
                        chord_probs = torch.softmax(chord_logits, dim=-1)
                        sampled_indices = torch.multinomial(chord_probs, 1).squeeze(-1)
                        sampled_chord_tokens = sampled_indices + chord_token_start
                        
                        # Append to generated sequence
                        generated_chords = torch.cat([generated_chords, sampled_chord_tokens.unsqueeze(1)], dim=1)
                    
                    # Calculate harmony at this beat
                    if beat < generated_chords.shape[1] and beat < melody_for_scenario.shape[1]:
                        chord_token = generated_chords[0, beat].item()
                        melody_token = melody_for_scenario[0, beat].item()
                        
                        harmony_score = calculate_harmony_at_timestep(
                            melody_token, chord_token, tokenizer_info
                        )
                        
                        if harmony_score is not None:
                            scenario_harmony_scores.append(harmony_score)
                        else:
                            scenario_harmony_scores.append(0.0)
                    else:
                        scenario_harmony_scores.append(0.0)
                
                # Store results for this scenario
                results[scenario]['harmony_scores'].append(scenario_harmony_scores)
                results[scenario]['beat_numbers'] = list(range(len(scenario_harmony_scores)))
            
            # Process all test sequences for comprehensive evaluation
            # Removed batch limit to use entire test set
    
    # Average across all test sequences with improved handling
    for scenario in scenarios:
        if results[scenario]['harmony_scores']:
            # Convert to numpy array and average, handling variable lengths
            scores_list = results[scenario]['harmony_scores']
            if scores_list:
                # Find the minimum length to avoid padding issues
                min_length = min(len(seq) for seq in scores_list)
                if min_length > 0:
                    # Truncate all sequences to minimum length for fair comparison
                    truncated_scores = [seq[:min_length] for seq in scores_list]
                    scores_array = np.array(truncated_scores)
                    results[scenario]['mean_harmony'] = np.mean(scores_array, axis=0)
                    results[scenario]['std_harmony'] = np.std(scores_array, axis=0)
                    results[scenario]['beat_numbers'] = list(range(len(results[scenario]['mean_harmony'])))
                    
                    # Log statistics for debugging
                    print(f"  Offline {scenario}: {len(scores_list)} sequences, {min_length} beats, "
                          f"mean harmony: {np.mean(results[scenario]['mean_harmony']):.3f}")
                else:
                    results[scenario]['mean_harmony'] = []
                    results[scenario]['std_harmony'] = []
                    results[scenario]['beat_numbers'] = []
            else:
                results[scenario]['mean_harmony'] = []
                results[scenario]['std_harmony'] = []
                results[scenario]['beat_numbers'] = []
        else:
            results[scenario]['mean_harmony'] = []
            results[scenario]['std_harmony'] = []
            results[scenario]['beat_numbers'] = []
    
    return results


def calculate_test_set_baseline_temporal(dataloader, tokenizer_info: Dict, max_beats: int = 32) -> Dict:
    """
    Calculate temporal baseline from test set ground truth.
    
    Args:
        dataloader: DataLoader containing test sequences
        tokenizer_info: Tokenizer information dictionary
        max_beats: Maximum number of beats to evaluate
        
    Returns:
        Dict with beat-by-beat harmony scores for ground truth
    """
    harmony_scores = []
    
    print("Calculating test set baseline temporal metrics...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Baseline Calculation")):
        # Fix: batch is a dictionary, not an object with attributes
        if 'input_tokens' in batch:
            # Online format
            input_tokens = batch['input_tokens']
            batch_size, seq_len = input_tokens.shape
            eval_beats = min(max_beats, seq_len // 2)
            
            batch_harmony_scores = []
            for beat in range(eval_beats):
                chord_idx = beat * 2
                melody_idx = beat * 2 + 1
                
                if melody_idx < seq_len:
                    chord_token = input_tokens[0, chord_idx].item()
                    melody_token = input_tokens[0, melody_idx].item()
                    
                    harmony_score = calculate_harmony_at_timestep(
                        melody_token, chord_token, tokenizer_info
                    )
                    
                    if harmony_score is not None:
                        batch_harmony_scores.append(harmony_score)
                    else:
                        batch_harmony_scores.append(0.0)
                else:
                    batch_harmony_scores.append(0.0)
            
            harmony_scores.append(batch_harmony_scores)
        
        # Process all test sequences for comprehensive baseline
        # Removed batch limit to use entire test set
    
    # Average across sequences with improved handling
    if harmony_scores and len(harmony_scores) > 0:
        # Use minimum length approach instead of padding to avoid bias
        min_length = min(len(seq) for seq in harmony_scores)
        if min_length > 0:
            # Truncate all sequences to minimum length for fair comparison
            truncated_scores = [seq[:min_length] for seq in harmony_scores]
            scores_array = np.array(truncated_scores)
            mean_harmony = np.mean(scores_array, axis=0)
            std_harmony = np.std(scores_array, axis=0)
            beat_numbers = list(range(len(mean_harmony)))
            
            # Log statistics for debugging
            print(f"  Baseline: {len(harmony_scores)} sequences, {min_length} beats, "
                  f"mean harmony: {np.mean(mean_harmony):.3f}")
        else:
            mean_harmony = np.array([])
            std_harmony = np.array([])
            beat_numbers = []
    else:
        mean_harmony = np.array([])
        std_harmony = np.array([])
        beat_numbers = []
    
    return {
        'mean_harmony': mean_harmony,
        'std_harmony': std_harmony,
        'beat_numbers': beat_numbers
    }


def log_temporal_metrics(results: Dict, model_name: str, run_name: str, project_name: str = "martydepth-temporal"):
    """
    Log timestep-by-timestep metrics to WandB with comprehensive visualization.
    
    Args:
        results: Results dictionary from temporal evaluation
        model_name: Name of the model (e.g., 'online', 'offline')
        run_name: Name for the WandB run
        project_name: WandB project name
    """
    wandb.init(
        project=project_name,
        name=f"{run_name}_{model_name}",
        job_type="temporal_evaluation",
        reinit=True
    )
    
    # Create comprehensive data table for all scenarios
    table_data = []
    
    # Use a global step counter to avoid conflicts
    global_step_counter = 0
    
    # Log metrics for each scenario
    for scenario_idx, (scenario, data) in enumerate(results.items()):
        if 'mean_harmony' in data and len(data['mean_harmony']) > 0:
            # Log individual timestep metrics with unique step numbers
            for beat, harmony_score in enumerate(data['mean_harmony']):
                std_score = data['std_harmony'][beat] if beat < len(data['std_harmony']) else 0
                
                # Log individual metrics with unique global step
                wandb.log({
                    f"{scenario}/harmony_quality": harmony_score,
                    f"{scenario}/harmony_std": std_score,
                    f"{scenario}/beat": beat,
                    "model": model_name,
                    "scenario": scenario,
                    "beat_in_scenario": beat
                }, step=global_step_counter)
                
                global_step_counter += 1
                
                # Add to table data
                table_data.append([
                    model_name,
                    scenario,
                    beat,
                    harmony_score,
                    std_score,
                    harmony_score - std_score,  # Lower bound
                    harmony_score + std_score   # Upper bound
                ])
            
            # Log scenario summary metrics with safety checks
            if len(data['mean_harmony']) > 0:
                mean_harmony = np.mean(data['mean_harmony'])
                final_harmony = data['mean_harmony'][-1]
                initial_harmony = data['mean_harmony'][0] if len(data['mean_harmony']) > 0 else 0
                harmony_trend = final_harmony - initial_harmony if len(data['mean_harmony']) > 1 else 0
            else:
                mean_harmony = 0.0
                final_harmony = 0.0
                initial_harmony = 0.0
                harmony_trend = 0.0
            
            wandb.log({
                f"{scenario}/mean_harmony": mean_harmony,
                f"{scenario}/final_harmony": final_harmony,
                f"{scenario}/harmony_trend": harmony_trend,
                "model": model_name
            }, step=global_step_counter)
            
            global_step_counter += 1
    
    # Create and log comprehensive data table
    if table_data:
        table = wandb.Table(
            columns=["Model", "Scenario", "Beat", "Harmony", "Std", "Lower_Bound", "Upper_Bound"],
            data=table_data
        )
        wandb.log({"temporal_data": table})
        
        # Create line plots for each scenario
        for scenario in set(row[1] for row in table_data):
            scenario_data = [row for row in table_data if row[1] == scenario]
            if scenario_data:
                beats = [row[2] for row in scenario_data]
                harmony = [row[3] for row in scenario_data]
                
                # Create line plot data
                plot_data = [[beat, harm] for beat, harm in zip(beats, harmony)]
                plot_table = wandb.Table(data=plot_data, columns=["Beat", "Harmony"])
                
                wandb.log({
                    f"{scenario}_line_plot": wandb.plot.line(
                        plot_table, 
                        "Beat", 
                        "Harmony",
                        title=f"Harmony Quality Over Time - {scenario.replace('_', ' ').title()} ({model_name})"
                    )
                })
    
    # Log model metadata
    wandb.log({
        "model_type": model_name,
        "total_scenarios": len(results),
        "evaluation_type": "temporal"
    })
    
    wandb.finish()
    print(f"Logged temporal metrics for {model_name} to WandB with comprehensive visualization")


def create_wandb_comparison_dashboard(online_results: Dict, offline_results: Dict, baseline_results: Dict,
                                    project_name: str = "martydepth-temporal", 
                                    run_name: str = "comparison_dashboard"):
    """
    Create a comprehensive WandB dashboard comparing all models across scenarios.
    
    Args:
        online_results: Temporal results from online model
        offline_results: Temporal results from offline model  
        baseline_results: Temporal results from test set baseline
        project_name: WandB project name
        run_name: Name for the comparison run
    """
    wandb.init(
        project=project_name,
        name=run_name,
        job_type="comparison_dashboard",
        reinit=True
    )
    
    # Prepare comprehensive comparison data
    all_data = []
    
    # Add baseline data
    if 'mean_harmony' in baseline_results:
        for beat, harmony in enumerate(baseline_results['mean_harmony']):
            std = baseline_results['std_harmony'][beat] if beat < len(baseline_results['std_harmony']) else 0
            all_data.append([
                "Baseline", "ground_truth", beat, harmony, std,
                harmony - std, harmony + std
            ])
    
    # Add online model data
    for scenario, data in online_results.items():
        if 'mean_harmony' in data:
            for beat, harmony in enumerate(data['mean_harmony']):
                std = data['std_harmony'][beat] if beat < len(data['std_harmony']) else 0
                all_data.append([
                    "Online", scenario, beat, harmony, std,
                    harmony - std, harmony + std
                ])
    
    # Add offline model data
    for scenario, data in offline_results.items():
        if 'mean_harmony' in data:
            for beat, harmony in enumerate(data['mean_harmony']):
                std = data['std_harmony'][beat] if beat < len(data['std_harmony']) else 0
                all_data.append([
                    "Offline", scenario, beat, harmony, std,
                    harmony - std, harmony + std
                ])
    
    # Create master comparison table
    comparison_table = wandb.Table(
        columns=["Model", "Scenario", "Beat", "Harmony", "Std", "Lower_Bound", "Upper_Bound"],
        data=all_data
    )
    wandb.log({"master_comparison": comparison_table})
    
    # Create scenario-specific comparison plots (simplified without groupby)
    scenarios = ['primed', 'cold_start', 'perturbed']
    
    for scenario in scenarios:
        scenario_data = [row for row in all_data if row[1] == scenario or row[1] == 'ground_truth']
        
        if scenario_data:
            # Create separate line plots for each model in this scenario
            models = set(row[0] for row in scenario_data)
            
            for model in models:
                model_data = [row for row in scenario_data if row[0] == model]
                if model_data:
                    plot_data = []
                    for row in model_data:
                        model_name, scen, beat, harmony, std, lower, upper = row
                        plot_data.append([beat, harmony])
                    
                    plot_table = wandb.Table(data=plot_data, columns=["Beat", "Harmony"])
                    
                    wandb.log({
                        f"{scenario}_{model}_comparison": wandb.plot.line(
                            plot_table,
                            "Beat", 
                            "Harmony",
                            title=f"{model} - {scenario.replace('_', ' ').title()} Scenario"
                        )
                    })
    
    # Create overall performance summary
    summary_data = []
    
    # Calculate summary stats for each model/scenario combination
    for model_name, model_results in [("Online", online_results), ("Offline", offline_results)]:
        for scenario, data in model_results.items():
            if 'mean_harmony' in data and len(data['mean_harmony']) > 0:
                mean_harmony = np.mean(data['mean_harmony'])
                final_harmony = data['mean_harmony'][-1]
                initial_harmony = data['mean_harmony'][0]
                trend = final_harmony - initial_harmony
                
                summary_data.append([
                    model_name, scenario, mean_harmony, final_harmony, 
                    initial_harmony, trend, len(data['mean_harmony'])
                ])
    
    # Add baseline summary
    if 'mean_harmony' in baseline_results and len(baseline_results['mean_harmony']) > 0:
        mean_harmony = np.mean(baseline_results['mean_harmony'])
        final_harmony = baseline_results['mean_harmony'][-1]
        initial_harmony = baseline_results['mean_harmony'][0]
        trend = final_harmony - initial_harmony
        
        summary_data.append([
            "Baseline", "ground_truth", mean_harmony, final_harmony,
            initial_harmony, trend, len(baseline_results['mean_harmony'])
        ])
    
    # Log summary table
    summary_table = wandb.Table(
        columns=["Model", "Scenario", "Mean_Harmony", "Final_Harmony", 
                "Initial_Harmony", "Trend", "Num_Beats"],
        data=summary_data
    )
    wandb.log({"performance_summary": summary_table})
    
    # Create bar chart for mean performance
    if summary_data:
        bar_data = []
        for row in summary_data:
            model, scenario, mean_harm, _, _, _, _ = row
            bar_data.append([f"{model}_{scenario}", mean_harm, model])
        
        bar_table = wandb.Table(data=bar_data, columns=["Model_Scenario", "Mean_Harmony", "Model"])
        wandb.log({
            "mean_performance_comparison": wandb.plot.bar(
                bar_table,
                "Model_Scenario",
                "Mean_Harmony",
                title="Mean Harmony Performance Comparison"
            )
        })
    
    # Log metadata
    wandb.log({
        "comparison_type": "temporal_evaluation",
        "models_compared": ["Online", "Offline", "Baseline"],
        "scenarios_evaluated": scenarios,
        "total_data_points": len(all_data)
    })
    
    wandb.finish()
    print("Created comprehensive WandB comparison dashboard") 


def create_matplotlib_comparison_plots(online_results: Dict, offline_results: Dict, baseline_results: Dict,
                                     save_dir: str = "temporal_plots", show_plots: bool = True) -> List[str]:
    """
    Create matplotlib comparison plots for temporal evaluation results.
    
    Args:
        online_results: Results from online model temporal evaluation
        offline_results: Results from offline model temporal evaluation  
        baseline_results: Results from baseline temporal evaluation
        save_dir: Directory to save the plots
        show_plots: Whether to display plots inline
        
    Returns:
        List of saved plot file paths
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    scenarios = ['primed', 'cold_start', 'perturbed']
    saved_files = []
    
    # Create one plot for each scenario
    for scenario in scenarios:
        plt.figure(figsize=(12, 8))
        
        # Plot baseline (ground truth)
        if 'mean_harmony' in baseline_results and len(baseline_results['mean_harmony']) > 0:
            beats = baseline_results['beat_numbers']
            harmony = baseline_results['mean_harmony']
            std = baseline_results['std_harmony']
            
            plt.plot(beats, harmony, 'k-', linewidth=3, label='Baseline (Ground Truth)', alpha=0.8)
            plt.fill_between(beats, 
                           np.array(harmony) - np.array(std), 
                           np.array(harmony) + np.array(std), 
                           alpha=0.2, color='black')
        
        # Plot online model
        if scenario in online_results and 'mean_harmony' in online_results[scenario]:
            data = online_results[scenario]
            if len(data['mean_harmony']) > 0:
                beats = data['beat_numbers']
                harmony = data['mean_harmony']
                std = data['std_harmony']
                
                plt.plot(beats, harmony, 'b-', linewidth=2.5, label='Online Model', marker='o', markersize=4)
                plt.fill_between(beats, 
                               np.array(harmony) - np.array(std), 
                               np.array(harmony) + np.array(std), 
                               alpha=0.2, color='blue')
        
        # Plot offline model
        if scenario in offline_results and 'mean_harmony' in offline_results[scenario]:
            data = offline_results[scenario]
            if len(data['mean_harmony']) > 0:
                beats = data['beat_numbers']
                harmony = data['mean_harmony']
                std = data['std_harmony']
                
                plt.plot(beats, harmony, 'r-', linewidth=2.5, label='Offline Model', marker='s', markersize=4)
                plt.fill_between(beats, 
                               np.array(harmony) - np.array(std), 
                               np.array(harmony) + np.array(std), 
                               alpha=0.2, color='red')
        
        # Customize plot
        plt.xlabel('Beat', fontsize=14, fontweight='bold')
        plt.ylabel('Harmony Quality', fontsize=14, fontweight='bold')
        plt.title(f'Temporal Harmony Quality - {scenario.replace("_", " ").title()} Scenario', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add perturbation line for perturbed scenario
        if scenario == 'perturbed':
            plt.axvline(x=17, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                       label='Perturbation (Beat 17)')
        
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(0, 1)
        
        # Add scenario-specific annotations
        if scenario == 'primed':
            plt.text(0.02, 0.98, 'Models start with 8 beats\nof ground truth context', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        elif scenario == 'cold_start':
            plt.text(0.02, 0.98, 'Models start with minimal\ncontext and must adapt', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        elif scenario == 'perturbed':
            plt.text(0.02, 0.98, 'Melody transposed +6 semitones\nat beat 17 (tritone shift)', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        filename = f"temporal_comparison_{scenario}.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(str(filepath))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Create summary statistics plot
    plt.figure(figsize=(14, 8))
    
    # Prepare data for summary plot
    models = []
    scenarios_list = []
    mean_harmonies = []
    final_harmonies = []
    trends = []
    
    # Add baseline data
    if 'mean_harmony' in baseline_results and len(baseline_results['mean_harmony']) > 0:
        models.append('Baseline')
        scenarios_list.append('Ground Truth')
        mean_harmonies.append(np.mean(baseline_results['mean_harmony']))
        final_harmonies.append(baseline_results['mean_harmony'][-1])
        trends.append(baseline_results['mean_harmony'][-1] - baseline_results['mean_harmony'][0])
    
    # Add model data
    for model_name, results in [('Online', online_results), ('Offline', offline_results)]:
        for scenario in scenarios:
            if scenario in results and 'mean_harmony' in results[scenario]:
                data = results[scenario]
                if len(data['mean_harmony']) > 0:
                    models.append(model_name)
                    scenarios_list.append(scenario.replace('_', ' ').title())
                    mean_harmonies.append(np.mean(data['mean_harmony']))
                    final_harmonies.append(data['mean_harmony'][-1])
                    trends.append(data['mean_harmony'][-1] - data['mean_harmony'][0])
    
    # Create subplots for summary statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Mean Harmony
    x_labels = [f"{m}\n{s}" for m, s in zip(models, scenarios_list)]
    colors = ['black' if m == 'Baseline' else 'blue' if m == 'Online' else 'red' for m in models]
    
    axes[0].bar(range(len(mean_harmonies)), mean_harmonies, color=colors, alpha=0.7)
    axes[0].set_title('Mean Harmony Quality', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Harmony', fontsize=12)
    axes[0].set_xticks(range(len(x_labels)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Final Harmony
    axes[1].bar(range(len(final_harmonies)), final_harmonies, color=colors, alpha=0.7)
    axes[1].set_title('Final Harmony Quality', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Final Harmony', fontsize=12)
    axes[1].set_xticks(range(len(x_labels)))
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Plot 3: Harmony Trend
    axes[2].bar(range(len(trends)), trends, color=colors, alpha=0.7)
    axes[2].set_title('Harmony Trend (Final - Initial)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Harmony Change', fontsize=12)
    axes[2].set_xticks(range(len(x_labels)))
    axes[2].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.suptitle('Temporal Evaluation Summary Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = "temporal_summary_statistics.png"
    summary_filepath = save_path / summary_filename
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(str(summary_filepath))
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"📊 Created {len(saved_files)} matplotlib plots in '{save_dir}' directory:")
    for filepath in saved_files:
        print(f"   • {Path(filepath).name}")
    
    return saved_files 