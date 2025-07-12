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

from src.evaluation.metrics import check_harmony
from src.config.tokenization_config import SILENCE_TOKEN, CHORD_TOKEN_START, PAD_TOKEN
from src.data.preprocess_frames import MIDITokenizer


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
    
    # Use MIDITokenizer to decode and re-encode
    melody_tokenizer = MIDITokenizer()
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
        transposed_token = melody_tokenizer.encode_token(transposed_midi, is_onset)
        return transposed_token
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
    
    # Get melody pitch using MIDITokenizer
    melody_tokenizer = MIDITokenizer()
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
        # Start with ground truth up to perturbation point
        # Need enough context to reach perturbation beat (beat 17 = token index 35 for melody)
        required_tokens = (perturbation_beat + 1) * 2  # +1 to include the perturbation beat
        context_length = min(required_tokens, seq_len)
        context_tokens = input_tokens[:, :context_length].clone()
        reference_tokens = target_tokens.clone()
        
        # Apply melody transposition perturbation at specified beat
        perturbation_melody_idx = perturbation_beat * 2 + 1  # Convert beat to melody token index
        if perturbation_melody_idx < context_length and perturbation_melody_idx < seq_len:
            # Transpose melody by tritone (6 semitones) as in paper
            original_melody_token = context_tokens[:, perturbation_melody_idx].item()
            transposed_melody_token = transpose_melody_token(original_melody_token, semitones=6)
            context_tokens[:, perturbation_melody_idx] = transposed_melody_token
            print(f"Online model: Applied melody transposition (+6 semitones) at beat {perturbation_beat} (token index {perturbation_melody_idx})")
        else:
            print(f"Warning: Cannot perturb at beat {perturbation_beat} - beyond sequence length (need index {perturbation_melody_idx}, have {context_length})")
    
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
                    
                    if token_idx >= generated_tokens.shape[1]:
                        break
                    
                    # If we need to generate this chord token
                    if token_idx >= context_tokens.shape[1]:
                        # Prepare input for model
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
                        
                        # Get the corresponding melody token with bounds checking
                        if beat < melody_seq_len:
                            melody_token = melody_sequences[:, beat]
                            
                            # Apply transposition if in perturbed scenario and beyond perturbation point
                            if apply_transposition and beat >= perturbation_beat:
                                melody_token = transpose_melody_token(melody_token.item(), semitones=6)
                                melody_token = torch.tensor([melody_token], device=device)
                            
                            # Add the generated chord and corresponding melody
                            new_tokens = torch.stack([
                                sampled_chord_tokens,
                                melody_token
                            ], dim=1)  # [batch_size, 2]
                            generated_tokens = torch.cat([generated_tokens, new_tokens], dim=1)
                        else:
                            # If we're beyond the melody sequence, break
                            break
                    
                    # Calculate harmony at this beat
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
            
            # Limit to a reasonable number of test sequences
            if batch_idx >= 10:  # Evaluate on first 10 batches
                break
    
    # Average across all test sequences
    for scenario in scenarios:
        if results[scenario]['harmony_scores']:
            # Convert to numpy array and average
            scores_array = np.array(results[scenario]['harmony_scores'])
            results[scenario]['mean_harmony'] = np.mean(scores_array, axis=0)
            results[scenario]['std_harmony'] = np.std(scores_array, axis=0)
            results[scenario]['beat_numbers'] = list(range(len(results[scenario]['mean_harmony'])))
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
                    # Start with ground truth up to perturbation beat
                    context_length = min(perturbation_beat, seq_length)
                    generated_chords = ground_truth_chord_tokens[:, :context_length].clone()
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
            
            # Limit to a reasonable number of test sequences
            if batch_idx >= 10:  # Evaluate on first 10 batches
                break
    
    # Average across all test sequences
    for scenario in scenarios:
        if results[scenario]['harmony_scores']:
            # Convert to numpy array and average
            scores_array = np.array(results[scenario]['harmony_scores'])
            results[scenario]['mean_harmony'] = np.mean(scores_array, axis=0)
            results[scenario]['std_harmony'] = np.std(scores_array, axis=0)
            results[scenario]['beat_numbers'] = list(range(len(results[scenario]['mean_harmony'])))
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
        if hasattr(batch, 'input_tokens'):
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
        
        # Limit to reasonable number of sequences
        if batch_idx >= 10:
            break
    
    # Average across sequences
    if harmony_scores:
        scores_array = np.array(harmony_scores)
        mean_harmony = np.mean(scores_array, axis=0)
        std_harmony = np.std(scores_array, axis=0)
        beat_numbers = list(range(len(mean_harmony)))
    else:
        mean_harmony = []
        std_harmony = []
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
    
    # Log metrics for each scenario
    for scenario, data in results.items():
        if 'mean_harmony' in data and len(data['mean_harmony']) > 0:
            # Log individual timestep metrics
            for beat, harmony_score in enumerate(data['mean_harmony']):
                std_score = data['std_harmony'][beat] if beat < len(data['std_harmony']) else 0
                
                # Log individual metrics
                wandb.log({
                    f"{scenario}/harmony_quality": harmony_score,
                    f"{scenario}/harmony_std": std_score,
                    f"{scenario}/beat": beat,
                    "model": model_name,
                    "scenario": scenario,
                    "global_step": beat
                }, step=beat)
                
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
            
            # Log scenario summary metrics
            mean_harmony = np.mean(data['mean_harmony'])
            final_harmony = data['mean_harmony'][-1] if data['mean_harmony'] else 0
            
            wandb.log({
                f"{scenario}/mean_harmony": mean_harmony,
                f"{scenario}/final_harmony": final_harmony,
                f"{scenario}/harmony_trend": final_harmony - data['mean_harmony'][0] if len(data['mean_harmony']) > 1 else 0,
                "model": model_name
            })
    
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
    
    # Create scenario-specific comparison plots
    scenarios = ['primed', 'cold_start', 'perturbed']
    
    for scenario in scenarios:
        scenario_data = [row for row in all_data if row[1] == scenario or row[1] == 'ground_truth']
        
        if scenario_data:
            # Create line plot for this scenario
            plot_data = []
            for row in scenario_data:
                model, scen, beat, harmony, std, lower, upper = row
                plot_data.append([beat, harmony, model])
            
            plot_table = wandb.Table(data=plot_data, columns=["Beat", "Harmony", "Model"])
            
            wandb.log({
                f"{scenario}_comparison": wandb.plot.line(
                    plot_table,
                    "Beat", 
                    "Harmony",
                    groupby="Model",
                    title=f"Model Comparison - {scenario.replace('_', ' ').title()} Scenario"
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