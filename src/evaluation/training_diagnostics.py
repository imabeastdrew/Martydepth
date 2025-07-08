#!/usr/bin/env python3
"""
Training Diagnostics: Evaluate model quality without full inference
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

def analyze_prediction_diversity(model, dataloader, tokenizer_info, device, num_batches=10):
    """Analyze how diverse the model's predictions are across batches."""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    chord_token_start = tokenizer_info['melody_vocab_size'] + 1
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            
            # Get model predictions
            logits = model(input_tokens)
            
            # Focus on chord prediction positions (even indices)
            chord_positions = torch.arange(0, logits.size(1), 2, device=device)
            chord_logits = logits[:, chord_positions, chord_token_start:]
            
            # Get predictions and probabilities
            probs = F.softmax(chord_logits, dim=-1)
            predictions = torch.argmax(chord_logits, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().flatten().tolist())
            all_probabilities.extend(torch.max(probs, dim=-1)[0].cpu().flatten().tolist())
    
    # Calculate diversity metrics
    prediction_counts = Counter(all_predictions)
    total_predictions = len(all_predictions)
    
    # Entropy of prediction distribution
    prediction_probs = np.array(list(prediction_counts.values())) / total_predictions
    prediction_entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10))
    
    # Top prediction dominance
    most_common_token, most_common_count = prediction_counts.most_common(1)[0]
    dominance_ratio = most_common_count / total_predictions
    
    # Average confidence
    avg_confidence = np.mean(all_probabilities)
    
    return {
        'prediction_entropy': prediction_entropy,
        'dominance_ratio': dominance_ratio,
        'most_common_token': most_common_token + chord_token_start,
        'unique_predictions': len(prediction_counts),
        'total_chord_vocab': tokenizer_info['chord_vocab_size'],
        'vocab_coverage': len(prediction_counts) / tokenizer_info['chord_vocab_size'],
        'avg_confidence': avg_confidence,
        'prediction_distribution': dict(prediction_counts)
    }

def analyze_logit_statistics(model, dataloader, tokenizer_info, device, num_batches=5):
    """Analyze logit distributions to detect training issues."""
    model.eval()
    
    logit_stats = {
        'means': [],
        'stds': [],
        'mins': [],
        'maxs': [],
        'nans': 0,
        'infs': 0
    }
    
    chord_token_start = tokenizer_info['melody_vocab_size'] + 1
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            input_tokens = batch['input_tokens'].to(device)
            logits = model(input_tokens)
            
            # Focus on chord prediction logits
            chord_positions = torch.arange(0, logits.size(1), 2, device=device)
            chord_logits = logits[:, chord_positions, chord_token_start:]
            
            # Calculate statistics
            logit_stats['means'].append(chord_logits.mean().item())
            logit_stats['stds'].append(chord_logits.std().item())
            logit_stats['mins'].append(chord_logits.min().item())
            logit_stats['maxs'].append(chord_logits.max().item())
            
            # Check for NaN/Inf
            logit_stats['nans'] += torch.isnan(chord_logits).sum().item()
            logit_stats['infs'] += torch.isinf(chord_logits).sum().item()
    
    return {
        'avg_mean': np.mean(logit_stats['means']),
        'avg_std': np.mean(logit_stats['stds']),
        'min_logit': min(logit_stats['mins']),
        'max_logit': max(logit_stats['maxs']),
        'total_nans': logit_stats['nans'],
        'total_infs': logit_stats['infs'],
        'logit_range': max(logit_stats['maxs']) - min(logit_stats['mins'])
    }

def analyze_attention_patterns(model, dataloader, device, num_batches=3):
    """Analyze attention patterns to see if model is learning structure."""
    model.eval()
    
    attention_stats = {
        'avg_entropy_per_layer': [],
        'avg_max_attention_per_layer': [],
        'position_attention_bias': []
    }
    
    def attention_hook(module, input, output):
        # output is (batch, num_heads, seq_len, seq_len)
        attn_weights = output[1] if isinstance(output, tuple) else output
        if attn_weights is not None:
            # Calculate entropy for each head
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
            attention_stats['layer_entropies'].append(attn_entropy.mean().item())
            
            # Max attention per position
            max_attn = torch.max(attn_weights, dim=-1)[0]
            attention_stats['layer_max_attn'].append(max_attn.mean().item())
    
    # Register hooks on transformer layers
    hooks = []
    attention_stats['layer_entropies'] = []
    attention_stats['layer_max_attn'] = []
    
    for layer in model.transformer.layers:
        hook = layer.self_attn.register_forward_hook(attention_hook)
        hooks.append(hook)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            input_tokens = batch['input_tokens'].to(device)
            _ = model(input_tokens)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if attention_stats['layer_entropies']:
        num_layers = len(model.transformer.layers)
        entries_per_layer = len(attention_stats['layer_entropies']) // num_layers
        
        for i in range(num_layers):
            start_idx = i * entries_per_layer
            end_idx = (i + 1) * entries_per_layer
            layer_entropies = attention_stats['layer_entropies'][start_idx:end_idx]
            layer_max_attn = attention_stats['layer_max_attn'][start_idx:end_idx]
            
            attention_stats['avg_entropy_per_layer'].append(np.mean(layer_entropies))
            attention_stats['avg_max_attention_per_layer'].append(np.mean(layer_max_attn))
    
    return attention_stats

def analyze_gradient_flow(model, dataloader, criterion, device, num_batches=3):
    """Analyze gradient flow to detect vanishing/exploding gradients."""
    model.train()
    
    gradient_stats = {
        'layer_grad_norms': defaultdict(list),
        'total_grad_norm': [],
        'param_update_ratios': defaultdict(list)
    }
    
    # Store initial parameters for update ratio calculation
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        
        # Forward pass
        logits = model(input_tokens)
        loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Calculate gradient norms
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                gradient_stats['layer_grad_norms'][name].append(param_norm.item())
                total_norm += param_norm.item() ** 2
        
        gradient_stats['total_grad_norm'].append(total_norm ** 0.5)
        
        # Calculate parameter update ratios (gradient magnitude / parameter magnitude)
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_magnitude = param.data.norm(2).item()
                grad_magnitude = param.grad.data.norm(2).item()
                if param_magnitude > 0:
                    update_ratio = grad_magnitude / param_magnitude
                    gradient_stats['param_update_ratios'][name].append(update_ratio)
    
    # Summarize results
    summary = {
        'avg_total_grad_norm': np.mean(gradient_stats['total_grad_norm']),
        'max_total_grad_norm': max(gradient_stats['total_grad_norm']),
        'layer_grad_summary': {},
        'param_update_summary': {}
    }
    
    for layer_name, grad_norms in gradient_stats['layer_grad_norms'].items():
        summary['layer_grad_summary'][layer_name] = {
            'mean': np.mean(grad_norms),
            'max': max(grad_norms),
            'min': min(grad_norms)
        }
    
    for param_name, update_ratios in gradient_stats['param_update_ratios'].items():
        summary['param_update_summary'][param_name] = {
            'mean': np.mean(update_ratios),
            'max': max(update_ratios)
        }
    
    return summary

def quick_model_diagnostic(model_path: str, data_dir: str, device_name: str = 'auto'):
    """Run comprehensive model diagnostics without full inference."""
    
    # Setup device
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"Running diagnostics on {device}")
    
    # Load model and tokenizer info
    from src.evaluation.evaluate import load_model_from_wandb
    from src.data.dataset import create_dataloader
    
    try:
        model, tokenizer_info, config = load_model_from_wandb(model_path, device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Create small dataloader for testing
    try:
        dataloader, _ = create_dataloader(
            data_dir=Path(data_dir),
            split="valid",
            batch_size=16,
            num_workers=0,
            sequence_length=64,  # Smaller for faster testing
            mode='online',
            shuffle=False
        )
        print("✓ Dataloader created successfully")
    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        return None
    
    # Run diagnostics
    diagnostics = {}
    
    print("\n1. Analyzing prediction diversity...")
    try:
        diversity_stats = analyze_prediction_diversity(model, dataloader, tokenizer_info, device)
        diagnostics['diversity'] = diversity_stats
        
        print(f"   Prediction entropy: {diversity_stats['prediction_entropy']:.3f}")
        print(f"   Dominance ratio: {diversity_stats['dominance_ratio']:.3f}")
        print(f"   Most common token: {diversity_stats['most_common_token']}")
        print(f"   Vocabulary coverage: {diversity_stats['vocab_coverage']:.3f}")
        print(f"   Average confidence: {diversity_stats['avg_confidence']:.3f}")
        
        # Quick diagnosis
        if diversity_stats['dominance_ratio'] > 0.8:
            print("   ⚠️  HIGH DOMINANCE: Model is overpredicting one token")
        if diversity_stats['vocab_coverage'] < 0.1:
            print("   ⚠️  LOW COVERAGE: Model uses <10% of vocabulary")
        if diversity_stats['avg_confidence'] > 0.9:
            print("   ⚠️  HIGH CONFIDENCE: Model may be overconfident")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n2. Analyzing logit statistics...")
    try:
        logit_stats = analyze_logit_statistics(model, dataloader, tokenizer_info, device)
        diagnostics['logits'] = logit_stats
        
        print(f"   Average logit mean: {logit_stats['avg_mean']:.3f}")
        print(f"   Average logit std: {logit_stats['avg_std']:.3f}")
        print(f"   Logit range: [{logit_stats['min_logit']:.2f}, {logit_stats['max_logit']:.2f}]")
        print(f"   NaN count: {logit_stats['total_nans']}")
        print(f"   Inf count: {logit_stats['total_infs']}")
        
        # Quick diagnosis
        if logit_stats['total_nans'] > 0:
            print("   ⚠️  NAN VALUES: Model has numerical instability")
        if logit_stats['avg_std'] < 0.5:
            print("   ⚠️  LOW VARIANCE: Logits may be too peaked")
        if logit_stats['logit_range'] > 20:
            print("   ⚠️  LARGE RANGE: Logits may be unstable")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n3. Analyzing gradient flow...")
    try:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_info.get('pad_token_id', 177))
        grad_stats = analyze_gradient_flow(model, dataloader, criterion, device)
        diagnostics['gradients'] = grad_stats
        
        print(f"   Average gradient norm: {grad_stats['avg_total_grad_norm']:.4f}")
        print(f"   Max gradient norm: {grad_stats['max_total_grad_norm']:.4f}")
        
        # Check for vanishing/exploding gradients
        if grad_stats['avg_total_grad_norm'] < 0.001:
            print("   ⚠️  VANISHING GRADIENTS: Gradients may be too small")
        if grad_stats['max_total_grad_norm'] > 10:
            print("   ⚠️  EXPLODING GRADIENTS: Gradients may be too large")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n4. Overall Assessment:")
    
    # Combine diagnostics for overall health score
    issues = []
    if 'diversity' in diagnostics:
        if diagnostics['diversity']['dominance_ratio'] > 0.8:
            issues.append("High token dominance")
        if diagnostics['diversity']['vocab_coverage'] < 0.1:
            issues.append("Low vocabulary coverage")
    
    if 'logits' in diagnostics:
        if diagnostics['logits']['total_nans'] > 0:
            issues.append("Numerical instability")
    
    if issues:
        print(f"   ⚠️  Issues detected: {', '.join(issues)}")
        print("   Recommendation: Model needs retraining with fixes")
    else:
        print("   ✓ Model appears healthy")
    
    return diagnostics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick model diagnostics")
    parser.add_argument("--model", type=str, required=True, help="Model artifact path")
    parser.add_argument("--data", type=str, default="data/interim", help="Data directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    diagnostics = quick_model_diagnostic(args.model, args.data, args.device)
    
    if diagnostics:
        # Save results
        output_file = f"model_diagnostics_{args.model.split('/')[-1]}.json"
        with open(output_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"\nDiagnostics saved to {output_file}") 