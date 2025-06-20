#!/usr/bin/env python3
"""
A collection of functions for visualizing and interpreting models.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
except ImportError:
    print("SHAP library not found. Please run 'pip install shap' to install it.")
    shap = None

try:
    from torchviz import make_dot
except ImportError:
    print("torchviz not found. Please run 'pip install torchviz' to install it.")
    make_dot = None

def plot_attention_heatmap(model, dataloader, device, tokenizer_info, save_path="attention_heatmap.png"):
    """
    Generates and saves a heatmap of the decoder-encoder attention
    for a single batch from the dataloader.
    """
    model.eval()
    
    # Get a single batch from the dataloader
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].to(device)
    chord_tokens = batch['chord_tokens'].to(device)

    with torch.no_grad():
        # This requires the model's forward pass to return attention scores.
        # We will assume a 'get_attention' method exists or can be added.
        # This is a common pattern for interpretability.
        try:
            # Assumes model.get_attention(mel, chd) returns cross-attention scores
            cross_attentions = model.get_attention(melody_tokens, chord_tokens)
        except AttributeError:
            print("Model does not have a 'get_attention' method. Cannot visualize attention.")
            print("Please modify your model to return attention scores from its forward pass.")
            # Create a dummy attention for demonstration purposes
            seq_len_dec = chord_tokens.shape[1]
            seq_len_enc = melody_tokens.shape[1]
            # Attention shape: [num_layers, batch_size, num_heads, decoder_seq_len, encoder_seq_len]
            num_layers = getattr(model, "num_layers", 1)
            num_heads = getattr(model, "num_heads", 1)
            batch_size = melody_tokens.shape[0]
            dummy_attention = torch.rand(num_layers, batch_size, num_heads, seq_len_dec, seq_len_enc)
            cross_attentions = [layer_attention for layer_attention in dummy_attention]


    # Visualize the attention from the last layer for the first item in the batch
    # Attention shape: (batch_size, num_heads, decoder_seq_len, encoder_seq_len)
    attention = cross_attentions[-1][0].cpu().numpy()
    
    # Average attention across all heads
    avg_attention = np.mean(attention, axis=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(avg_attention, cmap='viridis', aspect='auto')

    fig.colorbar(im, ax=ax)
    
    melody_id_to_token = {v: k for k, v in tokenizer_info['melody_token_to_id'].items()}
    chord_id_to_token = {v: k for k, v in tokenizer_info['chord_token_to_id'].items()}

    input_tokens = [melody_id_to_token.get(i.item(), '[UNK]') for i in melody_tokens[0]]
    output_tokens = [chord_id_to_token.get(i.item(), '[UNK]') for i in chord_tokens[0]]

    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    
    ax.set_xticklabels(input_tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(output_tokens, fontsize=8)
    
    ax.set_xlabel("Input Melody Tokens")
    ax.set_ylabel("Predicted Chord Tokens")
    ax.set_title("Cross-Attention between Melody and Chords (Last Layer)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Attention heatmap saved to {save_path}")
    plt.close()

def explain_reward_model_with_shap(model, dataloader, device, tokenizer_info, save_path="shap_waterfall.png"):
    """
    Uses SHAP to explain a single prediction of the contrastive reward model.
    """
    if shap is None:
        print("Cannot run SHAP explanation because the 'shap' library is not installed.")
        return
        
    model.eval()

    # Get a single batch to use one instance for explaining and the rest as background
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].to(device)
    chord_tokens = batch['chord_tokens'].to(device)
    
    # The instance to explain
    test_melody = melody_tokens[0:1]
    test_chord = chord_tokens[0:1]

    # The background dataset for SHAP to integrate over
    background_melody = melody_tokens[1:]
    background_chord = chord_tokens[1:]

    # A wrapper function for SHAP. It must take a numpy array of tokens
    # and return a numpy array of model output scores.
    def f(mel_toks):
        # SHAP passes a (num_samples, seq_len) numpy array
        mel_toks = torch.from_numpy(mel_toks).to(device).long()
        
        # We need a fixed chord sequence to compare against
        fixed_chord_toks = test_chord.repeat(mel_toks.shape[0], 1)

        mel_embed, chord_embed = model(mel_toks, fixed_chord_toks)
        
        # Score is the cosine similarity
        score = torch.sum(torch.nn.functional.normalize(mel_embed) * torch.nn.functional.normalize(chord_embed), dim=1)
        return score.detach().cpu().numpy()

    # Use SHAP's PartitionExplainer, which is suitable for text/token-based models
    explainer = shap.PartitionExplainer(f, background_melody.cpu().numpy())
    
    # Get human-readable token names for the plot
    melody_id_to_token = {v: k for k, v in tokenizer_info['melody_token_to_id'].items()}
    feature_names = [melody_id_to_token.get(i.item(), '[UNK]') for i in test_melody[0]]
    
    # Calculate SHAP values for our test instance
    shap_values = explainer(test_melody.cpu().numpy(), feature_names=feature_names)

    # Create and save the waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"SHAP waterfall plot saved to {save_path}")
    plt.close()

def plot_model_architecture(model, dataloader, device, save_path="model_architecture.png"):
    """
    Generates and saves a diagram of the model's architecture.
    """
    if make_dot is None:
        print("Cannot plot model architecture because 'torchviz' is not installed.")
        return

    model.eval()
    batch = next(iter(dataloader))
    
    # The visualization requires a forward pass to trace the graph.
    # We need to handle different model input signatures.
    if "melody_tokens" in batch and "chord_tokens" in batch:
        melody_tokens = batch['melody_tokens'].to(device)
        chord_tokens = batch['chord_tokens'].to(device)
        y = model(melody_tokens, chord_tokens)
    else: # Add more conditions for other models if needed
        print("Model input not recognized for architecture plotting.")
        return

    # make_dot can trace the graph from the output tensor back to the inputs.
    # The output 'y' might be a tuple, so we take the first element.
    output_tensor = y[0] if isinstance(y, tuple) else y
    
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    
    # The 'format' argument determines the output file type.
    # 'png' requires graphviz to be installed on the system.
    dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
    print(f"Model architecture diagram saved to {save_path}")

# To use this function, you would create a separate script that:
# 1. Loads your configuration and a trained OnlineTransformer model.
# 2. Creates a dataloader.
# 3. Calls this plot_attention_heatmap function.

# Placeholder for future content
print("Visualization script created.") 