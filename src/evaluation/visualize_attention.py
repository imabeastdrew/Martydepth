#!/usr/bin/env python3
"""
Script to visualize model attention patterns.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch.nn.functional as F

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
        # We need the model to output attention scores
        # This requires the model's forward pass to have an `output_attentions` flag
        # Assuming your OnlineTransformer can do this.
        # You may need to modify your model's forward pass to return attentions.
        # For a standard HuggingFace model, it would look like this:
        # outputs = model(input_ids=melody_tokens, decoder_input_ids=chord_tokens, output_attentions=True)
        # cross_attentions = outputs.cross_attentions  # Tuple of tensors, one for each layer
        
        # As a placeholder, let's assume a function get_attention exists
        try:
            cross_attentions = model.get_attention(melody_tokens, chord_tokens)
        except AttributeError:
            print("Model does not have a 'get_attention' method. Cannot visualize attention.")
            print("Please modify your model to return attention scores from its forward pass.")
            # Create a dummy attention for demonstration purposes
            seq_len_dec = chord_tokens.shape[1]
            seq_len_enc = melody_tokens.shape[1]
            cross_attentions = [torch.rand(melody_tokens.shape[0], model.num_heads, seq_len_dec, seq_len_enc)] * model.num_layers


    # Let's visualize the attention from the last layer for the first item in the batch
    # Attention shape: (batch_size, num_heads, decoder_seq_len, encoder_seq_len)
    attention = cross_attentions[-1][0].cpu().numpy()
    
    # Average attention across all heads
    avg_attention = np.mean(attention, axis=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_attention, cmap='viridis', aspect='auto')

    # Create colorbar
    fig.colorbar(im, ax=ax)
    
    # We need to map token IDs back to human-readable tokens
    melody_id_to_token = {v: k for k, v in tokenizer_info['melody_token_to_id'].items()}
    chord_id_to_token = {v: k for k, v in tokenizer_info['chord_token_to_id'].items()}

    input_tokens = [melody_id_to_token.get(i.item(), '[UNK]') for i in melody_tokens[0]]
    output_tokens = [chord_id_to_token.get(i.item(), '[UNK]') for i in chord_tokens[0]]

    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    
    ax.set_xticklabels(input_tokens, rotation=90)
    ax.set_yticklabels(output_tokens)
    
    ax.set_xlabel("Input Melody Tokens")
    ax.set_ylabel("Predicted Chord Tokens")
    ax.set_title("Cross-Attention between Melody and Chords")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Attention heatmap saved to {save_path}")
    # You could also log this image directly to wandb:
    # wandb.log({"attention_heatmap": wandb.Image(plt)})
    plt.close()

# To run this, you would need a main block like this:
# if __name__ == '__main__':
#     # 1. Load your config
#     # 2. Load your tokenizer_info
#     # 3. Create a dataloader
#     # 4. Load your trained OnlineTransformer model
#     # 5. Call plot_attention_heatmap(...)
#     pass

try:
    import shap
except ImportError:
    print("SHAP library not found. Please install it with 'pip install shap'")
    shap = None

def explain_reward_model_with_shap(model, dataloader, device, tokenizer_info, save_path="shap_waterfall.png"):
    """
    Uses SHAP to explain a single prediction of the contrastive reward model.
    """
    if shap is None:
        return
        
    model.eval()

    # Get a single batch
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].to(device)
    chord_tokens = batch['chord_tokens'].to(device)
    
    # We need a function that takes a subset of tokens and returns the model's output score.
    # For a contrastive model, the "score" is the dot product of the embeddings for a specific pair.
    # Let's explain the score for the first positive pair in the batch.
    
    background_melody = melody_tokens # Use the whole batch as background
    background_chord = chord_tokens
    
    test_melody = melody_tokens[0:1] # The instance to explain
    test_chord = chord_tokens[0:1]

    def f(mel_toks, chord_toks):
        # This function must accept numpy arrays and return a single score
        mel_toks = torch.from_numpy(mel_toks).to(device).long()
        chord_toks = torch.from_numpy(chord_toks).to(device).long()
        
        # We need to handle variable input lengths from SHAP
        # Let's pad them to the required sequence length
        pad_id = 0 # Assuming 0 is a pad token
        
        padded_mel = torch.full((mel_toks.shape[0], model.max_seq_length), pad_id, device=device, dtype=torch.long)
        padded_mel[:, :mel_toks.shape[1]] = mel_toks
        
        padded_chord = torch.full((chord_toks.shape[0], model.max_seq_length), pad_id, device=device, dtype=torch.long)
        padded_chord[:, :chord_toks.shape[1]] = chord_toks

        mel_embed, chord_embed = model(padded_mel, padded_chord)
        
        # Calculate cosine similarity for each pair in the SHAP batch against the test instance
        test_mel_embed, test_chord_embed = model(test_melody.repeat(mel_toks.shape[0], 1), test_chord.repeat(chord_toks.shape[0], 1))
        
        # Score is the dot product of normalized embeddings
        score = torch.sum(F.normalize(mel_embed) * F.normalize(test_chord_embed), dim=1)
        return score.detach().cpu().numpy()

    # To simplify, we'll explain the melody and chord tokens together
    combined_tokens = torch.cat([test_melody, test_chord], dim=1).cpu().numpy()
    
    def combined_f(tokens):
        # SHAP will pass a (batch, seq_len) numpy array
        melody_len = test_melody.shape[1]
        mel_toks = tokens[:, :melody_len]
        chord_toks = tokens[:, melody_len:]
        return f(mel_toks, chord_toks)

    # Use PartitionExplainer
    explainer = shap.PartitionExplainer(combined_f, np.vstack([combined_tokens, combined_tokens])) # Background needs > 1 sample
    
    # Get token names
    melody_id_to_token = {v: k for k, v in tokenizer_info['melody_token_to_id'].items()}
    chord_id_to_token = {v: k for k, v in tokenizer_info['chord_token_to_id'].items()}
    
    feature_names_mel = [melody_id_to_token.get(i.item(), '[UNK]') for i in test_melody[0]]
    feature_names_chord = [f"chord_{chord_id_to_token.get(i.item(), '[UNK]')}" for i in test_chord[0]]
    
    shap_values = explainer(combined_tokens, feature_names=feature_names_mel + feature_names_chord)

    # Create waterfall plot
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"SHAP waterfall plot saved to {save_path}")
    plt.close() 