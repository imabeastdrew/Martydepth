#!/usr/bin/env python3
"""
A collection of functions for visualizing and interpreting models.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from pathlib import Path
import json
import wandb
import tempfile

# Local application imports
from src.data.dataset import create_dataloader
from src.models.contrastive_reward_model import ContrastiveRewardModel
from src.models.online_transformer import OnlineTransformer


# Optional imports
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

# --- Visualization and Inspection Functions ---

def plot_attention_heatmap(model, dataloader, device, tokenizer_info, save_path="attention_heatmap.png"):
    """
    Generates and saves a heatmap of the decoder-encoder attention
    for a single batch from the dataloader.
    """
    model.eval()
    
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].to(device)
    chord_tokens = batch['chord_tokens'].to(device)

    with torch.no_grad():
        try:
            # Assumes model.get_attention(mel, chd) returns cross-attention scores
            cross_attentions = model.get_attention(melody_tokens, chord_tokens)
        except AttributeError:
            print("Model does not have a 'get_attention' method. Cannot visualize attention.")
            # Create a dummy attention for demonstration purposes
            seq_len_dec = chord_tokens.shape[1]
            seq_len_enc = melody_tokens.shape[1]
            num_layers = getattr(model, "num_layers", 1)
            num_heads = getattr(model, "num_heads", 1)
            batch_size = melody_tokens.shape[0]
            dummy_attention = torch.rand(num_layers, batch_size, num_heads, seq_len_dec, seq_len_enc)
            cross_attentions = [layer_attention for layer_attention in dummy_attention]

    attention = cross_attentions[-1][0].cpu().numpy()
    avg_attention = np.mean(attention, axis=0)

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
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].to(device)
    chord_tokens = batch['chord_tokens'].to(device)
    
    test_melody = melody_tokens[0:1]
    test_chord = chord_tokens[0:1]
    # Use a smaller, fixed-size background dataset to prevent memory issues.
    background_melody = melody_tokens[1:6]

    def f(mel_toks):
        mel_toks = torch.from_numpy(mel_toks).to(device).long()
        fixed_chord_toks = test_chord.repeat(mel_toks.shape[0], 1)
        mel_embed, chord_embed = model(mel_toks, fixed_chord_toks)
        score = torch.sum(torch.nn.functional.normalize(mel_embed) * torch.nn.functional.normalize(chord_embed), dim=1)
        return score.detach().cpu().numpy()

    explainer = shap.PartitionExplainer(f, background_melody.cpu().numpy())
    
    # Correctly decode feature names from the tokenizer info
    token_to_note = {int(k): v for k, v in tokenizer_info['token_to_note'].items()}
    feature_names = []
    for token_item in test_melody[0]:
        token_val = token_item.item()
        note_info = token_to_note.get(token_val)
        if note_info is None or note_info[0] == -1:
            feature_names.append("SILENCE")
        else:
            # Format as "MIDI pitch (onset/hold)"
            note_type = "onset" if note_info[2] else "hold"
            feature_names.append(f"MIDI {note_info[0]} ({note_type})")

    shap_values = explainer(test_melody.cpu().numpy())
    # Assign the feature names directly to the Explanation object
    shap_values.feature_names = feature_names

    plt.figure()
    # The plotting function will now find the feature names on the shap_values object
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
    
    if "melody_tokens" in batch and "chord_tokens" in batch:
        melody_tokens = batch['melody_tokens'].to(device)
        chord_tokens = batch['chord_tokens'].to(device)
        y = model(melody_tokens, chord_tokens)
    else:
        print("Model input not recognized for architecture plotting.")
        return

    output_tensor = y[0] if isinstance(y, tuple) else y
    dot = make_dot(output_tensor, params=dict(model.named_parameters()))
    dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
    print(f"Model architecture diagram saved to {save_path}")

def inspect_reward_model(model, dataloader, device, tokenizer_info, num_samples=5):
    """
    Loads samples, compares rewards for good/bad chords, and prints results.
    """
    model.eval()
    print(f"\n--- Inspecting {num_samples} random samples with the reward model ---")

    token_to_note = {int(k): v for k, v in tokenizer_info['token_to_note'].items()}
    token_to_chord = {int(k): v for k, v in tokenizer_info['token_to_chord'].items()}

    def format_note(token_val):
        note_info = token_to_note.get(token_val)
        return "SILENCE" if note_info is None or note_info[0] == -1 else f"MIDI {note_info[0]}"

    def format_chord(token_val):
        chord_info = token_to_chord.get(token_val)
        if chord_info is None or chord_info[0] == -1: return "NO_CHORD"
        root, intervals, inversion, _ = chord_info
        return f"R:{root} Inv:{inversion} {intervals}"

    for i, batch in enumerate(dataloader):
        if i >= num_samples: break
        melody_tokens = batch['melody_tokens'].to(device)
        good_chord_tokens = batch['chord_tokens'].to(device)

        # Create a more robust "bad" chord sequence by random sampling
        chord_vocab_size = tokenizer_info['chord_vocab_size']
        good_chords_list = good_chord_tokens.squeeze(0).tolist()
        bad_chord_list = []
        for true_chord_token in good_chords_list:
            while True:
                # Sample a random token from the chord vocabulary
                random_token = random.randint(0, chord_vocab_size - 1)
                # Ensure it's not the same as the ground truth token
                if random_token != true_chord_token:
                    bad_chord_list.append(random_token)
                    break
        bad_chord_tokens = torch.tensor([bad_chord_list], dtype=torch.long, device=device)

        with torch.no_grad():
            good_score = model.get_reward(melody_tokens, good_chord_tokens).item()
            bad_score = model.get_reward(melody_tokens, bad_chord_tokens).item()

        decoded_melody = [format_note(t.item()) for t in melody_tokens.squeeze(0)]
        decoded_good_chord = [format_chord(t.item()) for t in good_chord_tokens.squeeze(0)]
        decoded_bad_chord = [format_chord(t.item()) for t in bad_chord_tokens.squeeze(0)]

        print(f"\n--- Sample #{i+1} (Song ID: {batch['song_id'][0]}) ---")
        print(f"Reward (Good Chords): {good_score:.4f}")
        print(f"Reward (Bad Chords):  {bad_score:.4f}")
        print("-" * 65)
        print(f"{'Melody':<15} | {'Good Chord':<25} | {'Bad Chord (Shuffled)':<25}")
        print(f"{'-'*15:<15} | {'-'*25:<25} | {'-'*25:<25}")
        for m, gc, bc in zip(decoded_melody, decoded_good_chord, decoded_bad_chord):
            print(f"{m:<15} | {gc:<25} | {bc:<25}")

# --- Model Loading ---

def load_reward_model_from_wandb(artifact_path: str, device: torch.device):
    """
    Loads a contrastive reward model and its configuration from a W&B artifact.
    """
    print(f"Loading model from W&B artifact: {artifact_path}")
    api = wandb.Api()
    
    try: model_artifact = api.artifact(artifact_path, type='model')
    except wandb.errors.CommError as e: raise e
    if not model_artifact: raise ValueError(f"Artifact {artifact_path} not found.")
    print(f"Found model artifact: {model_artifact.name}")

    run = model_artifact.logged_by()
    config = argparse.Namespace(**run.config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        with open(Path(artifact_dir) / "tokenizer_info.json", 'r') as f:
            tokenizer_info = json.load(f)

        config.melody_vocab_size = tokenizer_info['melody_vocab_size']
        config.chord_vocab_size = tokenizer_info['chord_vocab_size']

        model = ContrastiveRewardModel(
            melody_vocab_size=config.melody_vocab_size,
            chord_vocab_size=config.chord_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_length=config.max_seq_length
        ).to(device)
        
        model_path = Path(artifact_dir) / "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    print("Reward model loaded successfully.")
    return model, config, tokenizer_info

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize or inspect a trained model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path for the model.")
    parser.add_argument("data_dir", type=str, help="Directory with processed data.")
    parser.add_argument("--task", type=str, default="inspect_reward",
                        choices=["inspect_reward", "shap", "attention", "architecture"],
                        help="The visualization or inspection task to run.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to inspect.")
    parser.add_argument("--split", type=str, default="valid", help="Data split to use.")
    parser.add_argument("--save_path", type=str, default="visualization.png", help="Path to save the output visualization.")
    
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # For now, this script only supports the reward model, so we load it directly.
    # Future work could add flags for loading different model types.
    model, config, tokenizer_info = load_reward_model_from_wandb(args.artifact_path, device)

    # Create a dataloader for the specified task
    if args.task == "shap":
        # SHAP needs a larger batch for the background dataset
        loader_batch_size = 32
    else:
        # Other tasks inspect one sample at a time
        loader_batch_size = 1
        
    dataloader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=0,
        sequence_length=config.max_seq_length,
        mode='contrastive'
    )

    # --- Task Dispatch ---
    if args.task == "inspect_reward":
        inspect_reward_model(model, dataloader, device, tokenizer_info, args.num_samples)
    elif args.task == "shap":
        explain_reward_model_with_shap(model, dataloader, device, tokenizer_info, args.save_path)
    elif args.task == "attention":
        plot_attention_heatmap(model, dataloader, device, tokenizer_info, args.save_path)
    elif args.task == "architecture":
        plot_model_architecture(model, dataloader, device, args.save_path)
    
    print("\nDone.")