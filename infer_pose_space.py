"""Inference script for visualizing various aspects of the learned pose space


"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import einops

from flowmo import models, train_utils


def build_model_for_inference(config):
    """Build model for inference without distributed training requirements."""
    import tempfile
    from mup import MuReadout, set_base_shapes
    
    models.MUP_ENABLED = config.model.enable_mup
    model_partial = models.FlowMo
    
    shared_kwargs = dict(config=config)
    model = model_partial(
        **shared_kwargs,
        width=config.model.mup_width,
    )
    
    if config.model.enable_mup:
        print("MuP enabled - setting up base shapes...")
        with tempfile.TemporaryDirectory() as log_dir:
            with torch.device("cpu"):
                base_model = model_partial(
                    **shared_kwargs, width=config.model.mup_width
                )
                delta_model = model_partial(
                    **shared_kwargs,
                    width=config.model.mup_width * 4
                    if config.model.mup_width == 1
                    else 1,
                )
                
                bsh_path = os.path.join(log_dir, "inference.bsh")
                set_base_shapes(
                    model, base_model, delta=delta_model, savefile=bsh_path
                )
            
            for module in model.modules():
                if isinstance(module, MuReadout):
                    module.width_mult = lambda: module.weight.infshape.width_mult()
    
    return model


def load_image(path, size=256):
    """Load and preprocess a single image to [-1, 1] range."""
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop((size, size)),
    ])
    image = Image.open(path).convert("RGB")
    image = transform(image)
    image = np.array(image)
    image = (image / 127.5 - 1.0).astype(np.float32)
    # Convert to tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def tensor_to_display(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for display."""
    return ((tensor.clamp(-1, 1) + 1) / 2).cpu().numpy()

@torch.no_grad()
def compute_pairwise_cosine_similarity(latent_codes):
    """
    Computes the NxN cosine similarity matrix for N latent vectors.
    
    Args:
        latent_codes: Tensor of shape [N, ...] containing the encoded poses.
    Returns:
        similarity_matrix: Numpy array of shape [N, N].
    """
    # Flatten everything except the batch dimension to shape [N, D]
    N = latent_codes.shape[0]
    flat_codes = latent_codes.view(N, -1) # [B, 256*4]
    # Assuming `all_pose_codes` is your [N, 1024] tensor of encoded poses
    mean_vector = flat_codes.mean(dim=0)
    print(mean_vector)
    print(f"Mean vector magnitude: {mean_vector.norm().item()}")
    print(f"Average feature value: {mean_vector.abs().mean().item()}")
    print(f"Flattened latent codes shape for similarity computation: {flat_codes.shape}")
    
    # Cosine Similarity = (A dot B) / (||A|| * ||B||)
    # By L2-normalizing the vectors first, the dot product becomes the cosine similarity.
    normalized_codes = F.normalize(flat_codes, p=2, dim=1)
    # normalized_codes = flat_codes
    
    # Efficient pairwise dot product via matrix multiplication: [N, D] @ [D, N] -> [N, N]
    similarity_matrix = torch.mm(normalized_codes, normalized_codes.t())
    
    return similarity_matrix.to(torch.float32).cpu().numpy()


def visualize_similarity_matrix(similarity_matrix, save_path, title="Latent Space Continuity (Pairwise Cosine Similarity)"):
    """
    Plots the NxN similarity matrix as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    
    # We use 'viridis' or 'plasma' as they are perceptually uniform and great for heatmaps.
    # vmin=0, vmax=1 assumes non-negative similarities are the focus, but you can adjust.
    im = plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest', origin='lower')
    
    plt.colorbar(im, label='Cosine Similarity')
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Image Index (Sequential Physics/Angle)', fontsize=12)
    plt.ylabel('Image Index (Sequential Physics/Angle)', fontsize=12)
    
    # Add a diagonal line to help guide the eye
    plt.plot([0, similarity_matrix.shape[0]-1], [0, similarity_matrix.shape[0]-1], 
             color='white', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity heatmap to: {save_path}")

def visualize_latent_trajectory_pca(latent_codes, save_path="pca_latent_trajectory.png"):
    """
    Projects high-dimensional latent codes to 2D using PCA and plots the trajectory.
    
    Args:
        latent_codes: Tensor of shape [N, D] (e.g., [200, 1024])
        save_path: Where to save the resulting plot.
    """
    print("Running PCA on latent codes...")
    
    # 1. Flatten and format the tensor for sklearn
    # Flatten everything except the batch (time) dimension
    N = latent_codes.shape[0]
    # flat_codes = latent_codes.reshape(N, -1)
    flat_codes = einops.rearrange(latent_codes, 'b t f -> b (f t)') # Ensure shape is [N, D]
    
    # Crucial: Cast to float32 and move to CPU/numpy (sklearn doesn't support bfloat16)
    np_codes = flat_codes.to(torch.float32).cpu().numpy()
    
    # 2. Fit PCA and project down to 2 dimensions
    pca = PCA(n_components=2)
    projected_codes = pca.fit_transform(np_codes)
    
    # Check how much variance is explained by the top 2 components
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA Top 2 components explain {var_explained:.2f}% of the variance.")
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    
    # Create an array representing time (frame index) for color-coding
    time_indices = np.arange(N)
    
    # Draw a faint line connecting the points in sequential order to show the path
    plt.plot(projected_codes[:, 0], projected_codes[:, 1], 
             color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Draw the points themselves, colored by time
    scatter = plt.scatter(projected_codes[:, 0], projected_codes[:, 1], 
                          c=time_indices, cmap='viridis', 
                          s=50, edgecolor='white', linewidth=0.5, zorder=2)
    
    # Add labels and a colorbar
    plt.colorbar(scatter, label="Frame Index (Time)")
    plt.title(f"2D PCA Projection of Pose Latent Space\n(Variance Explained: {var_explained:.1f}%)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    
    # Add Start/End annotations to help orient you
    plt.annotate("Start (0)", (projected_codes[0, 0], projected_codes[0, 1]), 
                 xytext=(5, 5), textcoords='offset points', fontweight='bold')
    plt.annotate("End", (projected_codes[-1, 0], projected_codes[-1, 1]), 
                 xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA trajectory plot to: {save_path}")

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="FlowMo Inference Script - Same Instance, Different Poses")
    parser.add_argument("--checkpoint", type=str, default="/viscam/u/panglexi/flowmo-instance-latent/results/flowmo_vae_contrastive_shrink/checkpoints/00105000.pth",
                        help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images",
                        help="Root directory containing images")
    parser.add_argument("--instance", type=str, default="20",
                        help="Instance ID (e.g., '00')")
    parser.add_argument("--output-dir", type=str, default="inference_outputs_pose_space",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EMA model weights (use --no-use-ema to disable)")
    parser.add_argument("--config", type=str, 
                        default="/viscam/u/panglexi/flowmo-instance-latent/results/flowmo_vae_contrastive_shrink/config.yaml",
                        help="Path to config file (use training config for matching architecture)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config and build model
    print(f"Loading config from: {args.config}")
    config = OmegaConf.load(args.config)
    print(f"Model config: mup_width={config.model.mup_width}, patch_size={config.model.patch_size}")
    model = build_model_for_inference(config)
    model = model.to(device)    
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = train_utils.load_state_dict(args.checkpoint)
    
    if args.use_ema and "model_ema_state_dict" in state_dict:
        print("Using EMA model weights")
        model.load_state_dict(state_dict["model_ema_state_dict"])
    else:
        print("Using regular model weights")
        model.load_state_dict(state_dict["model_state_dict"])
    
    print(f"Loaded checkpoint from step {state_dict.get('total_steps', 'unknown')}")
    
    prefix = f"{args.instance}_"

    # Filter to only include images of THAT specific instance
    image_paths = [
        os.path.join(args.data_root, fname) 
        for fname in os.listdir(args.data_root) if fname.startswith(prefix) and fname.endswith(('.png', '.jpg', '.jpeg'))
    ]

    def sort_by_pose(filepath):
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) == 2:
            instance_id, pose_id = parts
            return (int(pose_id), int(instance_id))
        return (0, 0)
    
    # Because YYY is zero-padded (000-199), alphabetical sort == physical angle sort
    image_paths = sorted(image_paths)

    save_path = os.path.join(args.output_dir, "pose_space_similarity.png")
    all_pose_codes = []
    
    print(f"Encoding {len(image_paths)} images...")
    
    # Process in batches to avoid OOM if the sequence is very long
    batch_size = 16 
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        
        # Using your load_image helper from the provided script
        images = [load_image(p, size=config.data.image_size) for p in batch_paths]
        image_batch = torch.stack(images).to(device)
        
        # Encode the pose (using autocast as seen in your interpolation code)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pose_codes = model.encode_pose(image_batch) # shape [B, 256, 4]
            print(f"Shape of pose codes: {pose_codes.shape}")
            all_pose_codes.append(pose_codes)

    # Concatenate all batches into one large tensor [N, ...]
    all_pose_codes = torch.cat(all_pose_codes, dim=0)
    print(f"Extracted latent codes shape: {all_pose_codes.shape}")
    
    # 1. Compute Metric
    similarity_matrix = compute_pairwise_cosine_similarity(all_pose_codes)
    
    # 2. Visualize
    visualize_similarity_matrix(
        similarity_matrix, 
        save_path=save_path,
        title=f"Pose Space Continuity ({len(image_paths)} sequential states)"
    )

    visualize_latent_trajectory_pca(
        all_pose_codes,
        save_path=os.path.join(args.output_dir, "pca_latent_trajectory.png")
    )

if __name__ == "__main__":
    main()