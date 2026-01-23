"""Inference script for FlowMo instance-pose disentanglement.

This script:
1. Loads a checkpoint and four images (2 instances × 2 poses)
2. Encodes all images to get pose and instance latents
3. Reconstructs images using their own latents
4. Swaps latents to generate all combinations
5. Creates visualizations comparing generated vs original images
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

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


def create_grid_visualization(images_dict, title, save_path):
    """Create a grid visualization of images.
    
    Args:
        images_dict: dict mapping labels to image tensors [C, H, W]
        title: title for the figure
        save_path: path to save the figure
    """
    n = len(images_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (label, img) in enumerate(images_dict.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        # img is [C, H, W], transpose to [H, W, C]
        display_img = tensor_to_display(img).transpose(1, 2, 0)
        ax.imshow(display_img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_swap_matrix_visualization(
    originals, reconstructions, swaps, 
    instances, poses, save_path
):
    """Create a comprehensive visualization showing the swap matrix.
    
    Layout:
    - Row 0: Header with pose labels
    - Col 0: Header with instance labels
    - 2x2 grid showing original images
    - 2x2 grid showing reconstructions
    - 4x4 grid showing all instance-pose combinations
    """
    inst_A, inst_B = instances
    pose_C, pose_D = poses
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create subplots layout
    # Top section: Original images (2x2) and Reconstructions (2x2)
    # Bottom section: Swap matrix (4x4 conceptually, but we show key swaps)
    
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.2)
    
    # Original images (top-left 2x2)
    orig_labels = [
        (f"Orig: {inst_A}_{pose_C}", originals[0]),
        (f"Orig: {inst_A}_{pose_D}", originals[1]),
        (f"Orig: {inst_B}_{pose_C}", originals[2]),
        (f"Orig: {inst_B}_{pose_D}", originals[3]),
    ]
    
    for idx, (label, img) in enumerate(orig_labels):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs[row, col])
        display_img = tensor_to_display(img).transpose(1, 2, 0)
        ax.imshow(display_img)
        ax.set_title(label, fontsize=11)
        ax.axis('off')
    
    # Reconstructions (top-right 2x2)
    recon_labels = [
        (f"Recon: {inst_A}_{pose_C}", reconstructions[0]),
        (f"Recon: {inst_A}_{pose_D}", reconstructions[1]),
        (f"Recon: {inst_B}_{pose_C}", reconstructions[2]),
        (f"Recon: {inst_B}_{pose_D}", reconstructions[3]),
    ]
    
    for idx, (label, img) in enumerate(recon_labels):
        row, col = idx // 2, 2 + idx % 2
        ax = fig.add_subplot(gs[row, col])
        display_img = tensor_to_display(img).transpose(1, 2, 0)
        ax.imshow(display_img)
        ax.set_title(label, fontsize=11)
        ax.axis('off')
    
    # Swap matrix visualization (bottom section)
    # Create a 2x4 grid for swap results
    # Show: instance_from + pose_from -> generated, compared to target original
    
    swap_info = [
        # (generated_img, description, target_original)
        (swaps[(inst_A, pose_C)], f"Inst({inst_A}) + Pose({pose_C})", originals[0]),
        (swaps[(inst_A, pose_D)], f"Inst({inst_A}) + Pose({pose_D})", originals[1]),
        (swaps[(inst_B, pose_C)], f"Inst({inst_B}) + Pose({pose_C})", originals[2]),
        (swaps[(inst_B, pose_D)], f"Inst({inst_B}) + Pose({pose_D})", originals[3]),
    ]
    
    # Add labels
    ax_title = fig.add_subplot(gs[2, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.95, "Latent Swap Results: Instance + Pose → Generated Image", 
                  transform=ax_title.transAxes, fontsize=14, ha='center', va='top',
                  fontweight='bold')
    
    plt.suptitle("Instance-Pose Disentanglement Visualization", fontsize=16, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_detailed_swap_visualization(
    originals, reconstructions, all_generations,
    instances, poses, save_path
):
    """Create a detailed swap matrix visualization.
    
    Shows a matrix where:
    - Rows = instance source images
    - Columns = pose source images  
    - Each cell = generated image using that instance + that pose
    """
    inst_A, inst_B = instances
    pose_C, pose_D = poses
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 22))
    
    # Section 1: Original Images
    fig.text(0.5, 0.96, "Original Images", fontsize=16, ha='center', fontweight='bold')
    
    for idx, (inst, pose) in enumerate([(inst_A, pose_C), (inst_A, pose_D), 
                                         (inst_B, pose_C), (inst_B, pose_D)]):
        ax = fig.add_axes([0.1 + idx * 0.2, 0.82, 0.18, 0.12])
        display_img = tensor_to_display(originals[idx]).transpose(1, 2, 0)
        ax.imshow(display_img)
        ax.set_title(f"{inst}_{pose}", fontsize=10)
        ax.axis('off')
    
    # Section 2: Reconstructions
    fig.text(0.5, 0.78, "Reconstructions (Same Instance + Same Pose)", 
             fontsize=16, ha='center', fontweight='bold')
    
    for idx, (inst, pose) in enumerate([(inst_A, pose_C), (inst_A, pose_D), 
                                         (inst_B, pose_C), (inst_B, pose_D)]):
        ax = fig.add_axes([0.1 + idx * 0.2, 0.64, 0.18, 0.12])
        display_img = tensor_to_display(reconstructions[idx]).transpose(1, 2, 0)
        ax.imshow(display_img)
        ax.set_title(f"Recon: {inst}_{pose}", fontsize=10)
        ax.axis('off')
    
    # Section 3: Swap Matrix
    fig.text(0.5, 0.58, "Swap Matrix: Instance (rows) × Pose (cols)", 
             fontsize=16, ha='center', fontweight='bold')
    
    # Create a 2x2 swap matrix for each instance-pose combination
    # Rows: instances (which image we take instance from)
    # Cols: poses (which image we take pose from)
    
    row_labels = [f"Instance from\n{inst_A}_{pose_C}", f"Instance from\n{inst_A}_{pose_D}",
                  f"Instance from\n{inst_B}_{pose_C}", f"Instance from\n{inst_B}_{pose_D}"]
    col_labels = [f"Pose from {inst_A}_{pose_C}", f"Pose from {inst_A}_{pose_D}",
                  f"Pose from {inst_B}_{pose_C}", f"Pose from {inst_B}_{pose_D}"]
    
    # Full 4x4 matrix showing all combinations
    instance_sources = [(inst_A, pose_C), (inst_A, pose_D), (inst_B, pose_C), (inst_B, pose_D)]
    pose_sources = [(inst_A, pose_C), (inst_A, pose_D), (inst_B, pose_C), (inst_B, pose_D)]
    
    for i, (inst_src, inst_src_pose) in enumerate(instance_sources):
        for j, (pose_src_inst, pose_src) in enumerate(pose_sources):
            ax = fig.add_axes([0.08 + j * 0.22, 0.42 - i * 0.14, 0.18, 0.12])
            
            # Get the generated image for this combination
            key = (inst_src, inst_src_pose, pose_src_inst, pose_src)
            if key in all_generations:
                display_img = tensor_to_display(all_generations[key]).transpose(1, 2, 0)
                ax.imshow(display_img)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=12)
            
            # Labels
            if i == 0:
                ax.set_title(f"Pose: {pose_src_inst}_{pose_src}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Inst: {inst_src}_{inst_src_pose}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_simple_swap_grid(
    originals_dict, reconstructions_dict, swaps_dict,
    instances, poses, save_path
):
    """Create a simple and clear swap visualization.
    
    Shows:
    - Original images
    - Reconstructions
    - Key swaps with clear labels
    """
    inst_A, inst_B = instances
    pose_C, pose_D = poses
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    
    # Row 0: Original images with labels
    axes[0, 0].set_ylabel("ORIGINALS", fontsize=12, fontweight='bold')
    orig_order = [(inst_A, pose_C), (inst_A, pose_D), (inst_B, pose_C), (inst_B, pose_D)]
    for col, (inst, pose) in enumerate(orig_order):
        img = originals_dict[(inst, pose)]
        axes[0, col].imshow(tensor_to_display(img).transpose(1, 2, 0))
        axes[0, col].set_title(f"{inst}_{pose}", fontsize=11)
        axes[0, col].axis('off')
    
    # Row 1: Reconstructions
    axes[1, 0].set_ylabel("RECONSTRUCTIONS", fontsize=12, fontweight='bold')
    for col, (inst, pose) in enumerate(orig_order):
        img = reconstructions_dict[(inst, pose)]
        axes[1, col].imshow(tensor_to_display(img).transpose(1, 2, 0))
        axes[1, col].set_title(f"Recon: {inst}_{pose}", fontsize=11)
        axes[1, col].axis('off')
    
    # Row 2: Swap instance A with all poses
    axes[2, 0].set_ylabel(f"INST {inst_A} +\nDIFF POSES", fontsize=11, fontweight='bold')
    for col, (_, pose_src) in enumerate(orig_order):
        key = (inst_A, pose_src)  # Instance A, varying pose
        if key in swaps_dict:
            img = swaps_dict[key]
            axes[2, col].imshow(tensor_to_display(img).transpose(1, 2, 0))
            axes[2, col].set_title(f"Inst({inst_A}) + Pose({pose_src})", fontsize=9)
        axes[2, col].axis('off')
    
    # Row 3: Swap instance B with all poses
    axes[3, 0].set_ylabel(f"INST {inst_B} +\nDIFF POSES", fontsize=11, fontweight='bold')
    for col, (_, pose_src) in enumerate(orig_order):
        key = (inst_B, pose_src)  # Instance B, varying pose
        if key in swaps_dict:
            img = swaps_dict[key]
            axes[3, col].imshow(tensor_to_display(img).transpose(1, 2, 0))
            axes[3, col].set_title(f"Inst({inst_B}) + Pose({pose_src})", fontsize=9)
        axes[3, col].axis('off')
    
    # Row 4: Cross-swaps comparison with ground truth
    axes[4, 0].set_ylabel("CROSS-SWAPS\n(should match)", fontsize=11, fontweight='bold')
    # Show swaps that should match specific originals
    cross_comparisons = [
        ((inst_A, pose_C), f"Should match\n{inst_A}_{pose_C}"),
        ((inst_A, pose_D), f"Should match\n{inst_A}_{pose_D}"),
        ((inst_B, pose_C), f"Should match\n{inst_B}_{pose_C}"),
        ((inst_B, pose_D), f"Should match\n{inst_B}_{pose_D}"),
    ]
    for col, (key, label) in enumerate(cross_comparisons):
        if key in swaps_dict:
            img = swaps_dict[key]
            axes[4, col].imshow(tensor_to_display(img).transpose(1, 2, 0))
            axes[4, col].set_title(label, fontsize=9)
        axes[4, col].axis('off')
    
    plt.suptitle("Instance-Pose Latent Swapping Results", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowMo Inference Script")
    parser.add_argument("--checkpoint", type=str, default="00010000.pth",
                        help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images",
                        help="Root directory containing images")
    parser.add_argument("--instance-a", type=str, default="00",
                        help="First instance ID")
    parser.add_argument("--instance-b", type=str, default="01",
                        help="Second instance ID")
    parser.add_argument("--pose-c", type=str, default="000",
                        help="First pose ID")
    parser.add_argument("--pose-d", type=str, default="001",
                        help="Second pose ID")
    parser.add_argument("--output-dir", type=str, default="inference_outputs",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EMA model weights (use --no-use-ema to disable)")
    parser.add_argument("--config", type=str, 
                        default="results/flowmo_instance_pretrain/config.yaml",
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
    
    # Define the four images to load
    # Format: {instance}_{pose}.png
    instances = (args.instance_a, args.instance_b)
    poses = (args.pose_c, args.pose_d)
    
    image_specs = [
        (args.instance_a, args.pose_c),  # AC
        (args.instance_a, args.pose_d),  # AD
        (args.instance_b, args.pose_c),  # BC
        (args.instance_b, args.pose_d),  # BD
    ]
    
    # Load images
    print("\nLoading images...")
    images = {}
    for inst, pose in image_specs:
        filename = f"{inst}_{pose}.png"
        path = os.path.join(args.data_root, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, trying .jpg")
            path = os.path.join(args.data_root, f"{inst}_{pose}.jpg")
        
        if os.path.exists(path):
            images[(inst, pose)] = load_image(path, size=config.data.image_size)
            print(f"  Loaded: {filename}")
        else:
            raise FileNotFoundError(f"Could not find image for instance={inst}, pose={pose}")
    
    # Stack images for batch processing
    image_batch = torch.stack([images[k] for k in image_specs]).to(device)
    print(f"\nImage batch shape: {image_batch.shape}")
    
    # Encode all images
    print("\nEncoding images...")
    with torch.no_grad():
        pose_codes = model.encode_pose(image_batch)
        instance_codes = model.encode_instance(image_batch)
    
    print(f"Pose codes shape: {pose_codes.shape}")
    print(f"Instance codes shape: {instance_codes.shape}")
    
    # Store codes by image
    pose_latents = {}
    instance_latents = {}
    for idx, (inst, pose) in enumerate(image_specs):
        pose_latents[(inst, pose)] = pose_codes[idx:idx+1]
        instance_latents[(inst, pose)] = instance_codes[idx:idx+1]
    
    # Step 1: Reconstruct all images using their own latents
    print("\nReconstructing images with own latents...")
    reconstructions = {}
    for inst, pose in image_specs:
        pose_img = images[(inst, pose)].unsqueeze(0).to(device)
        instance_img = images[(inst, pose)].unsqueeze(0).to(device)
        
        recon = model.generate_from_pose_instance(pose_img, instance_img)
        reconstructions[(inst, pose)] = recon[0]
        print(f"  Reconstructed: {inst}_{pose}")
    
    # Step 2: Generate all instance-pose combinations via swapping
    print("\nGenerating swapped combinations...")
    swaps = {}
    
    # For each instance, pair with each pose
    for inst_src, _ in image_specs:
        for _, pose_src in image_specs:
            # Get instance latent from inst_src image and pose latent from pose_src image
            # We need images that have these characteristics
            
            # Find an image with the instance we want
            inst_img_key = None
            for k in image_specs:
                if k[0] == inst_src:
                    inst_img_key = k
                    break
            
            # Find an image with the pose we want
            pose_img_key = None
            for k in image_specs:
                if k[1] == pose_src:
                    pose_img_key = k
                    break
            
            if inst_img_key and pose_img_key:
                instance_img = images[inst_img_key].unsqueeze(0).to(device)
                pose_img = images[pose_img_key].unsqueeze(0).to(device)
                
                generated = model.generate_from_pose_instance(pose_img, instance_img)
                swaps[(inst_src, pose_src)] = generated[0]
                print(f"  Generated: Instance({inst_src}) + Pose({pose_src})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Original images grid
    create_grid_visualization(
        {f"{inst}_{pose}": images[(inst, pose)] for inst, pose in image_specs},
        "Original Images",
        os.path.join(args.output_dir, "1_originals.png")
    )
    
    # 2. Reconstructions grid
    create_grid_visualization(
        {f"Recon {inst}_{pose}": reconstructions[(inst, pose)] for inst, pose in image_specs},
        "Reconstructions (Same Instance + Same Pose Latents)",
        os.path.join(args.output_dir, "2_reconstructions.png")
    )
    
    # 3. Side-by-side comparison: Original vs Reconstruction
    comparison_dict = {}
    for inst, pose in image_specs:
        comparison_dict[f"Orig {inst}_{pose}"] = images[(inst, pose)]
        comparison_dict[f"Recon {inst}_{pose}"] = reconstructions[(inst, pose)]
    
    create_grid_visualization(
        comparison_dict,
        "Original vs Reconstruction Comparison",
        os.path.join(args.output_dir, "3_orig_vs_recon.png")
    )
    
    # 4. Swap results visualization
    swap_display = {}
    for (inst, pose), img in swaps.items():
        swap_display[f"Inst({inst}) + Pose({pose})"] = img
    
    create_grid_visualization(
        swap_display,
        "Latent Swap Results: Instance Latent + Pose Latent → Generated",
        os.path.join(args.output_dir, "4_swap_results.png")
    )
    
    # 5. Comprehensive grid showing swaps vs expected outputs
    create_simple_swap_grid(
        images, reconstructions, swaps,
        instances, poses,
        os.path.join(args.output_dir, "5_comprehensive_grid.png")
    )
    
    # 6. Create a matrix visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Header row for poses
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, "Instance ↓\nPose →", ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    
    for col, (_, pose) in enumerate(image_specs, 1):
        if col <= 4:
            axes[0, col].axis('off')
            axes[0, col].text(0.5, 0.5, f"Pose: {pose}", ha='center', va='center',
                             fontsize=11, fontweight='bold')
    
    # Instance A row
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f"Instance:\n{args.instance_a}", ha='center', va='center',
                    fontsize=11, fontweight='bold')
    
    for col, pose in enumerate([args.pose_c, args.pose_d, args.pose_c, args.pose_d], 1):
        key = (args.instance_a, pose)
        if key in swaps and col <= 4:
            img = tensor_to_display(swaps[key]).transpose(1, 2, 0)
            axes[1, col].imshow(img)
            axes[1, col].axis('off')
    
    # Instance B row  
    axes[2, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, f"Instance:\n{args.instance_b}", ha='center', va='center',
                    fontsize=11, fontweight='bold')
    
    for col, pose in enumerate([args.pose_c, args.pose_d, args.pose_c, args.pose_d], 1):
        key = (args.instance_b, pose)
        if key in swaps and col <= 4:
            img = tensor_to_display(swaps[key]).transpose(1, 2, 0)
            axes[2, col].imshow(img)
            axes[2, col].axis('off')
    
    plt.suptitle("Swap Matrix: Rows = Instance Source, Cols = Pose Source", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "6_swap_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(args.output_dir, '6_swap_matrix.png')}")
    
    # 7. Final summary figure
    fig = plt.figure(figsize=(24, 16))
    
    # Use GridSpec for better layout control
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.15)
    
    # Title
    fig.suptitle("FlowMo Instance-Pose Disentanglement Results", fontsize=18, y=0.98)
    
    # Row 0-1: Originals and Reconstructions side by side
    fig.text(0.02, 0.88, "ORIGINALS", fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.68, "RECONSTRUCTIONS", fontsize=14, fontweight='bold', rotation=90, va='center')
    
    for col, (inst, pose) in enumerate(image_specs):
        # Original
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(tensor_to_display(images[(inst, pose)]).transpose(1, 2, 0))
        ax.set_title(f"{inst}_{pose}", fontsize=12)
        ax.axis('off')
        
        # Reconstruction
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(tensor_to_display(reconstructions[(inst, pose)]).transpose(1, 2, 0))
        ax.set_title(f"Recon", fontsize=10)
        ax.axis('off')
    
    # Row 2: Instance A with all poses
    fig.text(0.02, 0.42, f"INST {args.instance_a}", fontsize=14, fontweight='bold', rotation=90, va='center')
    unique_poses = list(set([p for _, p in image_specs]))
    for col, pose in enumerate(unique_poses):
        key = (args.instance_a, pose)
        if key in swaps:
            ax = fig.add_subplot(gs[2, col])
            ax.imshow(tensor_to_display(swaps[key]).transpose(1, 2, 0))
            ax.set_title(f"+ Pose {pose}", fontsize=10)
            ax.axis('off')
    
    # Add original reference images in rightmost columns for comparison
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(tensor_to_display(images[(args.instance_a, args.pose_c)]).transpose(1, 2, 0))
    ax.set_title(f"GT: {args.instance_a}_{args.pose_c}", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 5])
    ax.imshow(tensor_to_display(images[(args.instance_a, args.pose_d)]).transpose(1, 2, 0))
    ax.set_title(f"GT: {args.instance_a}_{args.pose_d}", fontsize=10)
    ax.axis('off')
    
    # Row 3: Instance B with all poses
    fig.text(0.02, 0.18, f"INST {args.instance_b}", fontsize=14, fontweight='bold', rotation=90, va='center')
    for col, pose in enumerate(unique_poses):
        key = (args.instance_b, pose)
        if key in swaps:
            ax = fig.add_subplot(gs[3, col])
            ax.imshow(tensor_to_display(swaps[key]).transpose(1, 2, 0))
            ax.set_title(f"+ Pose {pose}", fontsize=10)
            ax.axis('off')
    
    # Add original reference images
    ax = fig.add_subplot(gs[3, 4])
    ax.imshow(tensor_to_display(images[(args.instance_b, args.pose_c)]).transpose(1, 2, 0))
    ax.set_title(f"GT: {args.instance_b}_{args.pose_c}", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[3, 5])
    ax.imshow(tensor_to_display(images[(args.instance_b, args.pose_d)]).transpose(1, 2, 0))
    ax.set_title(f"GT: {args.instance_b}_{args.pose_d}", fontsize=10)
    ax.axis('off')
    
    plt.savefig(os.path.join(args.output_dir, "7_final_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(args.output_dir, '7_final_summary.png')}")
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")
    print("\nOutput files:")
    print("  1_originals.png - Original input images")
    print("  2_reconstructions.png - Reconstructions using own latents")
    print("  3_orig_vs_recon.png - Side-by-side comparison")
    print("  4_swap_results.png - All swap combinations")
    print("  5_comprehensive_grid.png - Full grid with labels")
    print("  6_swap_matrix.png - Matrix view of swaps")
    print("  7_final_summary.png - Complete summary figure")


if __name__ == "__main__":
    main()
