"""Inference script for testing instance-pose swapping with 4 images.

This script:
1. Takes 2 instance IDs and 2 pose IDs (4 images total)
2. Encodes all 4 images through both pose and instance encoders
3. Generates all 16 combinations of instance and pose latents
4. Creates a labeled grid visualization

Usage:
    python infer_quad.py --checkpoint path/to/ckpt.pth \
        --instance-1 00 --instance-2 05 \
        --pose-1 000 --pose-2 100
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


def get_image_path(data_root, instance, pose):
    """Get the path to an image given instance and pose IDs."""
    filename = f"{instance}_{pose}.png"
    path = os.path.join(data_root, filename)
    if not os.path.exists(path):
        path = os.path.join(data_root, f"{instance}_{pose}.jpg")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find image: {filename}")
    return path


def create_quad_grid(images, generated, instances, poses, save_path):
    """Create a 4x4 grid showing all instance-pose combinations.
    
    Args:
        images: dict mapping (instance, pose) -> tensor for the 4 input images
        generated: dict mapping (instance_src, pose_src) -> tensor for all 16 combinations
        instances: list of 2 instance IDs
        poses: list of 2 pose IDs
        save_path: path to save the figure
    """
    # Create figure with 5x5 grid (1 header row + 4 data rows, 1 header col + 4 data cols)
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))
    
    # All 4 input image keys
    input_keys = [(inst, pose) for inst in instances for pose in poses]
    
    # Header row: show pose sources (the 4 input images providing poses)
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, 'Instance↓\nPose→', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    
    for col, (inst, pose) in enumerate(input_keys):
        ax = axes[0, col + 1]
        ax.imshow(tensor_to_display(images[(inst, pose)]).transpose(1, 2, 0))
        ax.set_title(f'Pose from\n{inst}_{pose}', fontsize=10, fontweight='bold', color='blue')
        ax.axis('off')
    
    # Data rows: each row is an instance source
    for row, (inst_src, pose_src) in enumerate(input_keys):
        # Left column: instance source image
        ax = axes[row + 1, 0]
        ax.imshow(tensor_to_display(images[(inst_src, pose_src)]).transpose(1, 2, 0))
        ax.set_ylabel(f'Instance from\n{inst_src}_{pose_src}', fontsize=10, fontweight='bold', 
                      color='green', rotation=0, labelpad=60)
        ax.axis('off')
        
        # Generate columns: combine this instance with each pose source
        for col, (pose_inst, pose_pose) in enumerate(input_keys):
            ax = axes[row + 1, col + 1]
            gen_img = generated[(inst_src, pose_src, pose_inst, pose_pose)]
            ax.imshow(tensor_to_display(gen_img).transpose(1, 2, 0))
            
            # Highlight diagonal (reconstructions) vs off-diagonal (swaps)
            if (inst_src, pose_src) == (pose_inst, pose_pose):
                ax.patch.set_edgecolor('gold')
                ax.patch.set_linewidth(4)
                ax.set_title('Recon', fontsize=8, color='gold')
            else:
                ax.set_title(f'I:{inst_src}_{pose_src[:1]}\nP:{pose_inst}_{pose_pose[:1]}', 
                            fontsize=7, color='gray')
            ax.axis('off')
    
    plt.suptitle(
        f'Instance-Pose Disentanglement Grid\n'
        f'Instances: {instances[0]}, {instances[1]} | Poses: {poses[0]}, {poses[1]}\n'
        f'Rows=Instance source (green) | Cols=Pose source (blue) | Gold=Reconstruction',
        fontsize=14, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_simple_grid(images, generated, instances, poses, save_path):
    """Create a simpler 4x4 grid with clear labels.
    
    Rows: Instance source (which image provides the instance latent)
    Cols: Pose source (which image provides the pose latent)
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    
    input_keys = [(inst, pose) for inst in instances for pose in poses]
    labels = [f'{inst}_{pose}' for inst, pose in input_keys]
    
    for row, (inst_src, pose_src) in enumerate(input_keys):
        for col, (pose_inst, pose_pose) in enumerate(input_keys):
            ax = axes[row, col]
            gen_img = generated[(inst_src, pose_src, pose_inst, pose_pose)]
            ax.imshow(tensor_to_display(gen_img).transpose(1, 2, 0))
            
            # Diagonal = reconstruction
            if row == col:
                ax.patch.set_edgecolor('lime')
                ax.patch.set_linewidth(4)
            
            ax.axis('off')
    
    # Row labels (left side) - Instance source
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(f'Inst: {label}', fontsize=11, fontweight='bold',
                                 rotation=0, ha='right', labelpad=50)
    
    # Col labels (top) - Pose source
    for col, label in enumerate(labels):
        axes[0, col].set_title(f'Pose: {label}', fontsize=11, fontweight='bold')
    
    plt.suptitle(
        'Instance × Pose Combinations\n'
        'Rows = Instance source | Cols = Pose source | Green border = Reconstruction',
        fontsize=14, y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_diagnostic_grid(images, generated, instances, poses, save_path):
    """Create a diagnostic grid to check if instance/pose are swapped.
    
    This visualization helps identify if the model has swapped instance and pose:
    - If correct: rows should share appearance, cols should share position
    - If swapped: rows would share position, cols would share appearance
    """
    fig = plt.figure(figsize=(18, 20))
    
    input_keys = [(inst, pose) for inst in instances for pose in poses]
    labels = [f'{inst}_{pose}' for inst, pose in input_keys]
    
    # Main grid
    gs = fig.add_gridspec(5, 5, left=0.1, right=0.9, top=0.88, bottom=0.1, 
                          wspace=0.05, hspace=0.15)
    
    # Original images on top row and left column
    axes = [[None for _ in range(5)] for _ in range(5)]
    
    # Top-left corner label
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5, 'Pose →\n\nInstance ↓', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.axis('off')
    axes[0][0] = ax
    
    # Top row: original images as pose sources
    for col, (inst, pose) in enumerate(input_keys):
        ax = fig.add_subplot(gs[0, col + 1])
        ax.imshow(tensor_to_display(images[(inst, pose)]).transpose(1, 2, 0))
        ax.set_title(f'{inst}_{pose}', fontsize=10, fontweight='bold', color='#0066cc')
        ax.axis('off')
        axes[0][col + 1] = ax
    
    # Left column: original images as instance sources
    for row, (inst, pose) in enumerate(input_keys):
        ax = fig.add_subplot(gs[row + 1, 0])
        ax.imshow(tensor_to_display(images[(inst, pose)]).transpose(1, 2, 0))
        ax.set_ylabel(f'{inst}_{pose}', fontsize=10, fontweight='bold', color='#006600',
                     rotation=0, ha='right', labelpad=10)
        ax.axis('off')
        axes[row + 1][0] = ax
    
    # Main 4x4 grid
    for row, (inst_src, pose_src) in enumerate(input_keys):
        for col, (pose_inst, pose_pose) in enumerate(input_keys):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            gen_img = generated[(inst_src, pose_src, pose_inst, pose_pose)]
            ax.imshow(tensor_to_display(gen_img).transpose(1, 2, 0))
            
            # Mark diagonals
            if row == col:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FFD700')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            ax.set_xticks([])
            ax.set_yticks([])
            axes[row + 1][col + 1] = ax
    
    # Title and explanation
    fig.suptitle(
        'Instance-Pose Disentanglement Diagnostic\n'
        f'Instances: {instances[0]}, {instances[1]} | Poses: {poses[0]}, {poses[1]}',
        fontsize=16, fontweight='bold', y=0.96
    )
    
    # Explanation text
    explanation = (
        'How to interpret this grid:\n'
        '• Blue headers (top row): Original images providing POSE latent\n'
        '• Green headers (left column): Original images providing INSTANCE latent\n'
        '• Gold borders: Reconstructions (same source for both latents)\n\n'
        'If instance/pose are CORRECT:\n'
        '  → Same ROW should have same appearance (color, shape)\n'
        '  → Same COLUMN should have same position\n\n'
        'If instance/pose are SWAPPED:\n'
        '  → Same ROW would have same position\n'
        '  → Same COLUMN would have same appearance'
    )
    fig.text(0.5, 0.03, explanation, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowMo Quad Inference - 4 Images, All Combinations")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images",
                        help="Root directory containing images")
    parser.add_argument("--instance-1", type=str, required=True,
                        help="First instance ID (e.g., '00')")
    parser.add_argument("--instance-2", type=str, required=True,
                        help="Second instance ID (e.g., '05')")
    parser.add_argument("--pose-1", type=str, required=True,
                        help="First pose ID (e.g., '000')")
    parser.add_argument("--pose-2", type=str, required=True,
                        help="Second pose ID (e.g., '100')")
    parser.add_argument("--output-dir", type=str, default="inference_quad_outputs",
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
    
    # Parse instance and pose IDs
    instances = [args.instance_1, args.instance_2]
    poses = [args.pose_1, args.pose_2]
    
    print(f"\nInstances: {instances}")
    print(f"Poses: {poses}")
    
    # Load config and build model
    print(f"\nLoading config from: {args.config}")
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
    
    # Load all 4 images
    print(f"\nLoading images...")
    images = {}
    for inst in instances:
        for pose in poses:
            path = get_image_path(args.data_root, inst, pose)
            images[(inst, pose)] = load_image(path, size=config.data.image_size)
            print(f"  Loaded: {inst}_{pose}.png")
    
    # Create input keys for all 4 images
    input_keys = [(inst, pose) for inst in instances for pose in poses]
    
    print(f"\nEncoding images and generating all {len(input_keys)**2} combinations...")
    
    # Generate all combinations
    generated = {}
    with torch.no_grad():
        for inst_src, pose_src in input_keys:
            # This image provides the instance latent
            instance_img = images[(inst_src, pose_src)].unsqueeze(0).to(device)
            
            for pose_inst, pose_pose in input_keys:
                # This image provides the pose latent
                pose_img = images[(pose_inst, pose_pose)].unsqueeze(0).to(device)
                
                # Generate: instance from (inst_src, pose_src), pose from (pose_inst, pose_pose)
                result = model.generate_from_pose_instance(pose_img, instance_img)
                generated[(inst_src, pose_src, pose_inst, pose_pose)] = result[0]
                
                print(f"  Generated: Instance({inst_src}_{pose_src}) + Pose({pose_inst}_{pose_pose})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    create_diagnostic_grid(
        images, generated, instances, poses,
        os.path.join(args.output_dir, "quad_diagnostic.png")
    )
    
    create_simple_grid(
        images, generated, instances, poses,
        os.path.join(args.output_dir, "quad_simple.png")
    )
    
    create_quad_grid(
        images, generated, instances, poses,
        os.path.join(args.output_dir, "quad_full.png")
    )
    
    # Save individual generated images
    print("\nSaving individual images...")
    for key, img in generated.items():
        inst_src, pose_src, pose_inst, pose_pose = key
        filename = f"inst_{inst_src}_{pose_src}_pose_{pose_inst}_{pose_pose}.png"
        save_path = os.path.join(args.output_dir, filename)
        img_np = (tensor_to_display(img).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(save_path)
    print(f"  Saved {len(generated)} individual images")
    
    print(f"\n✓ All outputs saved to: {args.output_dir}")
    print("\nUsage example:")
    print(f"  python infer_quad.py --checkpoint path/to/ckpt.pth \\")
    print(f"      --instance-1 00 --instance-2 05 \\")
    print(f"      --pose-1 000 --pose-2 100")
    print("\nHow to interpret quad_diagnostic.png:")
    print("  • If INSTANCE encoder captures appearance: same ROW = same color/shape")
    print("  • If POSE encoder captures position: same COLUMN = same position")
    print("  • If they're swapped, the pattern will be reversed!")


if __name__ == "__main__":
    main()
