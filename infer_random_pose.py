"""Inference script for sampling random pose codes with a given instance.

This script:
1. Takes an instance image and encodes it
2. Samples random pose codes from a standard Gaussian distribution
3. Generates images for each random pose
4. Creates a grid visualization

Usage:
    python infer_random_pose.py --checkpoint path/to/ckpt.pth \
        --instance 00 --pose 000 --num-samples 16
"""

import argparse
import math
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


def create_grid(instance_img, generated_images, save_path, cols=4):
    """Create a grid visualization with instance image and generated samples."""
    n_samples = len(generated_images)
    rows = math.ceil((n_samples + 1) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # First cell: original instance image
    axes[0].imshow(tensor_to_display(instance_img).transpose(1, 2, 0))
    axes[0].set_title('Instance\n(original)', fontsize=10, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Remaining cells: generated images with random poses
    for i, img in enumerate(generated_images):
        ax = axes[i + 1]
        ax.imshow(tensor_to_display(img).transpose(1, 2, 0))
        ax.set_title(f'Random #{i+1}', fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n_samples + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Instance + Random Pose Codes (sampled from N(0,1))', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowMo Random Pose Sampling")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images",
                        help="Root directory containing images")
    parser.add_argument("--instance", type=str, required=True,
                        help="Instance ID (e.g., '00')")
    parser.add_argument("--pose", type=str, required=True,
                        help="Pose ID for the instance image (e.g., '000')")
    parser.add_argument("--num-samples", type=int, default=15,
                        help="Number of random pose samples to generate")
    parser.add_argument("--output-dir", type=str, default="inference_random_pose_outputs",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EMA model weights")
    parser.add_argument("--config", type=str, 
                        default="results/flowmo_instance_pretrain/config.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config and build model
    print(f"\nLoading config from: {args.config}")
    config = OmegaConf.load(args.config)
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
    
    # Load instance image
    print(f"\nLoading instance image: {args.instance}_{args.pose}")
    path = get_image_path(args.data_root, args.instance, args.pose)
    instance_img = load_image(path, size=config.data.image_size)
    instance_batch = instance_img.unsqueeze(0).to(device)
    
    # Generate samples with random pose codes
    print(f"\nGenerating {args.num_samples} samples with random pose codes...")
    generated_images = []
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Encode the instance
            instance_code = model.encode_instance(instance_batch)
            
            # Get pose code shape by encoding a dummy image
            pose_code_ref = model.encode_pose(instance_batch)
            pose_shape = pose_code_ref.shape  # (1, code_length, pose_context_dim)
            print(f"Pose code shape: {pose_shape}")
            
            sampling_config = config.eval.sampling
            
            for i in range(args.num_samples):
                # Sample random pose code from standard Gaussian
                random_pose_code = torch.randn(pose_shape, device=device, dtype=pose_code_ref.dtype)
                
                # Combine instance and pose codes
                code = torch.cat([instance_code, random_pose_code], dim=-1)
                
                # Quantize if needed (using model's quantization)
                code, _, _ = model._quantize(code, "noop", deterministic=True)
                
                # Add mask
                mask = torch.ones_like(code[..., :1])
                code_with_mask = torch.cat([code, mask], dim=-1)
                
                # Setup CFG
                cfg_mask = 0.0
                null_code = code_with_mask * cfg_mask if sampling_config.cfg != 1.0 else None
                
                # Sample
                _, _, h, w = instance_batch.shape
                z = torch.randn((1, 3, h, w), device=device)
                
                sample = models.rf_sample(
                    model,
                    z,
                    code_with_mask,
                    null_code=null_code,
                    sample_steps=sampling_config.sample_steps,
                    cfg=sampling_config.cfg,
                    schedule=sampling_config.schedule,
                )[-1].clip(-1, 1)
                
                generated_images.append(sample[0].to(torch.float32))
                print(f"  Generated sample {i+1}/{args.num_samples}")
    
    # Create grid visualization
    print("\nCreating visualization...")
    create_grid(
        instance_img,
        generated_images,
        os.path.join(args.output_dir, "random_pose_grid.png"),
        cols=4
    )
    
    # Save individual images
    print("Saving individual images...")
    for i, img in enumerate(generated_images):
        save_path = os.path.join(args.output_dir, f"random_pose_{i:02d}.png")
        img_np = (tensor_to_display(img).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(save_path)
    
    print(f"\n✓ All outputs saved to: {args.output_dir}")
    print("\nUsage example:")
    print(f"  python infer_random_pose.py --checkpoint path/to/ckpt.pth \\")
    print(f"      --instance 00 --pose 000 --num-samples 16 --seed 42")


if __name__ == "__main__":
    main()

