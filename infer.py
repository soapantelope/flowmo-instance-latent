"""Inference script for FlowMo instance-pose disentanglement.

This script:
1. Loads a checkpoint and two images from the same instance with different poses
2. Tests 4 generation scenarios using generate_from_pose_instance:
   - Image A: instance from A, pose from A (reconstruction)
   - Image B: instance from B, pose from B (reconstruction)
   - Swap 1: instance from A, pose from B
   - Swap 2: instance from B, pose from A
3. Creates visualizations comparing generated vs original images
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


def create_visualization(image_a, image_b, generations, instance_id, pose_a_id, pose_b_id, save_path):
    """Create a comprehensive visualization of the inference results.
    
    Args:
        image_a: original image A (instance with pose A)
        image_b: original image B (same instance with pose B)
        generations: dict with keys 'recon_a', 'recon_b', 'a_inst_b_pose', 'b_inst_a_pose'
        instance_id: the instance identifier
        pose_a_id: pose identifier for image A
        pose_b_id: pose identifier for image B
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 0: Original images and their reconstructions
    # Col 0: Original A
    axes[0, 0].imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    axes[0, 0].set_title(f"Original A\n{instance_id}_{pose_a_id}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Col 1: Reconstruction A (instance from A + pose from A)
    axes[0, 1].imshow(tensor_to_display(generations['recon_a']).transpose(1, 2, 0))
    axes[0, 1].set_title(f"Recon A\nInst(A) + Pose(A)", fontsize=12)
    axes[0, 1].axis('off')
    
    # Col 2: Original B
    axes[0, 2].imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    axes[0, 2].set_title(f"Original B\n{instance_id}_{pose_b_id}", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Col 3: Reconstruction B (instance from B + pose from B)
    axes[0, 3].imshow(tensor_to_display(generations['recon_b']).transpose(1, 2, 0))
    axes[0, 3].set_title(f"Recon B\nInst(B) + Pose(B)", fontsize=12)
    axes[0, 3].axis('off')
    
    # Row 1: Swap results
    # Col 0: Reference - Original A again
    axes[1, 0].imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    axes[1, 0].set_title(f"Reference: Orig A\n{instance_id}_{pose_a_id}", fontsize=11)
    axes[1, 0].axis('off')
    
    # Col 1: Swap - Instance from B + Pose from A (should look like A since same instance)
    axes[1, 1].imshow(tensor_to_display(generations['b_inst_a_pose']).transpose(1, 2, 0))
    axes[1, 1].set_title(f"Inst(B) + Pose(A)\n(Should match Orig A)", fontsize=11, color='blue')
    axes[1, 1].axis('off')
    
    # Col 2: Reference - Original B again
    axes[1, 2].imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    axes[1, 2].set_title(f"Reference: Orig B\n{instance_id}_{pose_b_id}", fontsize=11)
    axes[1, 2].axis('off')
    
    # Col 3: Swap - Instance from A + Pose from B (should look like B since same instance)
    axes[1, 3].imshow(tensor_to_display(generations['a_inst_b_pose']).transpose(1, 2, 0))
    axes[1, 3].set_title(f"Inst(A) + Pose(B)\n(Should match Orig B)", fontsize=11, color='blue')
    axes[1, 3].axis('off')
    
    plt.suptitle(f"Instance-Pose Disentanglement Test\nInstance: {instance_id}, Poses: {pose_a_id} & {pose_b_id}", 
                 fontsize=16, y=0.98)
    
    # Add explanation text
    fig.text(0.5, 0.02, 
             "Top row: Originals and their reconstructions | " +
             "Bottom row: Swapped latents (should match originals if instance codes are truly disentangled)",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_interpolation_visualization(image_a, image_b, interpolated_images, instance_id, pose_a_id, pose_b_id, save_path):
    num_steps = len(interpolated_images)
    
    fig, axes = plt.subplots(2, num_steps + 2, figsize=(3 * (num_steps + 2), 6))
    
    axes[0, 0].imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    axes[0, 0].set_title(f"Original A\n(Pose Source)", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    for i in range(1, num_steps + 1):
        axes[0, i].axis('off')
    
    axes[0, num_steps + 1].imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    axes[0, num_steps + 1].set_title(f"Original B\n(Pose Target)", fontsize=10, fontweight='bold')
    axes[0, num_steps + 1].axis('off')
    
    axes[1, 0].imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    axes[1, 0].set_title(f"Original A", fontsize=10, color='green')
    axes[1, 0].axis('off')
    
    for i, img in enumerate(interpolated_images):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        axes[1, i + 1].imshow(tensor_to_display(img[0]).transpose(1, 2, 0))
        axes[1, i + 1].set_title(f"α = {alpha:.2f}", fontsize=10)
        axes[1, i + 1].axis('off')
    
    axes[1, num_steps + 1].imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    axes[1, num_steps + 1].set_title(f"Original B", fontsize=10, color='green')
    axes[1, num_steps + 1].axis('off')
    
    plt.suptitle(f"Pose Interpolation: {instance_id} | Poses: {pose_a_id} → {pose_b_id}\n"
                 f"Instance from A, Pose interpolated from A to B", 
                 fontsize=14, y=0.98)
    
    fig.text(0.5, 0.02, 
             "Bottom row: Generated images with fixed instance (from A) and linearly interpolated pose latent",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_detailed_visualization(image_a, image_b, generations, instance_id, pose_a_id, pose_b_id, save_path):
    """Create a more detailed visualization with separate sections."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.2)
    
    # Section 1: Original Images
    fig.text(0.5, 0.95, "Original Images (Same Instance, Different Poses)", 
             fontsize=14, ha='center', fontweight='bold')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    ax.set_title(f"Image A: {instance_id}_{pose_a_id}", fontsize=12)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    ax.set_title(f"Image B: {instance_id}_{pose_b_id}", fontsize=12)
    ax.axis('off')
    
    # Section 2: Reconstructions
    fig.text(0.5, 0.62, "Reconstructions (Using Own Instance + Own Pose)", 
             fontsize=14, ha='center', fontweight='bold')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(tensor_to_display(generations['recon_a']).transpose(1, 2, 0))
    ax.set_title(f"Recon A\nInst(A) + Pose(A)", fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    ax.set_title("Ground Truth A", fontsize=11, color='green')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    ax.set_title("Ground Truth B", fontsize=11, color='green')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(tensor_to_display(generations['recon_b']).transpose(1, 2, 0))
    ax.set_title(f"Recon B\nInst(B) + Pose(B)", fontsize=11)
    ax.axis('off')
    
    # Section 3: Swaps (Key Test)
    fig.text(0.5, 0.30, "Latent Swaps (Testing Instance Disentanglement)", 
             fontsize=14, ha='center', fontweight='bold')
    
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(tensor_to_display(generations['b_inst_a_pose']).transpose(1, 2, 0))
    ax.set_title(f"Inst(B) + Pose(A)\n→ Should look like A", fontsize=11, color='blue')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(tensor_to_display(image_a).transpose(1, 2, 0))
    ax.set_title("Expected: Orig A", fontsize=11, color='green')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(tensor_to_display(image_b).transpose(1, 2, 0))
    ax.set_title("Expected: Orig B", fontsize=11, color='green')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(tensor_to_display(generations['a_inst_b_pose']).transpose(1, 2, 0))
    ax.set_title(f"Inst(A) + Pose(B)\n→ Should look like B", fontsize=11, color='blue')
    ax.axis('off')
    
    plt.suptitle(f"Instance: {instance_id} | Poses: {pose_a_id}, {pose_b_id}", 
                 fontsize=16, y=0.99)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowMo Inference Script - Same Instance, Different Poses")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images",
                        help="Root directory containing images")
    parser.add_argument("--instance", type=str, required=True,
                        help="Instance ID (e.g., '00')")
    parser.add_argument("--pose-a", type=str, required=True,
                        help="First pose ID (e.g., '000')")
    parser.add_argument("--pose-b", type=str, required=True,
                        help="Second pose ID (e.g., '001')")
    parser.add_argument("--output-dir", type=str, default="inference_outputs",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EMA model weights (use --no-use-ema to disable)")
    parser.add_argument("--config", type=str, 
                        default="results/flowmo_instance_pretrain/config.yaml",
                        help="Path to config file (use training config for matching architecture)")
    parser.add_argument("--interpolate", action="store_true",
                        help="Run pose interpolation between the two poses")
    parser.add_argument("--interpolate-steps", type=int, default=5,
                        help="Number of interpolation steps (including endpoints)")
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
    
    # Load the two images
    print(f"\nLoading images for instance {args.instance}...")
    
    filename_a = f"{args.instance}_{args.pose_a}.png"
    path_a = os.path.join(args.data_root, filename_a)
    if not os.path.exists(path_a):
        path_a = os.path.join(args.data_root, f"{args.instance}_{args.pose_a}.jpg")
    
    filename_b = f"{args.instance}_{args.pose_b}.png"
    path_b = os.path.join(args.data_root, filename_b)
    if not os.path.exists(path_b):
        path_b = os.path.join(args.data_root, f"{args.instance}_{args.pose_b}.jpg")
    
    if not os.path.exists(path_a):
        raise FileNotFoundError(f"Could not find image A: {filename_a}")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"Could not find image B: {filename_b}")
    
    image_a = load_image(path_a, size=config.data.image_size)
    image_b = load_image(path_b, size=config.data.image_size)
    
    print(f"  Loaded A: {filename_a}")
    print(f"  Loaded B: {filename_b}")
    
    # Add batch dimension for model input
    image_a_batch = image_a.unsqueeze(0).to(device)
    image_b_batch = image_b.unsqueeze(0).to(device)
    
    print(f"\nImage shape: {image_a_batch.shape}")
    
    # Generate the 4 test cases using generate_from_pose_instance
    print("\nGenerating images...")
    generations = {}
    
    with torch.no_grad():
        # 1. Reconstruction A: pose from A, instance from A
        print("  1. Recon A: pose(A) + instance(A)")
        recon_a = model.generate_from_pose_instance(image_a_batch, image_a_batch)
        generations['recon_a'] = recon_a[0]
        
        # 2. Reconstruction B: pose from B, instance from B
        print("  2. Recon B: pose(B) + instance(B)")
        recon_b = model.generate_from_pose_instance(image_b_batch, image_b_batch)
        generations['recon_b'] = recon_b[0]
        
        # 3. Swap: pose from B, instance from A
        print("  3. Swap: pose(B) + instance(A)")
        a_inst_b_pose = model.generate_from_pose_instance(image_b_batch, image_a_batch)
        generations['a_inst_b_pose'] = a_inst_b_pose[0]
        
        # 4. Swap: pose from A, instance from B
        print("  4. Swap: pose(A) + instance(B)")
        b_inst_a_pose = model.generate_from_pose_instance(image_a_batch, image_b_batch)
        generations['b_inst_a_pose'] = b_inst_a_pose[0]
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    create_visualization(
        image_a, image_b, generations,
        args.instance, args.pose_a, args.pose_b,
        os.path.join(args.output_dir, "inference_summary.png")
    )
    
    create_detailed_visualization(
        image_a, image_b, generations,
        args.instance, args.pose_a, args.pose_b,
        os.path.join(args.output_dir, "inference_detailed.png")
    )
    
    # Save individual generated images
    for name, img in generations.items():
        save_path = os.path.join(args.output_dir, f"{name}.png")
        img_np = (tensor_to_display(img).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(save_path)
        print(f"Saved: {save_path}")
    
    if args.interpolate:
        print(f"\nRunning pose interpolation ({args.interpolate_steps} steps)...")
        with torch.no_grad():
            interpolated_images = model.generate_pose_interpolation(
                instance_image=image_a_batch,
                pose_image_a=image_a_batch,
                pose_image_b=image_b_batch,
                num_steps=args.interpolate_steps,
            )
        
        create_interpolation_visualization(
            image_a, image_b, interpolated_images,
            args.instance, args.pose_a, args.pose_b,
            os.path.join(args.output_dir, "pose_interpolation.png")
        )
        
        for i, img in enumerate(interpolated_images):
            alpha = i / (args.interpolate_steps - 1) if args.interpolate_steps > 1 else 0
            save_path = os.path.join(args.output_dir, f"interp_{i:02d}_alpha_{alpha:.2f}.png")
            img_np = (tensor_to_display(img[0]).transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(img_np).save(save_path)
            print(f"Saved: {save_path}")
    
    print(f"\n✓ All outputs saved to: {args.output_dir}")
    print("\nUsage example:")
    print(f"  python infer.py --checkpoint path/to/ckpt.pth --instance 0 --pose-a 0 --pose-b 199")
    print(f"  python infer.py --checkpoint path/to/ckpt.pth --instance 0 --pose-a 0 --pose-b 199 --interpolate --interpolate-steps 7")
    print("\nKey test: Do 'b_inst_a_pose' and 'a_inst_b_pose' match originals A and B?")
    print("If yes → instance codes are properly disentangled from pose!")


if __name__ == "__main__":
    main()