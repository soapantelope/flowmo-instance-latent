"""PCA circle visualization for pose codes.

Encodes all poses from the dataset, fits PCA on the pose codes,
samples points around a circle in PCA space, inverse-transforms
back to the full pose code space, and generates images from those
synthetic pose codes with a fixed instance.
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA

from flowmo import models, train_utils
from infer import build_model_for_inference, load_image, tensor_to_display


def load_all_images(data_root, size=256):
    """Load all images grouped by instance, return dict of instance -> list of (pose_id, tensor)."""
    transform = T.Compose([T.Resize(size), T.CenterCrop((size, size))])
    instances = defaultdict(list)

    for file in sorted(os.listdir(data_root)):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        instance, pose_id = file.rsplit('_', 1)
        pose_id = pose_id.split('.')[0]

        path = os.path.join(data_root, file)
        image = Image.open(path).convert("RGB")
        image = transform(image)
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        instances[instance].append((pose_id, tensor))

    return instances


def encode_all_poses(model, instances, batch_size=16):
    """Encode pose codes for all images. Returns (codes [N, code_length*pose_dim], labels [N], instance_ids [N])."""
    all_codes = []
    all_labels = []
    all_instance_ids = []

    for instance_id, frames in instances.items():
        images = torch.stack([f[1] for f in frames])
        pose_ids = [f[0] for f in frames]

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].cuda()
            with torch.no_grad():
                pose_code = model.encode_pose(batch)  # [B, code_length, pose_dim]
            code_flat = pose_code.flatten(1)  # [B, code_length * pose_dim]
            all_codes.append(code_flat.cpu())
            all_labels.extend(pose_ids[i:i + batch_size])
            all_instance_ids.extend([instance_id] * len(batch))

    return torch.cat(all_codes, dim=0).numpy(), all_labels, all_instance_ids


def generate_from_codes(model, instance_code, pose_codes, image_size, batch_size=8):
    """Generate images from synthetic pose codes and a fixed instance code."""
    config = model.config.eval.sampling
    all_samples = []

    # Use fixed noise for consistency
    z_fixed = torch.randn((1, 3, image_size, image_size)).cuda()

    for i in range(0, len(pose_codes), batch_size):
        batch_pose = pose_codes[i:i + batch_size]  # [B, code_length, pose_dim]
        bs = len(batch_pose)
        inst = instance_code.expand(bs, -1, -1)  # [B, code_length, inst_dim]

        code = torch.cat([inst, batch_pose], dim=-1)
        mask = torch.ones_like(code[..., :1])
        code = torch.cat([code, mask], dim=-1)

        cfg_mask = 0.0
        null_code = code * cfg_mask if config.cfg != 1.0 else None

        z = z_fixed.expand(bs, -1, -1, -1).clone()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            samples = models.rf_sample(
                model, z, code,
                null_code=null_code,
                sample_steps=config.sample_steps,
                cfg=config.cfg,
                schedule=config.schedule,
            )[-1].clip(-1, 1)

        all_samples.append(samples.float().cpu())

    return torch.cat(all_samples, dim=0)


def create_pca_scatter(codes, instance_ids, pca, circle_points_2d, save_path):
    """Plot PCA scatter of all pose codes, colored by instance, with the sampling circle."""
    projected = pca.transform(codes)
    unique_instances = sorted(set(instance_ids))
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, inst in enumerate(unique_instances):
        mask = [iid == inst for iid in instance_ids]
        pts = projected[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(i)], label=inst, alpha=0.6, s=20)

    theta_plot = np.linspace(0, 2 * np.pi, 200)
    radius = np.sqrt((projected ** 2).sum(axis=1)).mean()
    ax.plot(radius * np.cos(theta_plot), radius * np.sin(theta_plot), 'k--', alpha=0.4, label='Sampling circle')
    ax.scatter(circle_points_2d[:, 0], circle_points_2d[:, 1], c='red', marker='x', s=40, zorder=5, label='Sampled points')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Pose Code PCA (top 2 components)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PCA scatter: {save_path}")


def create_circle_visualization(images, angles, save_path):
    """Arrange generated images in a circle layout, sized so images don't overlap."""
    n = len(images)

    # Each image subtends an arc of 2*pi/n. To avoid overlap, the image side
    # length must be at most the chord length between adjacent points:
    #   chord = 2 * R * sin(pi / n)
    # We pick R so that the chord equals a comfortable image size in inches,
    # then derive the figure size from R.
    img_inches = 1.5  # desired image size in inches
    if n > 1:
        R = img_inches / (2 * np.sin(np.pi / n)) * 1.15  # 15% extra breathing room
    else:
        R = img_inches * 2
    margin = img_inches  # space around the circle for labels
    fig_size = 2 * (R + margin)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    lim = R + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')

    half = img_inches / 2
    for img, angle in zip(images, angles):
        x = R * np.cos(angle)
        y = R * np.sin(angle)

        img_np = tensor_to_display(img).transpose(1, 2, 0)
        ax.imshow(img_np, extent=[x - half, x + half, y - half, y + half], zorder=2)

    ax.set_title('Generated images along PCA circle', fontsize=18, pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved circle visualization: {save_path}")


def create_grid_visualization(images, angles, save_path):
    """Arrange generated images in a grid, sorted by angle."""
    n = len(images)
    cols = 12
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, (img, angle) in enumerate(zip(images, angles)):
        axes[i].imshow(tensor_to_display(img).transpose(1, 2, 0))
        axes[i].set_title(f"{np.degrees(angle):.0f}", fontsize=8)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Generated images along PCA circle (sorted by angle)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PCA circle visualization of pose codes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="flowmo/dataset/images")
    parser.add_argument("--instance-image", type=str, required=True,
                        help="Path to the instance image to use for generation")
    parser.add_argument("--output-dir", type=str, default="pca_circle_outputs")
    parser.add_argument("--angle-gap", type=float, default=5.0,
                        help="Angle gap in degrees between samples on the circle")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config)
    model = build_model_for_inference(config)
    model = model.to(args.device).eval()

    state_dict = train_utils.load_state_dict(args.checkpoint)
    if args.use_ema and "model_ema_state_dict" in state_dict:
        print("Using EMA weights")
        model.load_state_dict(state_dict["model_ema_state_dict"])
    else:
        model.load_state_dict(state_dict["model_state_dict"])
    print(f"Loaded checkpoint from step {state_dict.get('total_steps', 'unknown')}")

    # 1. Load and encode all poses
    print("Loading all images...")
    instances = load_all_images(args.data_root, size=config.data.image_size)
    print(f"Found {sum(len(v) for v in instances.values())} images across {len(instances)} instances")

    print("Encoding all pose codes...")
    codes, labels, instance_ids = encode_all_poses(model, instances)
    print(f"Encoded {len(codes)} pose codes of dim {codes.shape[1]}")

    # 2. PCA
    pca = PCA(n_components=2)
    pca.fit(codes)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # 3. Sample points on a circle in PCA space
    projected = pca.transform(codes)
    radius = np.sqrt((projected ** 2).sum(axis=1)).mean()
    print(f"Mean radius in PCA space: {radius:.4f}")

    angles = np.deg2rad(np.arange(0, 360, args.angle_gap))
    circle_2d = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    circle_full = pca.inverse_transform(circle_2d)  # [N_angles, pose_dim]

    # 4. Create PCA scatter plot
    create_pca_scatter(codes, instance_ids, pca, circle_2d,
                       os.path.join(args.output_dir, "pca_scatter.png"))

    # 5. Reshape synthetic pose codes to [N, code_length, pose_dim]
    code_length = config.model.code_length
    pose_dim = config.model.pose_context_dim
    circle_codes = torch.from_numpy(circle_full).float().cuda()
    circle_codes = circle_codes.reshape(-1, code_length, pose_dim)  # [N, code_length, pose_dim]

    # 6. Encode instance
    print(f"Encoding instance from: {args.instance_image}")
    inst_img = load_image(args.instance_image, size=config.data.image_size)
    inst_img = inst_img.unsqueeze(0).to(args.device)
    with torch.no_grad():
        instance_code = model.encode_instance(inst_img)  # [1, code_length, inst_dim]

    # 7. Generate images
    print(f"Generating {len(angles)} images along PCA circle...")
    generated = generate_from_codes(model, instance_code, circle_codes,
                                     config.data.image_size)

    # 8. Visualize
    create_circle_visualization(generated, angles,
                                os.path.join(args.output_dir, "circle.png"))
    create_grid_visualization(generated, angles,
                              os.path.join(args.output_dir, "grid.png"))

    for i, (img, angle) in enumerate(zip(generated, angles)):
        save_path = os.path.join(args.output_dir, f"angle_{np.degrees(angle):06.1f}.png")
        img_np = (tensor_to_display(img).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(save_path)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
