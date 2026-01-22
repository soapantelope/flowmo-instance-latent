#!/usr/bin/env python3
"""
Pendulum Dataset Generator

Generates a dataset of primitive pendulum-like images with:
- Body: rectangular frame with variable height and width
- String: connects body to weight, variable length and thickness
- Weight: circle, square, or equilateral triangle with random rotation

All attributes are configurable.
"""

import os
import math
import random
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm


@dataclass
class PendulumConfig:
    """Configuration for pendulum dataset generation."""
    
    # Image settings
    image_size: int = 256
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White background
    
    # Body settings (rectangular frame)
    body_width_range: Tuple[int, int] = (60, 120)  # Min and max width
    body_height_range: Tuple[int, int] = (140, 220)  # Min and max height
    body_border_width: int = 3  # Border thickness of the body frame
    
    # String settings
    string_length_range: Tuple[float, float] = (0.3, 0.9)  # As fraction of max possible length
    string_thickness_range: Tuple[int, int] = (2, 3)  # Pixel thickness
    string_color: Tuple[int, int, int] = (50, 50, 50)  # Dark gray
    
    # Angle settings (degrees from vertical)
    angle_range: Tuple[float, float] = (-90.0, 90.0)
    
    # Weight settings
    weight_size_range: Tuple[int, int] = (15, 25)  # Radius or half-side length
    weight_shapes: List[str] = field(default_factory=lambda: ["circle", "square", "triangle"])
    
    # Dataset settings
    num_images: int = 10000
    output_dir: str = "dataset"
    save_metadata: bool = True
    seed: Optional[int] = 42


def get_random_color(exclude_color: Optional[Tuple[int, int, int]] = None, 
                     min_distance: int = 100) -> Tuple[int, int, int]:
    """Generate a random color, optionally different from an excluded color."""
    max_attempts = 50
    for _ in range(max_attempts):
        color = (random.randint(30, 225), 
                 random.randint(30, 225), 
                 random.randint(30, 225))
        
        if exclude_color is None:
            return color
        
        # Calculate color distance (Euclidean in RGB space)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(color, exclude_color)))
        if distance >= min_distance:
            return color
    
    # Fallback: return complementary-ish color
    if exclude_color:
        return tuple(255 - c for c in exclude_color)
    return color


def draw_equilateral_triangle(draw: ImageDraw.ImageDraw, 
                               center: Tuple[float, float], 
                               size: float, 
                               rotation: float, 
                               fill: Tuple[int, int, int],
                               outline: Optional[Tuple[int, int, int]] = None):
    """Draw an equilateral triangle centered at the given point.
    
    Args:
        draw: PIL ImageDraw object
        center: (x, y) center of the triangle
        size: Distance from center to each vertex
        rotation: Rotation angle in degrees
        fill: Fill color
        outline: Outline color (optional)
    """
    cx, cy = center
    rotation_rad = math.radians(rotation)
    
    # Three vertices, 120 degrees apart
    points = []
    for i in range(3):
        angle = rotation_rad + (2 * math.pi * i / 3)
        x = cx + size * math.sin(angle)
        y = cy - size * math.cos(angle)
        points.append((x, y))
    
    draw.polygon(points, fill=fill, outline=outline)


def draw_rotated_square(draw: ImageDraw.ImageDraw,
                        center: Tuple[float, float],
                        size: float,
                        rotation: float,
                        fill: Tuple[int, int, int],
                        outline: Optional[Tuple[int, int, int]] = None):
    """Draw a square centered at the given point with rotation.
    
    Args:
        draw: PIL ImageDraw object
        center: (x, y) center of the square
        size: Half the side length (distance from center to edge midpoint)
        rotation: Rotation angle in degrees
        fill: Fill color
        outline: Outline color (optional)
    """
    cx, cy = center
    rotation_rad = math.radians(rotation)
    
    # Four vertices, 90 degrees apart, starting from corner
    # Distance from center to corner is size * sqrt(2)
    corner_dist = size * math.sqrt(2)
    
    points = []
    for i in range(4):
        angle = rotation_rad + math.pi/4 + (math.pi * i / 2)
        x = cx + corner_dist * math.cos(angle)
        y = cy + corner_dist * math.sin(angle)
        points.append((x, y))
    
    draw.polygon(points, fill=fill, outline=outline)


def draw_weight(draw: ImageDraw.ImageDraw,
                center: Tuple[float, float],
                size: float,
                shape: str,
                rotation: float,
                fill: Tuple[int, int, int]):
    """Draw a weight shape at the given position.
    
    Args:
        draw: PIL ImageDraw object
        center: (x, y) center of the weight
        size: Size parameter (radius for circle, half-side for square, vertex distance for triangle)
        shape: One of "circle", "square", "triangle"
        rotation: Rotation angle in degrees (for square and triangle)
        fill: Fill color
    """
    cx, cy = center
    
    if shape == "circle":
        bbox = [cx - size, cy - size, cx + size, cy + size]
        draw.ellipse(bbox, fill=fill)
    elif shape == "square":
        draw_rotated_square(draw, center, size * 0.8, rotation, fill)
    elif shape == "triangle":
        draw_equilateral_triangle(draw, center, size, rotation, fill)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def generate_pendulum(config: PendulumConfig) -> Tuple[Image.Image, dict]:
    """Generate a single pendulum image with random attributes.
    
    Returns:
        Tuple of (PIL Image, metadata dict)
    """
    # Create image
    img = Image.new('RGB', (config.image_size, config.image_size), config.background_color)
    draw = ImageDraw.Draw(img)
    
    # Generate random attributes
    body_width = random.randint(*config.body_width_range)
    body_height = random.randint(*config.body_height_range)
    
    # Ensure body fits in image
    body_width = min(body_width, config.image_size - 20)
    body_height = min(body_height, config.image_size - 20)
    
    # Body position (centered horizontally, bottom aligned with floor)
    body_left = (config.image_size - body_width) // 2
    body_right = body_left + body_width
    body_bottom = config.image_size
    body_top = body_bottom - body_height
    
    # String attachment point (center-top of body interior)
    attach_x = config.image_size // 2
    attach_y = body_top + config.body_border_width + 5  # Slightly inside the body
    
    # Random angle (degrees from vertical)
    angle_deg = random.uniform(*config.angle_range)
    angle_rad = math.radians(angle_deg)
    
    # Weight settings
    weight_size = random.randint(*config.weight_size_range)
    weight_shape = random.choice(config.weight_shapes)
    weight_rotation = random.uniform(0, 360)
    
    # Calculate maximum string length that keeps weight inside the image
    # Weight must fit within image bounds (not just body bounds)
    
    margin = 5
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    # Calculate max length based on vertical constraint (weight stays in image)
    # weight_y + weight_size <= image_size - margin
    if abs(cos_angle) > 0.01:
        max_length_vertical = (config.image_size - margin - weight_size - attach_y) / cos_angle
    else:
        max_length_vertical = float('inf')
    
    # Calculate max length based on horizontal constraint (weight stays in image)
    # For positive angle (swinging right): attach_x + L * sin(angle) + weight_size <= image_size - margin
    # For negative angle (swinging left): attach_x + L * sin(angle) - weight_size >= margin
    if sin_angle > 0.01:
        max_length_horizontal = (config.image_size - margin - weight_size - attach_x) / sin_angle
    elif sin_angle < -0.01:
        max_length_horizontal = (margin + weight_size - attach_x) / sin_angle
    else:
        max_length_horizontal = float('inf')
    
    # Take minimum of both constraints
    max_string_length = min(max_length_vertical, max_length_horizontal)
    max_string_length = max(max_string_length, 20)  # Minimum viable length
    
    # Calculate minimum string length (at least some visible string)
    min_string_length = max(15, weight_size + 10)
    
    # Random string length within valid range
    if max_string_length > min_string_length:
        length_fraction = random.uniform(*config.string_length_range)
        string_length = min_string_length + length_fraction * (max_string_length - min_string_length)
    else:
        string_length = min_string_length
    
    # Calculate weight center position
    weight_x = attach_x + string_length * math.sin(angle_rad)
    weight_y = attach_y + string_length * math.cos(angle_rad)
    
    # Clamp weight position to stay within image bounds
    weight_x = max(weight_size + margin, 
                   min(config.image_size - weight_size - margin, weight_x))
    weight_y = max(attach_y + weight_size, 
                   min(config.image_size - weight_size - margin, weight_y))
    
    # Generate colors
    body_color = get_random_color()
    weight_color = get_random_color(exclude_color=body_color)
    
    # String settings
    string_thickness = random.randint(*config.string_thickness_range)
    
    # --- Draw the pendulum ---
    
    # 1. Draw body (solid filled rectangle)
    draw.rectangle([body_left, body_top, body_right, body_bottom], 
                   fill=body_color, outline=None)
    
    # 2. Draw string
    draw.line([(attach_x, attach_y), (weight_x, weight_y)], 
              fill=config.string_color, width=string_thickness)
    
    # 3. Draw weight
    draw_weight(draw, (weight_x, weight_y), weight_size, weight_shape, 
                weight_rotation, weight_color)
    
    # Compile metadata
    metadata = {
        "body": {
            "width": body_width,
            "height": body_height,
            "color": body_color,
            "position": {"left": body_left, "top": body_top, "right": body_right, "bottom": body_bottom}
        },
        "string": {
            "length": string_length,
            "thickness": string_thickness,
            "angle_degrees": angle_deg,
            "attachment_point": {"x": attach_x, "y": attach_y}
        },
        "weight": {
            "shape": weight_shape,
            "size": weight_size,
            "rotation": weight_rotation,
            "color": weight_color,
            "center": {"x": weight_x, "y": weight_y}
        }
    }
    
    return img, metadata


def generate_dataset(config: PendulumConfig):
    """Generate the full pendulum dataset.
    
    Args:
        config: Configuration for dataset generation
    """
    # Set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
    
    # Create output directory
    output_path = Path(config.output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    all_metadata = []
    
    print(f"Generating {config.num_images} pendulum images...")
    print(f"Output directory: {output_path.absolute()}")
    
    for i in tqdm(range(config.num_images)):
        img, metadata = generate_pendulum(config)
        
        # Save image
        filename = f"pendulum_{i:05d}.png"
        img.save(images_path / filename)
        
        # Store metadata
        metadata["filename"] = filename
        metadata["index"] = i
        all_metadata.append(metadata)
    
    # Save metadata
    if config.save_metadata:
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "config": asdict(config),
                "images": all_metadata
            }, f, indent=2)
        print(f"Metadata saved to {metadata_file}")
    
    # Save config separately for easy reference
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"Dataset generation complete!")
    print(f"  - {config.num_images} images saved to {images_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pendulum dataset")
    parser.add_argument("--num-images", type=int, default=10000, 
                        help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image resolution (square)")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Don't save metadata JSON")
    
    # Body settings
    parser.add_argument("--body-width-min", type=int, default=60)
    parser.add_argument("--body-width-max", type=int, default=120)
    parser.add_argument("--body-height-min", type=int, default=140)
    parser.add_argument("--body-height-max", type=int, default=220)
    
    # String settings
    parser.add_argument("--string-length-min", type=float, default=0.3,
                        help="Min string length as fraction of max")
    parser.add_argument("--string-length-max", type=float, default=0.9,
                        help="Max string length as fraction of max")
    parser.add_argument("--string-thickness-min", type=int, default=2)
    parser.add_argument("--string-thickness-max", type=int, default=3)
    
    # Angle settings
    parser.add_argument("--angle-min", type=float, default=-90.0,
                        help="Minimum angle from vertical (degrees)")
    parser.add_argument("--angle-max", type=float, default=90.0,
                        help="Maximum angle from vertical (degrees)")
    
    # Weight settings
    parser.add_argument("--weight-size-min", type=int, default=15)
    parser.add_argument("--weight-size-max", type=int, default=25)
    parser.add_argument("--weight-shapes", type=str, nargs="+",
                        default=["circle", "square", "triangle"],
                        help="Weight shapes to use")
    
    args = parser.parse_args()
    
    # Build config from arguments
    config = PendulumConfig(
        image_size=args.image_size,
        body_width_range=(args.body_width_min, args.body_width_max),
        body_height_range=(args.body_height_min, args.body_height_max),
        string_length_range=(args.string_length_min, args.string_length_max),
        string_thickness_range=(args.string_thickness_min, args.string_thickness_max),
        angle_range=(args.angle_min, args.angle_max),
        weight_size_range=(args.weight_size_min, args.weight_size_max),
        weight_shapes=args.weight_shapes,
        num_images=args.num_images,
        output_dir=args.output_dir,
        save_metadata=not args.no_metadata,
        seed=args.seed
    )
    
    generate_dataset(config)


if __name__ == "__main__":
    main()
