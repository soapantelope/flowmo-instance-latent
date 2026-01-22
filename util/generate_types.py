#!/usr/bin/env python3
"""
Pendulum Type Dataset Generator

Generates a dataset of pendulum images organized by type.
Each pendulum type has fixed attributes (weight shape, size, color, string length, etc.)
but varies in angle across multiple images.

Naming convention: XX_YYY.png
  - XX: two-digit pendulum type ID (00-99)
  - YYY: three-digit image counter within that type (000-999)
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
class PendulumTypeConfig:
    """Configuration for pendulum type dataset generation."""
    
    # Image settings
    image_size: int = 256
    background_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Number of pendulum types and images per type
    num_types: int = 50
    images_per_type: int = 200  # Different angles per type
    
    # Body settings (ranges for random type generation)
    # These are scaled for 256x256 images
    body_width_range: Tuple[int, int] = (40, 80)
    body_height_range: Tuple[int, int] = (100, 180)
    
    # String settings
    string_length_range: Tuple[float, float] = (0.3, 0.9)  # Fraction of max possible
    string_thickness_range: Tuple[int, int] = (2, 3)
    string_color: Tuple[int, int, int] = (50, 50, 50)
    
    # Angle settings (degrees from vertical)
    angle_range: Tuple[float, float] = (-90.0, 90.0)
    
    # Weight settings
    # Weight size is constrained so pendulum fits at all angles
    # Max weight size must be < image_size/2 - margin to allow horizontal swing
    weight_size_range: Tuple[int, int] = (10, 20)
    weight_shapes: List[str] = field(default_factory=lambda: ["circle", "square", "triangle"])
    
    # Dataset settings
    output_dir: str = "dataset"
    save_metadata: bool = True
    seed: Optional[int] = 42


@dataclass
class PendulumType:
    """A fixed pendulum type with all attributes except angle."""
    type_id: int
    body_width: int
    body_height: int
    body_color: Tuple[int, int, int]
    string_length: float  # Fixed absolute string length in pixels
    string_thickness: int
    weight_shape: str
    weight_size: int
    weight_rotation: float  # Fixed rotation for the weight
    weight_color: Tuple[int, int, int]


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
        
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(color, exclude_color)))
        if distance >= min_distance:
            return color
    
    if exclude_color:
        return tuple(255 - c for c in exclude_color)
    return color


def compute_max_string_length(body_height: int, weight_size: int, 
                               image_size: int, angle_range: Tuple[float, float]) -> float:
    """Compute the maximum string length that works for all angles in the range.
    
    The most constrained case depends on the angle range:
    - For vertical (angle=0), the weight goes furthest down
    - For horizontal (angle=±90), the weight goes furthest left/right
    
    We need to find the length that keeps the weight in bounds for ALL angles.
    """
    margin = 5
    attach_y = (image_size - body_height) + 8  # body_top + 8
    attach_x = image_size // 2
    
    # Check constraints at multiple angles to find the most restrictive
    min_max_length = float('inf')
    
    # Sample angles including extremes and intermediate points
    num_samples = 20
    test_angles = set([angle_range[0], angle_range[1], 0.0])
    for i in range(num_samples):
        test_angles.add(angle_range[0] + (angle_range[1] - angle_range[0]) * i / (num_samples - 1))
    
    for angle_deg in test_angles:
        angle_rad = math.radians(angle_deg)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Vertical constraint (weight must not go below image bottom)
        if cos_angle > 0.01:
            max_len_v = (image_size - margin - weight_size - attach_y) / cos_angle
            if max_len_v > 0:
                min_max_length = min(min_max_length, max_len_v)
        
        # Horizontal constraint (weight must stay within image sides)
        if sin_angle > 0.01:
            # Swinging right
            max_len_h = (image_size - margin - weight_size - attach_x) / sin_angle
            if max_len_h > 0:
                min_max_length = min(min_max_length, max_len_h)
        elif sin_angle < -0.01:
            # Swinging left
            max_len_h = (attach_x - margin - weight_size) / (-sin_angle)
            if max_len_h > 0:
                min_max_length = min(min_max_length, max_len_h)
    
    # Ensure we have a valid positive length
    if min_max_length == float('inf') or min_max_length < 15:
        min_max_length = 15
    
    return min_max_length


def generate_pendulum_types(config: PendulumTypeConfig) -> List[PendulumType]:
    """Generate random pendulum types with fixed attributes."""
    types = []
    
    for type_id in range(config.num_types):
        body_color = get_random_color()
        weight_color = get_random_color(exclude_color=body_color)
        
        body_width = random.randint(*config.body_width_range)
        body_height = random.randint(*config.body_height_range)
        weight_size = random.randint(*config.weight_size_range)
        
        # Compute the max string length that works for all angles
        max_string_length = compute_max_string_length(
            body_height, weight_size, config.image_size, config.angle_range
        )
        
        # Compute actual string length based on fraction of max
        min_string_length = max(15, weight_size + 10)
        length_fraction = random.uniform(*config.string_length_range)
        if max_string_length > min_string_length:
            string_length = min_string_length + length_fraction * (max_string_length - min_string_length)
        else:
            string_length = min_string_length
        
        ptype = PendulumType(
            type_id=type_id,
            body_width=body_width,
            body_height=body_height,
            body_color=body_color,
            string_length=string_length,
            string_thickness=random.randint(*config.string_thickness_range),
            weight_shape=random.choice(config.weight_shapes),
            weight_size=weight_size,
            weight_rotation=random.uniform(0, 360),
            weight_color=weight_color,
        )
        types.append(ptype)
    
    return types


def draw_string(draw: ImageDraw.ImageDraw,
                start: Tuple[float, float],
                end: Tuple[float, float],
                thickness: int,
                fill: Tuple[int, int, int]):
    """Draw a string as a rotated rectangle for consistent thickness at all angles.
    
    Args:
        draw: PIL ImageDraw object
        start: (x, y) start point of string
        end: (x, y) end point of string
        thickness: Width of the string in pixels
        fill: Fill color
    """
    x1, y1 = start
    x2, y2 = end
    
    # Calculate the angle of the line
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    
    # Calculate perpendicular offset for thickness
    half_thick = thickness / 2
    px = half_thick * math.sin(angle)  # Perpendicular x offset
    py = half_thick * math.cos(angle)  # Perpendicular y offset
    
    # Four corners of the rectangle
    points = [
        (x1 - px, y1 + py),
        (x1 + px, y1 - py),
        (x2 + px, y2 - py),
        (x2 - px, y2 + py),
    ]
    
    draw.polygon(points, fill=fill)


def draw_equilateral_triangle(draw: ImageDraw.ImageDraw, 
                               center: Tuple[float, float], 
                               size: float, 
                               rotation: float, 
                               fill: Tuple[int, int, int]):
    """Draw an equilateral triangle centered at the given point."""
    cx, cy = center
    rotation_rad = math.radians(rotation)
    
    points = []
    for i in range(3):
        angle = rotation_rad + (2 * math.pi * i / 3)
        x = cx + size * math.sin(angle)
        y = cy - size * math.cos(angle)
        points.append((x, y))
    
    draw.polygon(points, fill=fill)


def draw_rotated_square(draw: ImageDraw.ImageDraw,
                        center: Tuple[float, float],
                        size: float,
                        rotation: float,
                        fill: Tuple[int, int, int]):
    """Draw a square centered at the given point with rotation."""
    cx, cy = center
    rotation_rad = math.radians(rotation)
    corner_dist = size * math.sqrt(2)
    
    points = []
    for i in range(4):
        angle = rotation_rad + math.pi/4 + (math.pi * i / 2)
        x = cx + corner_dist * math.cos(angle)
        y = cy + corner_dist * math.sin(angle)
        points.append((x, y))
    
    draw.polygon(points, fill=fill)


def draw_weight(draw: ImageDraw.ImageDraw,
                center: Tuple[float, float],
                size: float,
                shape: str,
                rotation: float,
                fill: Tuple[int, int, int]):
    """Draw a weight shape at the given position."""
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


def render_pendulum(ptype: PendulumType, angle_deg: float, 
                    config: PendulumTypeConfig) -> Tuple[Image.Image, dict]:
    """Render a pendulum of a given type at a specific angle.
    
    Returns:
        Tuple of (PIL Image, metadata dict)
    """
    img = Image.new('RGB', (config.image_size, config.image_size), config.background_color)
    draw = ImageDraw.Draw(img)
    
    # Ensure body fits in image
    body_width = min(ptype.body_width, config.image_size - 20)
    body_height = min(ptype.body_height, config.image_size - 20)
    
    # Body position (centered horizontally, bottom aligned with floor)
    body_left = (config.image_size - body_width) // 2
    body_right = body_left + body_width
    body_bottom = config.image_size
    body_top = body_bottom - body_height
    
    # String attachment point (center-top of body)
    attach_x = config.image_size // 2
    attach_y = body_top + 8  # Slightly inside the body
    
    angle_rad = math.radians(angle_deg)
    
    # Use the fixed string length from the pendulum type
    string_length = ptype.string_length
    
    # Calculate weight center position
    weight_x = attach_x + string_length * math.sin(angle_rad)
    weight_y = attach_y + string_length * math.cos(angle_rad)
    
    # --- Draw the pendulum ---
    
    # 1. Draw body (solid filled rectangle)
    draw.rectangle([body_left, body_top, body_right, body_bottom], 
                   fill=ptype.body_color, outline=None)
    
    # 2. Draw string (as polygon for consistent thickness at all angles)
    draw_string(draw, (attach_x, attach_y), (weight_x, weight_y),
                ptype.string_thickness, config.string_color)
    
    # 3. Draw weight
    draw_weight(draw, (weight_x, weight_y), ptype.weight_size, ptype.weight_shape, 
                ptype.weight_rotation, ptype.weight_color)
    
    # Compile metadata
    metadata = {
        "type_id": ptype.type_id,
        "angle_degrees": angle_deg,
        "body": {
            "width": body_width,
            "height": body_height,
            "color": ptype.body_color,
        },
        "string": {
            "length": string_length,
            "thickness": ptype.string_thickness,
        },
        "weight": {
            "shape": ptype.weight_shape,
            "size": ptype.weight_size,
            "rotation": ptype.weight_rotation,
            "color": ptype.weight_color,
            "center": {"x": weight_x, "y": weight_y}
        }
    }
    
    return img, metadata


def generate_dataset(config: PendulumTypeConfig):
    """Generate the full pendulum type dataset."""
    if config.seed is not None:
        random.seed(config.seed)
    
    # Create output directory
    output_path = Path(config.output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Generate random angles (same angles used for all types)
    angle_min, angle_max = config.angle_range
    angles = sorted([random.uniform(angle_min, angle_max) for _ in range(config.images_per_type)])
    
    # Generate pendulum types
    print(f"Generating {config.num_types} pendulum types...")
    pendulum_types = generate_pendulum_types(config)
    
    # Generate images
    all_metadata = []
    type_metadata = []
    
    total_images = config.num_types * config.images_per_type
    print(f"Generating {total_images} images ({config.num_types} types x {config.images_per_type} angles)...")
    print(f"Output directory: {output_path.absolute()}")
    
    with tqdm(total=total_images) as pbar:
        for ptype in pendulum_types:
            type_info = asdict(ptype)
            type_metadata.append(type_info)
            
            for img_idx, angle in enumerate(angles):
                img, metadata = render_pendulum(ptype, angle, config)
                
                # Naming convention: XX_YYY.png
                filename = f"{ptype.type_id:02d}_{img_idx:03d}.png"
                img.save(images_path / filename)
                
                metadata["filename"] = filename
                metadata["image_index"] = img_idx
                all_metadata.append(metadata)
                
                pbar.update(1)
    
    # Save metadata
    if config.save_metadata:
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "config": asdict(config),
                "pendulum_types": type_metadata,
                "images": all_metadata
            }, f, indent=2)
        print(f"Metadata saved to {metadata_file}")
    
    # Save config
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"Dataset generation complete!")
    print(f"  - {config.num_types} pendulum types")
    print(f"  - {config.images_per_type} images per type")
    print(f"  - {total_images} total images saved to {images_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pendulum type dataset")
    parser.add_argument("--num-types", type=int, default=50,
                        help="Number of distinct pendulum types")
    parser.add_argument("--images-per-type", type=int, default=200,
                        help="Number of images (angles) per type")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image resolution (square)")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Don't save metadata JSON")
    
    # Body settings
    parser.add_argument("--body-width-min", type=int, default=40)
    parser.add_argument("--body-width-max", type=int, default=80)
    parser.add_argument("--body-height-min", type=int, default=100)
    parser.add_argument("--body-height-max", type=int, default=180)
    
    # String settings
    parser.add_argument("--string-length-min", type=float, default=0.3)
    parser.add_argument("--string-length-max", type=float, default=0.9)
    parser.add_argument("--string-thickness-min", type=int, default=2)
    parser.add_argument("--string-thickness-max", type=int, default=3)
    
    # Angle settings
    parser.add_argument("--angle-min", type=float, default=-90.0)
    parser.add_argument("--angle-max", type=float, default=90.0)
    
    # Weight settings
    parser.add_argument("--weight-size-min", type=int, default=10)
    parser.add_argument("--weight-size-max", type=int, default=20)
    parser.add_argument("--weight-shapes", type=str, nargs="+",
                        default=["circle", "square", "triangle"])
    
    args = parser.parse_args()
    
    config = PendulumTypeConfig(
        image_size=args.image_size,
        num_types=args.num_types,
        images_per_type=args.images_per_type,
        body_width_range=(args.body_width_min, args.body_width_max),
        body_height_range=(args.body_height_min, args.body_height_max),
        string_length_range=(args.string_length_min, args.string_length_max),
        string_thickness_range=(args.string_thickness_min, args.string_thickness_max),
        angle_range=(args.angle_min, args.angle_max),
        weight_size_range=(args.weight_size_min, args.weight_size_max),
        weight_shapes=args.weight_shapes,
        output_dir=args.output_dir,
        save_metadata=not args.no_metadata,
        seed=args.seed
    )
    
    generate_dataset(config)


if __name__ == "__main__":
    main()
