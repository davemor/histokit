from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from histokit.patchset.patchset import PatchSet


def visualise_patchset(
    patchset: PatchSet,
    width: int = 1024,
    height: int | None = None,
    alpha: float = 0.55,
    box_alpha: float = 0.6,
) -> None:
    """Visualise a PatchSet overlaid on a slide thumbnail.

    Patches whose annotation_label differs from the schema's fill label are
    coloured according to their annotation. All other patches are left as the
    original thumbnail. A legend is shown beneath the image.

    Args:
        patchset: The PatchSet to visualise. The first context's slide is used.
        width: Width of the output image in pixels.
        height: Height of the output image in pixels. If None, computed from
            width to preserve the slide's aspect ratio.
        alpha: Blending factor for the annotation overlay (0 = transparent, 1 = opaque).
        box_alpha: Opacity of the patch border boxes.
    """
    if not patchset.contexts:
        raise ValueError("PatchSet has no contexts")

    context = patchset.contexts[0]
    sample = context.sample
    patch_size = context.patch_size
    level = context.level
    annotation_schema = sample.annotation_schema

    with sample.open_slide() as slide:
        if height is None:
            slide_w, slide_h = slide.dimensions[0]
            height = max(1, round(width * slide_h / slide_w))
        thumb = slide.get_thumbnail_for_size(width, height)
        grid_size = slide.size_in_patches(patch_size, level)

    grid_w = grid_size.width   # number of patch columns
    grid_h = grid_size.height  # number of patch rows

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))

    if annotation_schema is None:
        ax.imshow(thumb)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return

    label_map = annotation_schema.label_map
    fill_val = label_map[annotation_schema.fill_label]

    # Bold, distinct colours that are easy to read on H&E slides — avoids pink/purple
    _PALETTE = [
        np.array([  0, 200, 255], dtype=np.uint8),  # sky blue
        np.array([255, 200,   0], dtype=np.uint8),  # amber
        np.array([255,  60,  60], dtype=np.uint8),  # red
        np.array([  0, 220, 100], dtype=np.uint8),  # green
        np.array([255, 140,   0], dtype=np.uint8),  # orange
        np.array([  0, 240, 240], dtype=np.uint8),  # cyan
        np.array([255, 255,   0], dtype=np.uint8),  # yellow
        np.array([255, 100,   0], dtype=np.uint8),  # deep orange
    ]

    label_values_sorted = sorted(label_map.values())
    label_colours: dict[int, np.ndarray] = {
        val: _PALETTE[i % len(_PALETTE)]
        for i, val in enumerate(label_values_sorted)
        if val != fill_val
    }

    # Build a patch-resolution label grid from the frame
    frame = patchset.frame
    label_grid = np.full((grid_h, grid_w), fill_val, dtype=np.float32)
    valid_frame = frame[frame["annotation_label"].notna()]
    label_grid[
        valid_frame["row"].astype(int).values,
        valid_frame["column"].astype(int).values,
    ] = valid_frame["annotation_label"].astype(float).values

    # Upscale label grid to thumbnail size using nearest-neighbour interpolation
    label_vis = np.array(
        Image.fromarray(label_grid).resize((width, height), Image.Resampling.NEAREST)
    )

    # Build overlay by painting each non-fill label over the thumbnail
    overlay = thumb.copy()
    for ann_val, colour in label_colours.items():
        mask = label_vis == ann_val
        if mask.any():
            overlay[mask] = colour

    blended = (
        thumb.astype(float) * (1 - alpha) + overlay.astype(float) * alpha
    ).astype(np.uint8)

    ax.imshow(blended)
    ax.axis("off")

    # Draw a black box around each kept patch
    cell_w = width / grid_w
    cell_h = height / grid_h
    has_keep = "keep" in frame.columns
    keep_lookup: set[tuple[int, int]] = set()
    if has_keep:
        kept = frame[frame["keep"] == True]
        keep_lookup = set(zip(kept["row"].astype(int), kept["column"].astype(int)))
    for row in range(grid_h):
        for col in range(grid_w):
            if has_keep and (row, col) not in keep_lookup:
                continue
            rect = mpatches.Rectangle(
                (col * cell_w, row * cell_h),
                cell_w,
                cell_h,
                linewidth=0.5,
                edgecolor="black",
                facecolor="none",
                alpha=box_alpha,
            )
            ax.add_patch(rect)

    val_to_name = {v: k for k, v in label_map.items()}
    legend_handles = [
        mpatches.Patch(
            color=colour / 255.0,
            label=val_to_name[val].replace("_", " ").title(),
        )
        for val, colour in label_colours.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
