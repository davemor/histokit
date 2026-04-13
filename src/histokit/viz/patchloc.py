from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from histokit.dataset.sample import Sample
from histokit.dataset.schema import AnnotationSchema
from histokit.patchset.patchset import PatchSet

# Severity gradient palette: blue (low) → green → yellow → orange → red (high)
_PALETTE = [
    np.array([ 30, 120, 255], dtype=np.uint8),  # 0 — blue
    np.array([ 80, 220,  80], dtype=np.uint8),  # 1 — light green
    np.array([255, 240,   0], dtype=np.uint8),  # 2 — yellow
    np.array([255, 160,   0], dtype=np.uint8),  # 3 — orange
    np.array([220,  50,  20], dtype=np.uint8),  # 4 — red-orange
    np.array([180,   0,   0], dtype=np.uint8),  # 5 — red
]


def build_label_colours(
    label_map: dict[str, int], fill_val: int, show_fill: bool,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return (all_label_colours, active_label_colours)."""
    label_values_sorted = sorted(label_map.values())
    all_colours: dict[int, np.ndarray] = {
        val: _PALETTE[i % len(_PALETTE)]
        for i, val in enumerate(label_values_sorted)
    }
    active_colours: dict[int, np.ndarray] = {
        val: colour
        for val, colour in all_colours.items()
        if val != fill_val or show_fill
    }
    return all_colours, active_colours


def blend_overlay(
    thumb: np.ndarray,
    frame: pd.DataFrame,
    grid_h: int,
    grid_w: int,
    width: int,
    height: int,
    fill_val: int,
    label_colours: dict[int, np.ndarray],
    show_only_keep: bool,
    alpha: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the blended thumbnail with annotation colours.

    Returns the blended image and the valid_frame used (for downstream present-label queries).
    """
    active_frame = frame
    if show_only_keep and "keep" in frame.columns:
        active_frame = frame[frame["keep"] == True]

    label_grid = np.full((grid_h, grid_w), fill_val, dtype=np.float32)
    render_mask = np.zeros((grid_h, grid_w), dtype=bool)
    valid_frame = active_frame[active_frame["annotation_label"].notna()]
    rows = valid_frame["row"].astype(int).values
    cols = valid_frame["column"].astype(int).values
    label_grid[rows, cols] = valid_frame["annotation_label"].astype(float).values
    render_mask[rows, cols] = True

    label_vis = np.array(
        Image.fromarray(label_grid).resize((width, height), Image.Resampling.NEAREST)
    )
    render_vis = np.array(
        Image.fromarray(render_mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
    ).astype(bool)

    overlay = thumb.copy()
    for ann_val, colour in label_colours.items():
        mask = render_vis & (label_vis == ann_val)
        if mask.any():
            overlay[mask] = colour

    blended = (
        thumb.astype(float) * (1 - alpha) + overlay.astype(float) * alpha
    ).astype(np.uint8)
    return blended, valid_frame


def build_title(
    sample: Sample,
    val_to_name: dict[int, str],
    present_label_vals: set[int],
) -> str:
    present_label_names = [
        val_to_name[v].replace("_", " ").title()
        for v in sorted(present_label_vals)
        if v in val_to_name
    ]
    title_parts = [f"Slide: {sample.id}"]
    if sample.metadata:
        meta_str = "  |  ".join(
            f"{str(k).replace('_', ' ').title()}: {str(v).replace('_', ' ').title()}"
            for k, v in sample.metadata.items()
        )
        title_parts.append(meta_str)
    title_parts.append(
        "Present labels: " + (", ".join(present_label_names) if present_label_names else "none")
    )
    return "\n".join(title_parts)


def draw_patch_boxes(
    ax: Axes,
    frame: pd.DataFrame,
    grid_h: int,
    grid_w: int,
    width: int,
    height: int,
    box_alpha: float,
) -> None:
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


def build_legend_handles(
    label_colours: dict[int, np.ndarray],
    val_to_name: dict[int, str],
    present_label_vals: set[int],
    fill_val: int,
    show_fill: bool,
    show_only_keep: bool,
    frame: pd.DataFrame,
) -> list[mpatches.Patch]:
    handles = [
        mpatches.Patch(
            color=colour / 255.0,
            label=val_to_name[val].replace("_", " ").title(),
        )
        for val, colour in label_colours.items()
        if val in present_label_vals and not (show_only_keep and val == fill_val)
    ]
    all_label_vals = set(
        frame[frame["annotation_label"].notna()]["annotation_label"].astype(int).unique()
    )
    if not show_fill and fill_val in val_to_name and fill_val in all_label_vals:
        handles.insert(0,
            mpatches.Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=2,
                label=val_to_name[fill_val].replace("_", " ").title(),
            )
        )
    return handles


def visualise_patchset(
    patchset: PatchSet,
    width: int = 1024,
    height: int | None = None,
    alpha: float = 0.45,
    box_alpha: float = 0.75,
    show_fill: bool = False,
    show_only_keep: bool = True,
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
        show_fill: If True, fill-label patches are also coloured.
            If False, fill-label patches are left as the original thumbnail.
        show_only_keep: If True, patches where keep is False are not shown.
            Has no effect if there is no keep column.
    """
    if not patchset.contexts:
        raise ValueError("PatchSet has no contexts")

    context = patchset.contexts[0]
    sample = context.sample
    annotation_schema = sample.annotation_schema

    with sample.open_slide() as slide:
        if height is None:
            slide_w, slide_h = slide.dimensions[0]
            height = max(1, round(width * slide_h / slide_w))
        thumb = slide.get_thumbnail_for_size(width, height)

    frame = patchset.frame
    grid_w = int(frame["column"].max()) + 1
    grid_h = int(frame["row"].max()) + 1

    # Create figure with fixed padding for title and legend
    fig_h = height / 100
    fig_w = width / 100
    top_pad = 0.6
    bottom_pad = 0.9
    total_h = fig_h + top_pad + bottom_pad
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, total_h))
    fig.subplots_adjust(
        top=1.0 - top_pad / total_h,
        bottom=bottom_pad / total_h,
        left=0.0,
        right=1.0,
    )

    if annotation_schema is None:
        ax.imshow(thumb)
        ax.axis("off")
        plt.show()
        return

    label_map = annotation_schema.label_map
    fill_val = label_map[annotation_schema.fill_label]
    val_to_name = {v: k for k, v in label_map.items()}

    _, label_colours = build_label_colours(label_map, fill_val, show_fill)

    blended, valid_frame = blend_overlay(
        thumb, frame, grid_h, grid_w, width, height,
        fill_val, label_colours, show_only_keep, alpha,
    )

    ax.imshow(blended)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.0)

    present_label_vals = set(valid_frame["annotation_label"].astype(int).unique())
    title = build_title(sample, val_to_name, present_label_vals)
    ax.set_title(title, fontsize=18, pad=12)

    draw_patch_boxes(ax, frame, grid_h, grid_w, width, height, box_alpha)

    legend_handles = build_legend_handles(
        label_colours, val_to_name, present_label_vals,
        fill_val, show_fill, show_only_keep, frame,
    )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=max(1, len(legend_handles)),
            fontsize=18,
            frameon=True,
            bbox_to_anchor=(0.5, 0.005),
        )

    plt.show()
