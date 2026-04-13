from histokit.pipelines.params import param
from histokit.pipelines.pipeline import Pipeline
from histokit.pipelines.stages import Grid, TissueMask, AssignLabels, FilterPatches

pipeline: Pipeline = (
    Grid(
        level=param("level", 1),
        patch_size=param("patch_size", 256),
        stride=param("stride"),
    )
    >> TissueMask(
        method=param("tissue_method", "per_patch_canny_ranker"),
    )
    >> AssignLabels(
        policy=param("label_policy", "majority"),
    )
    >> FilterPatches(
        tissue_threshold=param("tissue_threshold", 0.02),
        drop_background=param("drop_background", True),
        require_label=param("require_label", False),
    )
)
