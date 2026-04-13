from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSetManifest:
    # TODO: note that the setting for the pipeline should also be here
    created_at: str