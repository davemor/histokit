# Histokit

Histokit is a histopathology whole slide image preprocessing package for Python and command line tool.

# Install as a tool

Histokit can be installed as a command line tool directly from this repository using uv. Here is an example:

```bash
uv tool update-shell
uv tool install git+https://github.com/davemor/histokit
```

# CLI Interface

```bash
❯ histokit --help

 Usage: histokit [OPTIONS] COMMAND [ARGS]...

 histokit — histopathology toolkit CLI.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ list     List available built-in pipelines.                                  │
│ plan     Show pipeline stages and resolved parameters.                       │
│ run      Run a pipeline on a dataset and save the resulting PatchSet.        │
│ preview  Preview a pipeline on a single sample with diagnostic images.       │
│ export   Export patch images from a saved PatchSet to label-name             │
│          directories.                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯

```

## `histokit list`

Show all built-in pipelines that ship with histokit. Each entry shows the import reference and stage count.

```bash
histokit list
```

```
Available pipelines:

  histokit.pipelines.presets.basic:pipeline  (4 stages)
  histokit.pipelines.presets.research:pipeline  (4 stages)
```

## `histokit plan`

Inspect a pipeline's stages and see what parameters they use. Use `--set` to preview overrides without running anything.

```bash
# Show default parameters
histokit plan histokit.pipelines.presets.basic:pipeline

# Preview with overrides
histokit plan histokit.pipelines.presets.basic:pipeline --set patch_size=512 --set level=0
```

## `histokit run`

Run a pipeline on a full dataset and save the combined PatchSet to disk.

```bash
# Basic run
histokit run histokit.pipelines.presets.basic:pipeline \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output runs/cervical_basic

# With parameter overrides
histokit run histokit.pipelines.presets.basic:pipeline \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output runs/cervical_512 \
  --set patch_size=512

# Overwrite a previous run
histokit run histokit.pipelines.presets.basic:pipeline \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output runs/cervical_basic \
  --overwrite
```

## `histokit preview`

Run a pipeline on a single sample and save diagnostic images (thumbnail, patch overlay) to an output directory. Useful for checking pipeline settings before a full run.

```bash
# Preview the first sample in the dataset
histokit preview histokit.pipelines.presets.basic:pipeline \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output preview/cervical

# Preview a specific sample
histokit preview histokit.pipelines.presets.basic:pipeline \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output preview/cervical \
  --sample IC-CX-00001-01
```

## `histokit export`

Export patch images from a saved PatchSet to label-name subdirectories, compatible with `torchvision.datasets.ImageFolder`.

```bash
histokit export runs/cervical_basic \
  --index data/icaird/cervical_mini/index.csv \
  --labels data/icaird/cervical_mini/labels.json \
  --output patches/cervical_basic
```

The output directory will contain one folder per label with individual patch PNG files and a `provenance.json` recording how the export was produced.

# Pipeline Stages

Histokit is based around configurable pipelines with the following stages:

1. Patch Selection
2. Foreground Identification
3. Patch Labelling
4. Patch Filtering
5. Patch Rendering

Pipelines consume dataset objects that represent a set of slides and, optionally, their annotations. They output a patchset: an object that stores patch coordinates, their labels, and their provenance (the slide they came from, the way they have been processed). The patchset can then be used to generate patch images or an input into a downstream feature extraction model.

## Patch Selection

This stage decides how the patches will be sampled from the whole slide images, based on:

- geometry (level, patch size, stride)
- strategy (grid, from annotation)
- sampling (dense, sparse, multiscale)

## Foreground Identification

The stage classifies the patches into Foreground and Background. This usually means tissue detection but it might also mean blood and mucus detection.

- methods (fixed threshold, otsu)
- resolution (thumbnail vs patch-level)
- threshold (minimum tissue fraction)

## Patch Labelling

Assign labels to the patches derived from annotations. This involves rendering the annotation polygons and converting them to patch labels. This may involve having a default label that everything that is not labelled uses.

## Patch Selection

The stage applies selection rules such as:

- should the background be dropped?
- should only labelled patches be used?
- remove unreliable or corrupted patches based on quality

## Patch Rendering

The patchset can then be used to export the patch images. The patches can be normalised using stain normalisation.

# Data Model

_Datasets_ - represent a set of slide, their annoations, and meta data. Datasets provide the following information:

- a list of the slides
- a list of slide annotations (optional)
- the format to use to load the slides
- meta data about each slide (multiple slide level labels)

_Patchset_ - an index of the patches on a slide, their labels, provinence, and other meta data.

The patchset is constructed thoughout the different stages, i.e. the different stages add the
