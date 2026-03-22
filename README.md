# Histokit
Histokit is a histopathology whole slide image preprocessing package for Python and command line tool.

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
*Datasets* - represent a set of slide, their annoations, and meta data. Datasets provide the following information:
- a list of the slides
- a list of slide annotations (optional)
- the format to use to load the slides
- meta data about each slide (multiple slide level labels)

*Patchset* - an index of the patches on a slide, their labels, provinence, and other meta data.

The patchset is constructed thoughout the different stages, i.e. the different stages add the 