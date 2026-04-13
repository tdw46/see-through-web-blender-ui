# Hallway Avatar Gen

Blender extension scaffold for importing See-through-style layered PSD files into Blender. The long-term goal is 2.5-D avatar generation from See-through inputs, but this first release is focused on importing and organizing those layers cleanly.

## Current scope

- PSD import via extension-local `psd_tools` and `Pillow` dependencies
- Recursive PSD layer traversal with hidden/empty-layer skipping
- Cached PNG export per visible raster layer
- Heuristic See-through-aware name classification
- Layer-aligned Blender mesh creation for imported visible parts

## Current limitations

- The add-on is currently positioned as a See-through layer importer, not a finished 2.5-D avatar generator yet
- 2.5-D generation, rigging, Gemini integration, and See-through image parsing are planned follow-up work
- The alpha mesh stage currently uses an internal fallback grid mesh over the visible alpha bounds instead of the full `Import Meshed Alpha` tracing pipeline

## Usage

1. Add local `psd_tools` and `Pillow` wheels to [wheels/README.md](/Users/tylerwalker/Library/Application%20Support/Blender/5.0/extensions/user_default/hallway_avatar_gen/wheels/README.md) or extracted packages to [vendor/README.md](/Users/tylerwalker/Library/Application%20Support/Blender/5.0/extensions/user_default/hallway_avatar_gen/vendor/README.md).
2. Open the add-on preferences or the `Hallway` sidebar panel and run `Install PSD Backend`.
3. Import a PSD with `Import PSD Avatar`.
4. Review the imported layers and skipped-layer summary in Blender.
