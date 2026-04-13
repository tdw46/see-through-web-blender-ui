# Vendored Python Dependencies

This folder is for extension-local Python packages that should ship with or live beside the add-on.

For PSD import support, you can place extracted package folders here, such as:

- `vendor/psd_tools`
- `vendor/PIL`

If you prefer wheel files instead, place compatible wheels in the sibling `wheels/` folder and use the add-on's bundled PSD install action to extract them into `vendor/site-packages/`.

No See-through generation dependencies are required in this folder yet.
