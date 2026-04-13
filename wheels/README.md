# Bundled Wheels

Place extension-local dependency wheels here when you want the add-on to install them without using `pip`.

Current supported optional dependency set:

- `psd_tools`
- `Pillow`

After adding compatible wheel files for Blender's bundled Python version and your platform, run the add-on's `Install PSD Backend` action. The add-on will extract those wheels into `vendor/site-packages/`.

No See-through generation wheels are expected here yet.
