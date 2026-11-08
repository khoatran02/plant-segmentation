# Demo samples

These five images are copied from the PlantSeg test split. Each matching `.txt` file contains the disease prompt; `prompts.json` records the prompt and source image. They are intended for demonstration, not as a substitute for evaluation on the full test set. Source and usage rights follow the original PlantSeg dataset.

Run `python demo.py` from the repository root and open <http://127.0.0.1:8000>. For these samples the demo looks up the reference disease from image pixels and uses it to generate a predicted mask; this is not disease classification. See the root README for setup and details.
