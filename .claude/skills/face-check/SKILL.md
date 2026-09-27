---
name: face-check
description: Check generated images for identifiable faces before they are committed or published, and exclude the ones that fail. Use for sample grids, failure galleries, and any figure built from dataset or field-test images.
---

Every image in `docs/assets/`, the README, a gallery, or a video is public.
Nothing goes out with a face someone could recognize (rule 3).

1. Regenerate the images. For sample grids that is `make inspect`.
2. List the file behind each tile. For grids, run
   `uv run frc-inspect --list-grid`, which logs `<key> r<row>c<col> <split>/<file>`
   in the order the grid draws them.
3. Open every image at full size. A tile fails if a person's face can be
   made out, including drive teams, pit crews, and spectators close to the
   camera. Distant crowds in overhead broadcast shots pass. When unsure, it
   fails.
4. Add each failing file name to the exclude list and regenerate. For grids
   the list is `inspect.grid.exclude` in `configs/project.yaml`. Excluding a
   file swaps in the next image in the seeded order, so only the replaced
   tiles are new. Check just those.
5. Repeat until nothing fails. Keep every PNG under the 500 KB hook.
6. If the policy changed, record it in docs/DECISIONS.md (D-010 holds the
   current one). Tell Nick which images were excluded and ask him to look
   at the final images before the PR merges.
