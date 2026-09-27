# Decisions

One entry per decision that someone might later question. Newest last.

## Carried over from a previous attempt

D-001 to D-004 came from an earlier setup of this project. All four were
re-checked on 2026-09-26 during Task 1 and still apply.

## D-001: Use supervision 0.29.x and make inference optional (2026-09-26)

Context. `inference` 1.7.x requires `supervision>=0.29,<0.30` and
`pydantic<2.12`, so the latest supervision (0.30) cannot be installed next to
it.

Decision. Use supervision 0.29.x. Keep runtime ranges in `pyproject.toml`
loose so the `infer` extra co-resolves, and let `uv.lock` hold the exact pins.
Ship `inference` as an optional extra (`uv sync --extra infer`).

Why. Evaluation and edge deployment both need `inference`, so its pins win
over having the newest supervision. Making it optional keeps CI light and
offline.

Consequences. Anything written against supervision has to use the 0.29 API.
Code that needs `inference` only works after `uv sync --extra infer`.

Re-verified 2026-09-26. `inference` 1.7.2 still pins supervision below 0.30
and pydantic below 2.12, while supervision 0.30.5 is out. `uv.lock` resolves
the `infer` extra with supervision 0.29.1 and pydantic 2.11.10.

## D-002: Add console scripts with their modules (2026-09-26)

Context. `pyproject.toml` and the Makefile could list every planned console
script and pipeline target up front.

Decision. Add each console script and its Make target in the same commit as
the module it runs, never as a stub.

Why. A stubbed entry point is broken until its module lands, and a reader
cloning the repo mid-way would hit it.

Consequences. The Makefile and `pyproject.toml` grow task by task, and every
target that exists works.

## D-003: Read package versions with importlib.metadata (2026-09-26)

Context. Run metadata records package versions (SPEC section 3.5), which the
spec describes as `uv pip freeze`.

Decision. Read them with `importlib.metadata`.

Why. It gives the same information without spawning a subprocess or
depending on `uv` being on the path at run time.

Consequences. Metadata still lists every installed distribution and its
version, in a format the package controls.

## D-004: Rebase-merge only, with branch protection after first CI run (2026-09-26)

Context. SPEC section 3.2 asks for atomic commits landing linearly on `main`
and merges only with CI green.

Decision. Limit GitHub merges to rebase-merge. Add branch protection that
requires the `lint, typecheck, test` check on `main` once that check has run
at least once.

Why. GitHub only offers a check as required after it has reported on the
repo, so protection has to wait for the first CI run.

Consequences. Between the first PR and enabling protection, CI green is
enforced by habit only.

## D-005: Support Python 3.11 to 3.13 only (2026-09-26)

Context. The project targets Python 3.11. `inference` 1.7.2 and its
`inference-models` dependency require Python below 3.14.

Decision. Set `requires-python = ">=3.11,<3.14"` and pin 3.11 in
`.python-version`.

Why. uv resolves one lock for every Python version the project allows, and
`inference` has no release that installs on 3.14.

Consequences. The project will not install on 3.14 until `inference`
supports it. Check the Colab Python version when the notebook lands.

## D-006: Exclude uv.lock from the large-file hook (2026-09-26)

Context. pre-commit rejects files over 500 KB so data and weights never get
committed. With the `infer` extra resolved, `uv.lock` is about 780 KB.

Decision. Exclude `uv.lock` from `check-added-large-files` by exact path.
Every other file keeps the limit.

Why. The lock is generated text that has to be committed (SPEC section 3.1).
It is not what the limit protects against.

Consequences. A large lock diff shows up in review instead of being blocked.

## D-007: Make supervision, roboflow, and imagehash core dependencies (2026-09-26)

Context. Task 2 downloads datasets with the `roboflow` SDK, draws sample
grids with `supervision`, and finds near-duplicate images with perceptual
hashes. supervision was only in the `infer` extra.

Decision. Add `roboflow>=1.5,<1.6`, `supervision>=0.29,<0.30`, and
`imagehash>=4.3,<5` to the core dependencies, along with `numpy` and
`pillow`, which the code imports directly.

Why. Download and inspection run without `inference`, so these packages
can't live in the extra. The supervision range matches what `inference`
pins (D-001), so the extra still co-resolves. `imagehash` only adds PyWavelets,
since scipy is already in the lock. numpy stays below 2.4 because `roboflow` 1.5.1 requires
it.

Consequences. `uv.lock` resolves roboflow 1.5.1, supervision 0.29.1, and
imagehash 4.3.2. `requirements.txt` grows, since Colab needs the same
packages to download data.

## D-008: Pin the candidate dataset versions (2026-09-26)

Context. SPEC section 5 lists five candidates with unverified slugs,
versions, and classes. `make download ARGS=--resolve` looked each one up on
2026-09-26 and wrote `reports/dataset_resolution.json`. The listing of every
`robot-zftp2` project is in the version of that file committed with the
pins. Later lookups only cover the pinned projects.

Decision. Pin the latest published version of each project:
`marswars` version 5, `testingfrfr` version 1, `scorekeeper` version 1,
`robotzftp2` (`rebuilt-dataset-hcmwl`) version 1, and a sixth key,
`robotzftp2_fuel` (`frc-2026-fuel-ndrbj`) version 1. Leave `lava`
unpinned.

Why. The `robot-zftp2` workspace holds three REBUILT projects.
`rebuilt-dataset-hcmwl` is the largest one with a published version.
`frc-2026-fuel-ndrbj` is the only other one with a version, so it is kept
as a separate key and inspection decides whether it is worth using.
`2026-rebuilt-ezy58` has no version, and its classes and image count match
`marswars` version 3, so it looks like a copy. The rest of the workspace is
unrelated to FRC. `lava` has no published version, and Roboflow only
exports versions.

Consequences. `make download` reports `lava` as a failure with the reason
"project or version not pinned". The spec's class list for `marswars` was
missing `red_active`. `testingfrfr` and `scorekeeper` versions hold far
more images than their source projects (see the resolution file), which
suggests augmented copies in the train split. Task 3 has to check this before
either is used for evaluation.

## D-009: Commit per-directory image digests instead of a per-file list (2026-09-26)

Context. SPEC section 3.5 asks for a small committed hash file per dataset.
A `sha256sum` line for every image came to megabytes for the larger
datasets, over the 500 KB large-file hook.

Decision. `reports/data_manifests/<key>.sha256` lists each annotation and
README file with its own `sha256sum` line, plus one comment line per image
directory with the file count and one hash over that directory's image
hashes. The full per-file list stays in the gitignored
`data/raw/<key>/MANIFEST.json`.

Why. Each file stays under a kilobyte, and a re-download still shows in
`git diff` which annotation file or split changed. `sha256sum -c` checks
the annotation lines. It warns that the comment lines are improperly
formatted but still exits 0, so `--strict` cannot be used.

Consequences. A changed image shows up as a changed directory hash, and
finding the exact image needs the local manifest.

## D-010: Inspection thresholds and the face policy for sample grids (2026-09-26)

Context. `inspect_datasets.py` needs box-size buckets, a perceptual-hash
distance for near duplicates, and a rule for which images may appear in the
published sample grids.

Decision. Box sizes are bucketed by their share of the image area, with
edges at COCO's small and medium limits (32 and 96 pixels square) scaled to
a 640 pixel square image. Two images are near duplicates when their 64-bit
pHashes differ in at most 4 bits. Grid images in which a person's face can
be made out are listed by file name under `inspect.grid.exclude` in
`configs/project.yaml`, and distant crowds in overhead broadcast shots are
kept.

A box counts as invalid when it has no area or overhangs the image edge by
more than 1 pixel.

Why. Fractions keep the buckets comparable across datasets whose images
range from 512 pixels square to 1920 by 1080. pHash sets about half of its
bits, so distances are almost always even. Contact sheets of pairs at
distance 2 and 4 with file names from different videos showed the same
scene every time. Most were frames of one fixed-camera broadcast seconds
apart, which is the kind of match that leaks between train and test. The
1 pixel tolerance allows for rounding in exported coordinates. When it is
unclear whether a face can be made out, the image is excluded. Grid sampling shuffles once and skips excluded files, so each
exclusion swaps in one new image to check instead of a new grid.

Consequences. The near-duplicate counts in `reports/duplicates.json`
include adjacent video frames, not only re-uploads of the same photo.
Anyone regenerating the grids with new data has to check the faces again.

## D-011: Use marswars as Dataset A, with robotzftp2 and scorekeeper (2026-09-26)

Context. SPEC section 5 asks for 2 to 3 datasets and a baseline Dataset A
chosen from Task 2's inspection. `docs/DATASETS.md` describes all five. No
dataset labels REBUILT robots, and only `scorekeeper` and `testingfrfr` label
robots at all.

Decision. Dataset A is `marswars` version 5, B is `robotzftp2` version 1, and
C is `scorekeeper` version 1 reduced to one image per source photo.
`testingfrfr` and `robotzftp2_fuel` are not used for evaluation. `testingfrfr`
may be used as training data for the merged model only after every image
matching a test image is removed.

Why. `marswars` is the only robot-camera dataset, matching where the detector
runs, and it shares no near duplicates with any other dataset. `robotzftp2`
differs from it in viewpoint, venue, lighting, and resolution, and has no
augmented copies. `scorekeeper` adds robot labels and a tabletop label style.
`testingfrfr` overlaps heavily with B and C, and `robotzftp2_fuel` is one
large ball per image.

Consequences. The baseline is trained and scored on fuel only, so robot
results start with the merged model and cover robots from past games only.
Task 4 has to group every split by recording and, for `scorekeeper`, by photo,
since source names alone undercount photos. Removing `testingfrfr` matches to
test images needs a hash check that also covers flipped and rotated copies.

## D-012: Map every label to fuel, robot, or DROP (2026-09-27)

Context. SPEC section 4 fixes two classes and asks for every source label
to be mapped explicitly. The five downloaded datasets use four names for
fuel, and `marswars` also labels the hub's lit state.

Decision. `configs/class_map.yaml` maps `game_piece`, `fuel`, `FUEL`, and
`Fuels` to fuel and `robot` to robot. The `marswars` labels `blue_active`,
`red_active`, and `inactive` map to DROP. Their boxes are removed and their
images kept, so an image that only showed a dropped label becomes a
background image. Roboflow's placeholder category at id 0 is ignored because
no annotation uses it. Harmonized files hold exactly two categories, fuel
with id 1 and robot with id 2.

Why. The state classes record whether the hub is lit, which is field state
rather than an object to detect. supervision builds its class list from the
category names sorted by id, so a leftover placeholder would shift every
class id by one. In `robotzftp2` the placeholder is even named `fuel`, the
same as the real class. A test checks the class map against
`reports/class_counts.csv`, so a label found by a later inspection cannot be
missed. An unmapped label raises `UnmappedLabelError`.

Consequences. Every harmonized dataset has the same class ids, including
datasets that label only fuel. `reports/class_coverage.json` records which
classes each dataset labels, and evaluation scores only those.

## D-013: Split A and B by frame order and C by related-image groups (2026-09-27)

Context. The Roboflow splits leak in all three chosen datasets (D-011).
Adjacent video frames sit on both sides, `robotzftp2` has no test split, and
`scorekeeper` spreads copies of one photo across splits. `marswars` has only
13 recordings, and a few of them hold most of its fuel boxes, so holding out
whole recordings made test depend on which two or three recordings landed
there. It could also put all the official-field frames in one split.

Decision. `marswars` and `robotzftp2` are cut by frame order within each
recording: the first 70 percent of images go to train, the next 15 percent
to valid, and the last 15 percent to test. Valid therefore sits between train
and test in time. An image is dropped when the first frame of the next split
is at most 5 frames after it. Recordings under 20 images go whole to train,
and so do FIRST's official videos in `marswars`. `scorekeeper` joins images
that share a video recording, a source name, or a pHash within 4 bits, and
assigns whole groups, largest first, to the split furthest below its target,
with ties ordered by a seeded hash. A final pass drops any train or valid
image within 4 bits of a test image, and the run fails if one remains.
Settings are under `splits` in `configs/project.yaml`.

Why. Before choosing, we measured how alike each test frame is to its
nearest train frame under several candidate splits. With random blocks of
consecutive frames, a large share of `marswars` test frames had a near
duplicate in train even with a buffer, because the robot often sits still or
comes back to the same spot. Cutting each recording in frame order gave test
frames about as far from train as holding out whole recordings did, while
keeping every Basler run in test. Frames 5 apart were rarely near
duplicates. The official videos are edited from many shots, so their last
frames are a different scene, not later in one run. Cut in frame order, a
few dense full-field frames held most of A-test's fuel boxes and would have
set the in-domain score. `scorekeeper` has hundreds of groups, so grouped
assignment is not lumpy there.

Consequences. Test holds 0 near duplicates of train or valid by
construction (see `reports/splits.json`). A-test and B-test measure "later
in the same runs", not "a new run". They share each run's lighting, balls,
and background with train, which a pixel-similarity check cannot rule out.
A-test has no official-field frames. Each test split rests on few
independent scenes (6 recordings for A and 7 for B), so Task 6 should report
that count and consider a bootstrap over recordings for confidence
intervals.

## D-014: Keep one augmented copy per scorekeeper photo (2026-09-27)

Context. D-011 reduces `scorekeeper` to one image per source photo. Its
train split holds three augmented copies of each photo, made with random
rotation, exposure, and blur. Some source names are shared by several
photos.

Decision. Train images are grouped by source name plus the number of boxes
of each class. From each group the copy with the smallest total box area is
kept, with ties going to the first file name. Valid and test are not
augmented and are left alone.

Why. pHash cannot find these copies. During planning, copies of one photo
were usually as far apart in pHash bits as unrelated images, because
rotation moves the whole picture. Rotation and exposure changes keep the
number of boxes of each class, so the name and those counts separate photos
that share a name. Rotating a box and taking its axis-aligned hull only
makes it larger, so the smallest total area picks the least rotated copy. A
contact sheet of copy groups confirmed this.

Consequences. `reports/splits.json` lists the copy group sizes. Most groups
hold exactly three copies. A few groups of six or nine are different photos
that share both a name and box counts, and they lose all but one photo. A
few groups of one or two are most likely copies that lost an edge box to
rotation, and they keep an extra copy. The untouched original is usually
not among the copies, so the kept image is still slightly rotated, with
slightly loose boxes, and its exposure or blur is changed.
