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
there. Holding out whole recordings could also put every official-field
frame in one split.

Decision. `marswars` and `robotzftp2` are cut by frame order within each
recording. The first 70 percent of images go to train, the next 15 percent
to valid, and the last 15 percent to test, so valid sits between train and
test in time. An image is dropped when the first frame of the next split is
at most 5 frames after it. Recordings under 20 images go whole to train, and
so do FIRST's official videos in `marswars`. `scorekeeper` joins images that
share a video recording, a source name, or a pHash within 4 bits, and
assigns whole groups, largest first, to the split furthest below its target,
with ties ordered by a seeded hash. A final pass drops any train or valid
image within 4 bits of a test image. The run fails if one remains, or if two
re-split datasets share a near duplicate. Settings are under `splits` in
`configs/project.yaml`.

Why. An exploratory check during planning, which `make harmonize` does not
reproduce, compared how alike each test frame is to its nearest train frame
under several candidate splits. With random blocks of consecutive frames,
many `marswars` test frames had a near duplicate in train even with a
buffer, because the robot often sits still or comes back to the same spot.
Cutting each recording in frame order left test frames about as far from
train as holding out whole recordings did, while keeping every Basler run
long enough to cut in test. In the same check, frames 5 apart were rarely
near duplicates. The official videos are edited from many shots, so the end
of a video shows a different scene, not a later moment of the same run. Cut
in frame order, a few dense full-field frames held most of A-test's fuel
boxes and would have set the in-domain score. `scorekeeper` has hundreds of
groups, so grouped assignment is not lumpy there.

Consequences. By construction, no test image has a near duplicate in train
or valid. A-test and B-test hold later frames of runs seen in training. They
share each run's lighting, balls, and background with train, and the pHash
check cannot detect that kind of similarity. A-test has no official-field
frames. Each test split rests on few independent scenes (6 recordings for A
and 7 for B, see `units` in `reports/splits.json`). A-test is thinner than
that count suggests. One of its recordings has no fuel, and the densest
Basler run holds most of its fuel boxes. Task 6 should report fuel boxes per
recording in each test split, and consider error bars from resampling whole
recordings (a bootstrap).

## D-014: Keep one augmented copy per scorekeeper photo (2026-09-27)

Context. D-011 reduces `scorekeeper` to one image per source photo. Its
train split holds three augmented copies of each photo, made with random
rotation, exposure, and blur. Some source names are shared by several
photos.

Decision. Train images are grouped by source name plus the number of boxes
of each class. From each group the copy with the smallest total box area is
kept, with ties going to the first file name. Valid and test are not
augmented and are left alone.

Why. pHash cannot find these copies. Rotation moves the whole picture, so
copies of one photo are often as many bits apart as unrelated images.
Rotation and exposure changes keep the number of boxes of each class, so the
name and those counts separate photos that share a name. Rotating a box and
taking its axis-aligned hull only makes it larger, so the smallest total
area picks the least rotated copy. We checked this by eye on a few copy
groups. The contact sheet is not published, since such sheets can show faces.

Consequences. `reports/splits.json` lists the copy group sizes. Most groups
hold exactly three copies. A few groups of six or nine are different photos
that share both a name and box counts, and they lose all but one photo. A
few groups of one or two are most likely copies that lost an edge box to
rotation, and they keep an extra copy. The untouched original is usually
not among the copies, so the kept image is still slightly rotated, with
slightly loose boxes, and its exposure or blur is changed.

## D-015: Upload each split by name and check the version before training (2026-09-27)

Context. Task 5 trains the baseline on the Roboflow platform from
`data/harmonized/marswars/`, and D-013's leak guarantees hold only if
Roboflow keeps every image in its split. In `roboflow` 1.5.1 a directory
upload guesses each image's split from its path and defaults to train, the
zip upload leaves the split to the server, and a failed image upload is
printed rather than raised. The SDK's folder parser also read `info` and
`licenses` from every COCO file, which harmonized files did not have.

Decision. `frc-upload` uploads train, valid, and test in separate calls with
`split` set, only to a project that already exists, after checking split
sizes against `reports/splits.json` and the images against the field test
set. `frc-verify-upload` downloads the generated version and matches every
image to its harmonized file by upload name. Training starts only after it
passes. Harmonized COCO files carry empty `info` and `licenses`. The
version keeps auto-orient, adds a resize only if the architecture requires
one, and has no augmentation and no null filter.

Why. Parsed one folder at a time, the SDK put every harmonized image in
train. Checking counts in the web app would miss an image that moved from
test to train while another moved back. Names survive Roboflow
re-encoding the images, while hashes do not. The export drops the
`.rf.<hash>` suffix and a `_jpg` before it, so both sides are compared
without them. On the first real export this matched every image exactly
once, and the box count check catches an image
whose annotations failed to upload. Filtering nulls would drop the
background images that D-012 keeps.

Consequences. Each platform run leaves
`reports/platform_upload_<key>.json` with the version's preprocessing and
augmentation as the API reports them, and `reports/models.yaml` holds the
settings from the training page. The architecture and input size are
recorded there once the run is set up.

## D-016: Train the baseline on a 384x384 stretched version (2026-09-27)

Context. D-015 left the resize to whatever the architecture asks for. On
version 1, which had no resize, Roboflow's training page warned that
384x384 is recommended for RF-DETR Nano. The `marswars` frames are wider
than they are tall and larger than 384 pixels on each side.

Decision. The baseline is RF-DETR Nano trained on version 2, which adds
"Resize, Stretch to 384x384" and is otherwise identical to version 1.
Stretch was chosen over Fit, which pads with black borders.
Version 1 stays in the project unused. Settings and the platform's own
test-split numbers are in `reports/models.yaml`.

Why. The model runs at 384x384 either way, so putting the resize in the
version records the pixels it trained on. Stretch matches how the model is fed at inference time when the
version's preprocessing is reused.

Consequences. Stretching changes the aspect ratio, so round fuel becomes
slightly oval, and shrinking full frames to 384 pixels leaves distant fuel
only a few pixels across. The baseline is likely weakest on small boxes,
which Task 8's box-size slices should measure. The numbers on Roboflow's
model page use a confidence threshold that the platform picked on the
test split, so they are recorded for reference only and are not quoted as
results.

## D-017: Evaluate hosted models through inference-sdk, installed on request (2026-09-27)

Context. Task 6 scores models on harmonized splits. The baseline is a
Roboflow platform model, and SPEC section 7 also allows local weights
through `inference`. The `infer` extra pulls in CPU-only torch, so it is kept
out of `make setup` and CI (D-001).

Decision. `frc-evaluate` calls the hosted model with `inference_sdk`'s
`InferenceHTTPClient` against `https://serverless.roboflow.com`, imported only
when a model is called. `make setup-infer` installs the extra, and
tests that need a real model are marked `integration`. The model is named as
`<project>/<version>` from `reports/models.yaml`. Local weights are left for
Task 12, behind the same predictor interface. The API key travels in both the
query string and an `Authorization` header (`api_key_transport="both"`), and is
redacted from logs either way. Sending it only in the header needs server
support that was not checked for the hosted service, and naming the transport
stops the SDK's warning that none was chosen.

Why. With inference-sdk 1.7.2, a probe on one A-test image showed
`frc-rebuilt-fuel-a/2` resolving to the trained model
`frc-rebuilt-fuel-a-2-rfdetr-nano-t1`, the same model as the
workspace-qualified name, so the workspace never has to be looked up. The
hosted endpoint applies the version's own 384x384 stretch and returns boxes in
the original image's pixels. Its response also names the workspace, which is
made from an email address, so raw responses are never stored. The evaluator keeps only boxes, class names, and
confidences, and checks that the resolved model name (the part after the
slash) matches `model_id`.

Consequences. Every evaluation needs the network and an API key, and hosted
calls count toward the account's usage. CI stays offline and never installs
torch. The server may change backends (TensorRT fp16 on the probe), so the
backend and quantization it reports are recorded in each run's `meta.json`.

## D-018: How frc-evaluate scores a run (2026-09-27)

Context. Task 6 needs scores that can be compared across datasets and
re-computed later, from test splits that come from a few recordings each
(D-013).

Decision. mAP50 and mAP50-95 use every prediction down to a confidence
floor of 0.01. Precision, recall, and the confusion matrix count predictions
at or above 0.5, a neutral default until Task 11 picks a threshold for the
robot. Only the classes in `labeled` in `reports/class_coverage.json` are
scored, and predictions of other classes are dropped before scoring.
Precision is stored as null when a class has no predictions, and a box size
with no labeled boxes as null instead of COCO's -1. Predictions are cached
with confidence to 4 decimals and boxes to 0.1 pixel, and scores are stored
to 3 decimals and printed unchanged by `make report`. A live run is scored
from the rounded values it caches, so `--from-cache` reproduces it exactly,
and a rescore must name the same model, dataset, and split as the run it
reads. A scored class with no labeled box in the split has no mAP or recall,
and a bootstrap resample with no labeled box of a class is left out of that
class's interval. Intervals come from
1000 resamples of whole recordings (temporal datasets) or related-image
groups (scorekeeper), rebuilt with the split's own rules and checked
against the unit counts in `reports/splits.json`. The report publishes the
newest clean whole-split run for each model, dataset, and split, and leaves
out a rescore of a run made from a dirty tree.

Why. Mean average precision needs the low-confidence tail, and the floor
keeps a full split's predictions small enough to commit. The platform's
0.46 threshold was picked on the test split, so it is not reused. Frames
from one recording are alike, so resampling images would give intervals
that are too narrow. A full A-test run at the platform's threshold came
within a few boxes of the counts on the model's page, which checks the
harness against an independent evaluation. Size scores differed widely from
the platform's, because COCO's size buckets are fixed pixel areas and the platform
measures them on the 384x384 resized image while `frc-evaluate` uses the
original. For the same reason size scores are not comparable between
datasets of different resolutions.

Consequences. Hits, false positives, and misses come from supervision's
confusion matrix, which matches boxes regardless of class and needs IoU
above 0.5, while precision and recall match within a class at IoU 0.5 or
more. With one scored class these agree except for a box at exactly 0.5.
A bootstrap takes a few minutes per split. Task 8's box-size slices should
use sizes relative to the image, as `inspect.area_buckets` does, instead of
COCO's buckets.

## D-019: Score only the classes both the dataset and the model know (2026-09-27)

Context. baseline-a was trained on `marswars`, which labels fuel only.
`scorekeeper` labels robots too, so scoring it on every labeled class would
publish a robot row with mAP50 and recall of 0. It would also halve the
run's overall mAP, which averages over classes.

Decision. Nick chose on 2026-09-27 that a run scores the classes the dataset
labels that the model's training dataset also labels, both read from
`labeled` in `reports/class_coverage.json`. The rest are stored as
`unscored_classes` in `metrics.json`, and the report names them under the
run's table.

Why. A zero for a class the model was never shown says only that its
training data lacked that class. The project asks how a fuel detector does
on other teams' fuel.

Consequences. The baseline's cross-dataset results are about fuel only.
`scorekeeper`'s robot boxes are first scored by a model trained on robots in
Task 10. A model's training dataset needs an entry in the coverage report,
so a merged dataset must write one.

## D-020: Keep the 25 most confident predictions per image (2026-09-27)

Context. At the 0.01 confidence floor, baseline-a returned so many faint
boxes on `scorekeeper` test that its prediction cache would have been
2922540 bytes, over the 450000 byte limit that keeps it under the 500 KB
large-file hook. The run stopped before writing anything. A-test and B-test
had fit, with up to 107 predictions on one image.

Decision. Nick chose on 2026-09-27 to cache and score only the 25 most
confident predictions per image (`max_predictions_per_image` in
`configs/project.yaml`) on every dataset. Ties keep the model's order. A
rescore of an older cache applies the same limit. A-test and B-test were
rescored under it from their earlier caches, which hold every box down to
the floor, so the model was not called again.

Why. No test image of A, B, or C has more than 16 labeled fuel boxes, so 25
still lets every labeled box be found. COCO's own evaluation keeps at most
100 per image and class. The cache stays in git, so `--from-cache` works from a
fresh clone. Raising the floor instead would cut the low-confidence tail on
every image, including sparse ones.

Consequences. The dropped boxes can include low-confidence hits as well as
false positives, so mAP can move either way compared with no limit. On
A-test it came out slightly lower (compare the two A-test runs'
`metrics.json`). When every labeled box survives the limit, as on
`scorekeeper` test where recall is 1.0 even at the lowest threshold, only
false positives were dropped, so mAP there is an upper bound. When an image
reaches the limit with every kept box at or above the threshold, boxes that
would have counted at the threshold were dropped, so that run's false
positive count is a lower bound and its precision an upper bound. Each run
records both counts under `per_image_limit` in `metrics.json`, and the
report prints them under the run's table. The limit counts boxes of every
class the model predicts, before unscored classes are dropped, so a model
that also predicts robots could lose fuel boxes to it on a fuel-only
dataset. That needs another look before Task 10. The first A-test and
B-test runs, made without the limit, stay under `reports/runs/`, since the
rescores read their caches, and the report uses the newer rescores.

## D-021: How frc-diagnose slices errors and judges them by eye (2026-09-27)

Context. Task 8 asks where baseline-a's errors fall, whether C's false
positives land on photos without fuel, why B is no harder than A, and which
errors come from labeling style. Numbers in the docs must come from
generated reports, and the diagnosis should not call the hosted model
again.

Decision. `frc-diagnose` reads the caches behind each published run and
matches boxes by the rule supervision's confusion matrix uses: pairs of the
same class first, then by IoU from high to low, with IoU strictly above 0.5.
The Task 8 plan said confidence first, but only this rule reproduces a run's
counts, and the command fails if any slice or whole split disagrees with
them. A false positive is a duplicate, a localization error (IoU with a
label above `localization_floor`, 0.1), inside an unscored label such as a
robot, or background. The floor is low enough to count a loose or offset box
on a real ball as a localization error, and high enough to leave out boxes
that barely touch a label. Box sizes are shares of the image area
(`inspect.area_buckets`). Brightness and sharpness (variance of the
Laplacian) are measured on the image stretched to 384 pixels, as the model
sees it (D-016), and cut into three equal-count bins pooled over the
diagnosed test splits. Crowding bins start at 0, 1, 2, 5, and 10 labeled
boxes. C is sliced by kind of photo, from patterns on the source name, and A
and B by recording. Errors are judged by eye from numbered crops: every
false positive and miss when a split has at most 40 of a kind, and a seeded
sample of 40 otherwise. Verdicts live in `reports/diagnosis/review.csv`, and
the run fails if they do not match the sample. The gallery alternates false
positives and misses, spreads over sources, and keeps frames of one
recording at least 30 apart. `frc-report` draws the figures from the
diagnosis, so `make report` rebuilds them with the tables, and every bar is
labeled with its value from the JSON.

Why. Matching the run's own counts ties every slice to the published
numbers. Relative sizes and the 384 pixel view make datasets of different
resolutions comparable. Pooled bins avoid picking edges after seeing the
results. A sample of 40 is small enough for one person to review. The
review's verdict list grew during the first pass, when people, other yellow
objects, and balls cut off by the image edge turned up.

Consequences. Slices have no intervals. On C, the brightness and sharpness
bins mostly separate fuel photos from broadcasts, so they are confounded
with source. One person judged the review, in a single pass. The face check
left out five gallery tiles and the whole 2024 Milford broadcast, whose
frames all show spectators close to the camera, so the largest source of C's
false positives has no tile. A gets 3 tiles instead of 4, since its other
errors are excluded or within 30 frames of a pick, and the command logs a
warning when that happens. The check against a run's published counts
matches every scored class at once, so it also holds for the two-class
models of Task 10.

## D-022: Choose the lockbox dataset after v0.1.0 (2026-09-27)

Context. With no field test set yet, a Universe dataset held back until
Task 10 was proposed as an unbiased check of the fix. Ideally it is chosen
before the diagnosis can shape the choice.

Decision. Nick chose on 2026-09-27 to go ahead with Task 8 and pick the
lockbox after v0.1.0.

Why. The v0.1.0 deadline leaves no time to look up and check new Universe
datasets before the write-up, and Task 8 does not need one.

Consequences. The lockbox is picked knowing that people and balls from
other games cause most false positives on C. Its decision entry must say so,
and it must still pass the duplicate check against every training source.

## D-023: The Colab notebook rescores committed predictions (2026-09-27)

Context. SPEC section 7 asks for a thin Colab notebook that reproduces
download, harmonize, and eval. baseline-a is a hosted Roboflow model named
by project and version, and the hosted API looks it up in the caller's own
workspace (D-017), so another person's key can't call it. The `infer` extra
pulls CPU-only torch, which would replace Colab's CUDA build.

Decision. Nick chose on 2026-09-27 that `notebooks/reproduce_baseline.ipynb`
clones the release, installs `requirements.txt` and the package without
extras, downloads and harmonizes with the reader's own key, and runs
`frc-evaluate --from-cache` on the predictions behind each published run.
It compares the new `metrics.json` with the committed one and removes the
new run folder, so the next rescore starts from a clean tree without
`--allow-dirty`. A live call is left as an optional, commented-out cell
that installs `inference-sdk` alone.

Why. A rescore needs no model access and no inference client, and it
reproduces the published scores exactly (D-018), so anyone can check them.
The notebook's name differs from SPEC section 6's `train_and_eval.ipynb`,
since training runs on the platform and isn't in the notebook.

Consequences. The notebook checks the data pipeline and the scoring, not
the model's predictions, which only the training workspace can make again.
From a fresh clone in a new Python 3.11 environment, all three published
runs matched. The Colab-only cells (secrets and the Colab runtime) have not
been run on Colab.

## D-024: Skip unpinned datasets instead of failing the download (2026-09-27)

Context. D-008 left `lava` in `configs/datasets.yaml` without a version,
since it has none published, and `make download` recorded it as a failure.
`frc-download` returned 1 on every run, so `make download harmonize`
stopped before harmonizing, and the Colab notebook showed an error that
meant nothing had gone wrong.

Decision. Nick chose on 2026-09-27 that a dataset with no pinned project or
version is skipped with a warning and is not a failure.
`reports/download_failures.json` now lists only datasets that were tried
and failed, and is empty on a clean run.

Why. An unpinned entry is a candidate kept for the record, with nothing to
export, so it is not an error. Keeping it in the config documents why it was
not used (`docs/DATASETS.md`). A real failure, such as a network error or a
bad key, still exits 1.

Consequences. The warning is the only trace of a skipped dataset in a run.
A dataset unpinned by mistake would also be skipped without stopping the
run, and `frc-harmonize` only warns about a dataset with no manifest, so
the two warnings and the missing harmonized dataset are what would show it.

## D-025: Limit predictions per class, after dropping unscored classes (2026-09-28)

Context. D-020 keeps the 25 most confident predictions per image, counted
over every class the model predicts, before the classes a run does not
score are dropped. A model trained on fuel and robots could then lose fuel
boxes on a fuel-only dataset to confident robot boxes that are never
scored, and on a dataset with both classes one class could crowd out the
other.

Decision. Predictions of classes the run does not score are dropped first,
and then each remaining class keeps its 25 most confident boxes per image.
The cache, rescores of older caches, and `frc-diagnose` all apply the same
rule. A run counts an image as reaching the limit when any of its scored
classes does. The setting keeps its name, `max_predictions_per_image`, and
so does the `per_image_limit` record in `metrics.json`, so the published
runs still load.

Why. COCO's own evaluation keeps its limit per image and class. A class the
run does not score cannot change its scores (D-019), so it should not change
which boxes are kept either. Dropping those boxes before caching also keeps
the cache smaller.

Consequences. baseline-a predicts only fuel, so every published run keeps
the same boxes under this rule, as rescoring every cache under
`reports/runs/` shows, and its scores are unchanged. A two-class model's
cache can hold up to twice as many boxes per image. The size limit still
stops a run before it writes anything, so a cache too large for C would
fail the run rather than be cut silently. Robot predictions on a
fuel-only dataset are not cached, so they cannot be studied later without
calling the model again.

## D-026: Use pankratz as the lockbox, picked after the diagnosis (2026-09-28)

Context. D-022 put off choosing a lockbox until after v0.1.0. Task 10 needs
a dataset that no model is trained on and that no step of the fix was tuned
on, to check the merged model with. The field test set is still not
confirmed.

Decision. Nick chose on 2026-09-28 to use version 6 of Joshua Pankratz's
`myworkspace-mliyg/frc-2026-rebuilt-fuel-detection` as the lockbox, under
the key `pankratz`. It is harmonized with a new `eval_only` split method,
which puts every image in test and groups related images the way the
grouped method does, so its error bars resample whole groups. It is never
trained on, and `frc-merge` protects it like a test split. The sample grid
leaves out every source name starting with `WIN_` (a new
`inspect.grid.exclude_patterns` setting) and one more photo, since the
webcam frames all show the same person's face close to the camera.

Why. A Universe search on 2026-09-28 found seven more REBUILT fuel
projects, but all except this one reuse data already in the project. Three
workspaces share one 2784-image aggregate labeled with `robotzftp2_fuel`'s
`Fuels` among other names, one has exactly `robotzftp2_fuel`'s split sizes,
and a fourth and its forks use `Fuels` too. This one has its own label name,
camera, and venues. Every lockbox image was also hashed under the 8 flips and
90 degree rotations and compared with every image of the other five
datasets. None came within 4 bits. Six `testingfrfr` images came within 8
bits and were checked by eye, and each is a different photo of a single ball
on carpet.

Consequences. The lockbox was picked knowing that people and balls from
other games cause most of the baseline's false positives on C. Its main
scene is a person close to the camera holding and tossing fuel, so it tests
that failure, and a better score on it cannot be read as an
unbiased sample of new footage. It is small, 320 by 240, and from one team's
workshop, and the webcam frames make up most of it, so it adds one new
domain. It labels fuel only, so robots are not scored on it.

## D-027: Score fuel labels at the frame edge as neither hits nor misses (2026-09-28)

Context. The diagnosis found that most of A's misses, all of B's, and half
of A's false positives are balls cut off by the frame edge, where the label
and the model disagree on how much of the ball to box (D-021). The Task 10
plan was to drop thin edge labels in every split. On A-test, dropping fuel
labels that touch the edge and are less than half as wide as they are long
would remove 15 of the 24 edge-cut misses, but also 38 edge labels the model
already hits, which would then count as false positives. The edge-cut misses
range from very thin strips to nearly square boxes, so no aspect ratio
separates them from the hits.

Decision. Nick chose on 2026-09-28 to leave the labels alone and change the
scoring instead, the way COCO treats crowd regions. A fuel label within 1
pixel of the image edge is left out of scoring, so it is neither a hit nor a
miss. A fuel prediction is dropped with it when their overlap, divided by
the smaller box's area, is above 0.5, unless the prediction has IoU of at
least 0.5 with a fuel label that is scored. Settings are under
`evaluate.edge_ignore` in `configs/project.yaml`. Each run records the
settings and how many labels and predictions were left out, and
`frc-diagnose` applies the rule only to runs that recorded it. Training data
keeps every label.

Why. Ignoring needs no threshold chosen by looking at results, so the same
rule applies to every dataset. The overlap over the smaller box, rather than
IoU, is what lets a prediction that is much larger or smaller than a thin
edge label still be dropped, which is the disagreement being set aside. A hit on a scored ball is never dropped, so the rule cannot hide a
correct detection. A robot on the frame's edge still counts, since the
diagnosis found no problem there.

Consequences. Scores say nothing about balls cut off by the frame edge,
which a robot does see. Recall and precision are measured on the balls fully
in view. The rule also sets aside large balls that only touch the edge,
which the model mostly finds, so it removes some easy hits along with the
hard cases. A false positive at the edge that overlaps no label is still
counted. The v0.1.0 runs stay committed and published as they were until
baseline-a is rescored from its caches under this rule, which happens with
the merged model's runs so that tables and text change together.

## D-028: Build the merged training set with frc-merge (2026-09-28)

Context. Task 10 retrains on merged data aimed at the diagnosed failures,
which are people and other games' balls taken for fuel. Every source keeps
its own test split, so the merged model can be compared with the baseline
on the same images. `testingfrfr` overlaps with B and C and holds flipped
and rotated copies that a plain pHash check misses (D-011).

Decision. Nick chose on 2026-09-28 to train one two-class model on fuel and
robots. `frc-merge` writes `data/harmonized/merged/` from A, B, and C with
their splits as they are, and from `testingfrfr` in train only.
`robotzftp2_fuel` is left out. Three sets of images are left out of train
and valid. A's frames from FIRST's official videos go, since their robots
have no robot labels and a two-class model would learn them as background.
C's fuel photos go, since their clusters have unlabeled balls (D-021). And
`testingfrfr` keeps one augmented copy per photo, chosen as in D-014. A
`testingfrfr` image is also dropped when its source name belongs to any
valid, test, or lockbox image, when it is a frame of a recording those
images come from, or when its pHash under any of the 8 flips and 90 degree
rotations is within 4 bits of one. A frame counts when harmonize would have
kept it out of train: any frame of a held-out recording in a dataset split
by groups, and in one cut in frame order, any frame at or after the first
held-out frame less the 5 frame buffer (D-013). A photo whose copy is
dropped for any of these reasons loses all its copies. The run fails on a
field-test image anywhere in the output, and on a merged train image within
4 bits of any test or lockbox image under any transform. Counts are in
`reports/merge.json`, and the merged classes in
`reports/class_coverage.json`.

Why. Keeping each source's test split whole is what makes the comparison
with the baseline fair. `testingfrfr` brings robots from five seasons and
broadcast frames full of people, which are the negatives the diagnosis asks
for, including 2020 frames whose yellow balls carry no fuel label. Its
source names repeat the other datasets' files, so matching names catches
copies that shear or exposure changes put out of pHash reach. Train frames
close to their own valid split are allowed, as they were for the baseline
(D-013). The first run found 16 such pairs in A and B, all within one
recording and none needing a transform. A review of that run found that
pHash alone let in frames from recordings in C-test, such as the Milford
broadcast behind half of it, and B frames inside the buffer before a test
cut, since other frames of one scene are not near duplicates. It also found
photos whose dropped copy matched a held-out image while the kept copy did
not. The recording and copy-group rules close both. Keeping every
augmented copy would have made most training images `testingfrfr` copies,
about three per photo, some with broadcast frames on their side.

Consequences. Matching by source name is coarse. Names such as `frame_12`
are shared by unrelated photos, so some clean images are dropped. The
merged valid split lacks C's fuel photos, so validation during training
sees fuel mostly from A and B. The pHash check still misses heavier edits,
such as a strong crop. `make harmonize` rewrites
`reports/class_coverage.json` without the merged entry, so run `make merge`
after it.

## D-029: Train the merged model twice, without and with augmentation (2026-09-28)

Context. SPEC section 7 asks for merged data plus augmentation aimed at the
diagnosed failure modes. Changing both at once would leave no way to say
which one moved the scores. The baseline used 0.71 credits.

Decision. The merged data is uploaded once and trained as two versions with
the baseline's architecture, checkpoint, and 384x384 stretch.
`merged-noaug` has no augmentation. `merged-aug` adds a horizontal flip, a
crop of 0 to 20 percent, and brightness of -25 to +25 percent, on train
only. If credits allow one run, it is `merged-aug`, and the write-up says the
two changes are confounded. `frc-verify-upload --augmented` checks the
augmented version. Train copies may repeat within train and their boxes are
not compared, while valid and test are checked in full.

Why. Crops make balls cut off by the frame edge, which caused most of A's
and all of B's misses. Brightness covers B's dark misses. The false
positives on people and other games' balls are left to the merged
negatives. Hue and saturation shifts would weaken the
color cue that separates fuel from other balls, and rotation loosens boxes,
as seen in `scorekeeper`'s copies (D-014).

Consequences. Two runs cost about twice the credits of one. The crop and
brightness ranges were chosen from the diagnosis, not tuned, and
Roboflow's random augmentation is not reproducible bit for bit. The edge
rule of D-027 leaves labels at the frame edge out of scoring, so gains from
the crop on cut-off balls will not show in the scores.
