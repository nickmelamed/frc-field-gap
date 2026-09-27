# Evaluation

This page explains how models are scored and lists every published result.
Everything between the `EVALUATION` markers below is written by
`make report` from the runs under `reports/runs/`. Do not edit it by hand.

## Reading the numbers

A detector draws boxes and gives each one a confidence between 0 and 1,
which is how sure the model is that the box holds an object of that class.
A predicted box counts as a hit when it overlaps a labeled box of the same
class enough. Overlap is measured as intersection over union (IoU), which
is the area the two boxes share divided by the area they cover together.
Each labeled box can be matched by one prediction at most.

- Precision is the share of predicted boxes that are hits. Low precision
  means the model sees fuel that isn't there.
- Recall is the share of labeled boxes that a prediction found. Low recall
  means the model misses fuel.
- mAP50 (mean average precision at IoU 0.5) sorts the predictions by
  confidence and measures how precise the model stays as it finds more of
  the labeled boxes. It is 1 when every labeled box is found before any
  false positive. It uses up to 25 predictions per image, the most
  confident ones, down to a confidence of 0.01.
- mAP50-95 averages the same score over stricter overlap requirements, from
  IoU 0.5 up to 0.95, so it also rewards boxes that sit tightly on the
  object.

Precision and recall depend on the confidence threshold. Raising it drops
unsure predictions, which usually raises precision and lowers recall. The
tables give both at one threshold, and a sweep over thresholds for each
class. A score shown as n/a is undefined. Precision is undefined when the
model predicted nothing of that class, and the other scores are undefined
when the split has no labeled box of that class.

Every test split is small and comes from a few recordings, and frames from
one recording look alike. The intervals next to mAP50 and recall show how
much a score moves when whole recordings (or, for scorekeeper, groups of
related photos) are resampled with replacement. A wide interval means the
score depends heavily on which recordings happen to be in the test split.

A dataset is scored only on the classes it labels, as recorded in
`reports/class_coverage.json`, and only on those the model's training data
labels too (D-019). A robot prediction on a fuel-only dataset is neither a
hit nor a false positive. A fuel-only model is not scored on robot boxes,
and the run's section says which classes were left out.

## Making a run

Scoring a hosted model needs the `inference` extra and a Roboflow API key in
`.env`:

```bash
make setup-infer
make eval MODEL=baseline-a DATASET=marswars                  # test split
make eval MODEL=baseline-a DATASET=marswars ARGS="--limit 10" # quick check
make report
```

Each run writes `reports/runs/<run_id>/` with three files.
`predictions.json` holds the 25 most confident boxes per image down to the
confidence floor in `configs/project.yaml` (D-020), rounded as described in
D-018. The run is scored
from those rounded values, so `--from-cache <run_id>` reproduces its scores
exactly without calling the model again, and can rescore them at another
threshold. A rescore must name the same model, dataset, and split as the run
it reads. `metrics.json` holds the scores, and `meta.json` records the
commit and tree, config and data hashes, the model entry from
`reports/models.yaml`, the backend the server reported, and package
versions.

A run refuses to start when the repo has uncommitted changes, unless
`--allow-dirty` is passed. The report leaves out those runs, rescores of
them, and runs on only part of a split.

## Limits

The model runs on Roboflow's hosted service, which reported a TensorRT
fp16 backend, so scores can differ slightly from those on the model's page
in the Roboflow app. The service applies the dataset version's own 384x384
resize before running the model, and boxes are scored in the original
image's pixels.

Box sizes follow COCO, which calls a box small when its area is under 32x32
pixels and large when it is over 96x96. They are measured on the original
image. The datasets have different resolutions, so size results are not
comparable between datasets, or with the Roboflow app, which measures sizes
on the resized image.

Precision and recall come from supervision, which matches a prediction to a
box of the same class with IoU of at least 0.5. The hit, false positive, and
miss counts come from the confusion matrix, which matches boxes by overlap
alone and needs IoU above 0.5. In the confusion matrix, a box with the wrong
class counts as a mix-up between two classes. In precision and recall, it
counts as a false positive for one class and a miss for the other. With one
scored class the two agree except for a box at exactly 0.5.

The field test set has not been scored yet.

## Results

<!-- EVALUATION:START -->
No model has been scored on a whole split yet.
<!-- EVALUATION:END -->
