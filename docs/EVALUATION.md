# Evaluation

This page explains how models are scored and lists every published result.
Everything between the `EVALUATION` markers below is written by
`make report` from the runs under `reports/runs/`. Do not edit it by hand.

## Reading the numbers

A detector draws boxes and gives each one a confidence between 0 and 1,
which is how sure the model is that the box holds an object of that class.
A predicted box counts as a hit when it overlaps a labeled box of the same
class enough. Overlap is measured as intersection over union (IoU): the area
the two boxes share divided by the area they cover together. Each labeled
box can be matched by one prediction at most.

- Precision is the share of predicted boxes that are hits. Low precision
  means the model sees fuel that isn't there.
- Recall is the share of labeled boxes that a prediction found. Low recall
  means the model misses fuel.
- Both depend on the confidence threshold. Raising it drops unsure
  predictions, which usually raises precision and lowers recall. The tables
  give both at one threshold, and a sweep over thresholds for each class.
- mAP50 (mean average precision at IoU 0.5) summarizes precision at every
  level of recall, using all predictions regardless of confidence. It is 1
  when every labeled box is found before any false positive. mAP50-95
  averages the same score over stricter overlap requirements, from IoU 0.5
  up to 0.95, so it also rewards boxes that sit tightly on the object.

Every test split is small and comes from a few recordings, and frames from
one recording look alike. The intervals next to mAP50 and recall show how
much a score moves when whole recordings (or, for scorekeeper, groups of
related photos) are resampled with replacement. A wide interval means the
score depends heavily on which recordings happen to be in the test split.

A dataset is scored only on the classes it labels, as recorded in
`reports/class_coverage.json`. A robot prediction on a fuel-only dataset is
neither a hit nor a false positive.

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
`predictions.json` holds every predicted box down to the confidence floor in
`configs/project.yaml`, so scores at any threshold can be recomputed with
`--from-cache <run_id>` without calling the model again. `metrics.json`
holds the scores, and `meta.json` records the commit, config and data
hashes, the model entry from `reports/models.yaml`, the backend the server
reported, and package versions. A run refuses to start from a tree with
uncommitted changes unless `--allow-dirty` is passed, and the report leaves
such runs out, along with runs on a slice of a split.

## Limits

The model runs on Roboflow's hosted service, which reported a TensorRT
fp16 backend, so scores can differ slightly from those on the model's page
in the Roboflow app. The service applies the dataset version's own 384x384
resize before running the model, and boxes are scored in the original
image's pixels.

Box sizes follow COCO's fixed pixel areas on the original image. The
datasets have different resolutions, so size results are not comparable
between datasets, or with the Roboflow app, which measures sizes on the
resized image.

Precision and recall come from supervision, which matches a prediction to a
box of the same class with IoU of at least 0.5. The hit, false positive, and
miss counts come from the confusion matrix, which matches boxes by overlap
alone and needs IoU above 0.5. A box drawn with the wrong class is a
confusion between two classes there, and a false positive for one class and
a miss for the other in precision and recall. With one scored class the two
agree except for a box at exactly 0.5.

The field test set has not been scored yet.

## Results

<!-- EVALUATION:START -->
No model has been scored on a whole split yet.
<!-- EVALUATION:END -->
