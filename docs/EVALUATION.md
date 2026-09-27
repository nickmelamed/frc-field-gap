# Evaluation

This page explains how models are scored and lists every published result.
Everything between the `EVALUATION` markers below is written by
`make report` from the runs under `reports/runs/`. Do not edit it by hand.

## How the baseline does on other teams' data

baseline-a was trained on fuel from `marswars` (Dataset A) and scored on
the test splits of A, `robotzftp2` (B), and `scorekeeper` (C). The terms
are explained under "Reading the numbers" below.

On A, its own dataset, fuel mAP50 is 0.936 (interval 0.923 to 1.0), with
precision 0.963 and recall 0.927 at a confidence of 0.5.

On B, the model does not get worse. Fuel mAP50 is 0.994 (interval 0.985 to
1.0), with precision 0.966 and recall 0.973. So on these test frames, B is
not harder for this model than A. Task 8 looks at why.

On C, fuel mAP50 drops to 0.836 (interval 0.773 to 0.895). That interval
does not overlap A's, so the drop is larger than what the choice of test
photos alone would explain. Every labeled fuel box is found (375 hits,
recall 1.0), so the whole drop comes from false positives. The model also
draws 1083 boxes that match no label, which puts precision at 0.257.
Raising the threshold helps only partway. At 0.8, precision is 0.724 and
recall 0.981.

C's test split mixes two kinds of photos (see `docs/DATASETS.md`). One kind
shows fuel indoors. The other shows robots in pits and match broadcasts
from earlier games, one of which used balls, and none of those photos has a
fuel label. A first look at the saved predictions puts most of the false
positives on photos without fuel labels. If Task 8 confirms this, the model
is mistaking other round objects for fuel. That is a real model error, since
those photos correctly have no fuel labels.

These results have limits. A-test holds later frames of the recordings the
model trained on (D-013), so A's score is an optimistic reference point.
A and B each rest on a handful of recordings, which the intervals account
for. C is scored on fuel only, since baseline-a never learned robots
(D-019). Each image keeps its 25 most confident predictions (D-020). The
field test set has not been scored.

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
### baseline-a on marswars test

Run `baseline-a__marswars-test__20260927T215138Z`, scored on 171 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 6 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 455 | 0.936 | 0.923 to 1.0 | 0.623 | 0.963 | 0.927 | 0.908 to 1.0 | 422 | 16 | 33 |

mAP50-95 by labeled box size. A box is small when its area is under 32x32 pixels and large when it is over 96x96, measured on the original image.

| Small | Medium | Large |
|---|---|---|
| 0.295 | 0.648 | 0.77 |

Confusion matrix. Rows are labeled boxes and columns are predictions. The background row holds predictions that matched no labeled box, and the background column holds labeled boxes the model missed.

| | fuel | background |
|---|---|---|
| fuel | 422 | 33 |
| background | 16 | 0 |

Precision and recall for fuel at each confidence threshold:

| Confidence | Precision | Recall |
|---|---|---|
| 0.05 | 0.585 | 0.949 |
| 0.1 | 0.784 | 0.949 |
| 0.15 | 0.867 | 0.947 |
| 0.2 | 0.915 | 0.945 |
| 0.25 | 0.931 | 0.945 |
| 0.3 | 0.939 | 0.943 |
| 0.35 | 0.945 | 0.943 |
| 0.4 | 0.953 | 0.936 |
| 0.45 | 0.957 | 0.932 |
| 0.5 | 0.963 | 0.927 |
| 0.55 | 0.967 | 0.914 |
| 0.6 | 0.979 | 0.901 |
| 0.65 | 0.981 | 0.895 |
| 0.7 | 0.988 | 0.879 |
| 0.75 | 0.992 | 0.837 |
| 0.8 | 0.994 | 0.749 |
| 0.85 | 0.996 | 0.488 |
| 0.9 | 1.0 | 0.088 |
| 0.95 | n/a | 0.0 |

Recordings or groups with the most labeled boxes:

| Recording or group | Images | fuel boxes |
|---|---|---|
| `Basler_daA1280-54uc__24770352__20260112_181311364` | 69 | 335 |
| `Basler_daA1280-54uc__24770352__20260112_180938780` | 22 | 39 |
| `Basler_daA1280-54uc__24770352__20260112_180745507` | 26 | 33 |
| `Basler_daA1280-54uc__24770352__20260112_180633579` | 21 | 32 |
| `Basler_daA1280-54uc__24770352__20260112_180305685` | 23 | 16 |
| `Basler_daA1280-54uc__24770352__20260112_180602871` | 10 | 0 |

### baseline-a on robotzftp2 test

Run `baseline-a__robotzftp2-test__20260927T215447Z`, scored on 286 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 7 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 376 | 0.994 | 0.985 to 1.0 | 0.864 | 0.966 | 0.973 | 0.94 to 1.0 | 366 | 13 | 10 |

mAP50-95 by labeled box size. A box is small when its area is under 32x32 pixels and large when it is over 96x96, measured on the original image.

| Small | Medium | Large |
|---|---|---|
| n/a | 0.795 | 0.897 |

Confusion matrix. Rows are labeled boxes and columns are predictions. The background row holds predictions that matched no labeled box, and the background column holds labeled boxes the model missed.

| | fuel | background |
|---|---|---|
| fuel | 366 | 10 |
| background | 13 | 0 |

Precision and recall for fuel at each confidence threshold:

| Confidence | Precision | Recall |
|---|---|---|
| 0.05 | 0.239 | 1.0 |
| 0.1 | 0.451 | 1.0 |
| 0.15 | 0.605 | 1.0 |
| 0.2 | 0.693 | 0.997 |
| 0.25 | 0.783 | 0.997 |
| 0.3 | 0.842 | 0.995 |
| 0.35 | 0.871 | 0.989 |
| 0.4 | 0.909 | 0.984 |
| 0.45 | 0.931 | 0.976 |
| 0.5 | 0.966 | 0.973 |
| 0.55 | 0.973 | 0.968 |
| 0.6 | 0.981 | 0.965 |
| 0.65 | 1.0 | 0.957 |
| 0.7 | 1.0 | 0.952 |
| 0.75 | 1.0 | 0.947 |
| 0.8 | 1.0 | 0.936 |
| 0.85 | 1.0 | 0.896 |
| 0.9 | 1.0 | 0.532 |
| 0.95 | 1.0 | 0.035 |

Recordings or groups with the most labeled boxes:

| Recording or group | Images | fuel boxes |
|---|---|---|
| `IMG_5833_MOV` | 79 | 129 |
| `IMG_5835_MOV` | 65 | 117 |
| `IMG_5839_MOV` | 44 | 36 |
| `IMG_5836_MOV` | 35 | 35 |
| `IMG_5838_MOV` | 29 | 25 |
| `IMG_5837_MOV` | 20 | 20 |
| `IMG_5834_MOV` | 14 | 14 |

### baseline-a on scorekeeper test

Run `baseline-a__scorekeeper-test__20260927T215915Z`, scored on 314 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 109 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 375 | 0.836 | 0.773 to 0.895 | 0.725 | 0.257 | 1.0 | 1.0 to 1.0 | 375 | 1083 | 0 |

The robot boxes in scorekeeper are not scored, since baseline-a was not trained on robot.

mAP50-95 by labeled box size. A box is small when its area is under 32x32 pixels and large when it is over 96x96, measured on the original image.

| Small | Medium | Large |
|---|---|---|
| 0.73 | 0.751 | n/a |

Confusion matrix. Rows are labeled boxes and columns are predictions. The background row holds predictions that matched no labeled box, and the background column holds labeled boxes the model missed.

| | fuel | background |
|---|---|---|
| fuel | 375 | 0 |
| background | 1083 | 0 |

Precision and recall for fuel at each confidence threshold:

| Confidence | Precision | Recall |
|---|---|---|
| 0.05 | 0.059 | 1.0 |
| 0.1 | 0.063 | 1.0 |
| 0.15 | 0.066 | 1.0 |
| 0.2 | 0.07 | 1.0 |
| 0.25 | 0.077 | 1.0 |
| 0.3 | 0.091 | 1.0 |
| 0.35 | 0.114 | 1.0 |
| 0.4 | 0.148 | 1.0 |
| 0.45 | 0.195 | 1.0 |
| 0.5 | 0.257 | 1.0 |
| 0.55 | 0.319 | 1.0 |
| 0.6 | 0.389 | 1.0 |
| 0.65 | 0.466 | 1.0 |
| 0.7 | 0.573 | 1.0 |
| 0.75 | 0.671 | 1.0 |
| 0.8 | 0.724 | 0.981 |
| 0.85 | 0.809 | 0.845 |
| 0.9 | 0.814 | 0.128 |
| 0.95 | n/a | 0.0 |

Recordings or groups with the most labeled boxes:

| Recording or group | Images | fuel boxes |
|---|---|---|
| `frame_0233_jpg.rf.938d117831ebb49b9ab34bcfbed3ec2a.jpg` | 6 | 78 |
| `frame_0458_jpg.rf.355f6f2192332ea6dc865a496062d5ea.jpg` | 4 | 50 |
| `frame_0183_jpg.rf.90656b7c2eeed698cc936b06545ad856.jpg` | 4 | 40 |
| `frame_0228_jpg.rf.6f4f73dec0d529dcec5b7db7f4915935.jpg` | 3 | 39 |
| `frame_0294_jpg.rf.97adfc4a534b7a91f4d3d40faaa82941.jpg` | 2 | 28 |
| `frame_0051_jpg.rf.13768b3a1df8e51d2f8f8a07a9ba5ca7.jpg` | 2 | 19 |
| `frame_0250_jpg.rf.57cf4d97294a93e290917337b047ce07.jpg` | 1 | 14 |
| `frame_0316_jpg.rf.8024aa0b0627ddc2854338d7db10edb7.jpg` | 1 | 14 |
| `frame_0371_jpg.rf.2e1beb63eb6df2a82462512864bbcb7a.jpg` | 1 | 14 |
| `frame_0384_jpg.rf.7bf14483bb17c298466b8f860c5370ad.jpg` | 1 | 13 |

The other 99 hold 289 images and 66 fuel boxes.
<!-- EVALUATION:END -->
