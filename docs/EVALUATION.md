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

On B, the model does as well as on A. Fuel mAP50 is 0.994 (interval 0.985
to 1.0), with precision 0.966 and recall 0.973. Why B is no harder than A is
still open.

On C, fuel mAP50 drops to 0.836 (interval 0.773 to 0.895). That interval
does not overlap A's, so the drop is larger than what the choice of test
photos alone would explain. Every labeled fuel box is found (375 hits,
recall 1.0), so the whole drop comes from false positives. The model also
draws 1083 boxes that match no label, which puts precision at 0.257.
Raising the threshold helps only partway. At 0.8, precision is 0.724 and
recall 0.981.

C's numbers are bounds. Most of C's images reached the limit of 25 kept
predictions per image (D-020, and the note under C's table), and on a few
of them every kept box is at or above 0.5. Since recall stays at 1.0 even at the
lowest threshold, the limit dropped only false positives. The true mAP50 is
therefore at most 0.836, and the true false positive count at 0.5 is at
least 1083. The drop is at least as large as shown.

C's test split mixes two kinds of photos (see `docs/DATASETS.md`). One kind
shows fuel indoors. The other shows robots in pits and match broadcasts
from earlier games, one of which used balls, and none of those photos has a
fuel label. A first look at the saved predictions puts most of the false
positives on photos without fuel labels. If that holds up, the model is
mistaking other round objects for fuel. That is a real model error, since
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
  false positive. It uses each image's 25 most confident predictions down
  to a confidence of 0.01.
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
D-018. The run is scored from those rounded values, so
`--from-cache <run_id>` reproduces its scores exactly without calling the
model again, and can rescore them at another threshold. A rescore must name the same model, dataset, and split as the run
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

Run `baseline-a__marswars-test__20260927T221432Z`, scored on 171 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 6 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 455 | 0.936 | 0.923 to 1.0 | 0.623 | 0.963 | 0.927 | 0.908 to 1.0 | 422 | 16 | 33 |

35 of 171 images reached the limit of 25 predictions per image (D-020), so fainter boxes may have been dropped there.

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

Run `baseline-a__robotzftp2-test__20260927T221735Z`, scored on 286 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 7 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 376 | 0.994 | 0.985 to 1.0 | 0.864 | 0.966 | 0.973 | 0.94 to 1.0 | 366 | 13 | 10 |

122 of 286 images reached the limit of 25 predictions per image (D-020), so fainter boxes may have been dropped there.

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

Run `baseline-a__scorekeeper-test__20260927T222144Z`, scored on 314 images. Precision, recall, and the confusion matrix count predictions with confidence of at least 0.5.
Intervals come from 1000 resamples of the split's 109 recordings or groups, and hold the middle 95% of the resampled scores. <!-- numbers: ok -->

| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | Recall interval | Hits | False positives | Misses |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel | 375 | 0.836 | 0.773 to 0.895 | 0.725 | 0.257 | 1.0 | 1.0 to 1.0 | 375 | 1083 | 0 |

The robot boxes in scorekeeper are not scored, since baseline-a was never trained to find them.

286 of 314 images reached the limit of 25 predictions per image (D-020), so fainter boxes may have been dropped there. In 4 of them every kept box is at or above the threshold, so boxes that would have counted at it were dropped, and the false positive count at the threshold is a lower bound.

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

## Diagnosis

Everything between the `DIAGNOSIS` markers is written by `make report` from
`reports/diagnosis/`, which `make diagnose MODEL=<model>` makes from the
published runs' cached predictions. Do not edit it by hand.

<!-- DIAGNOSIS:START -->
### Where baseline-a's errors fall

Hits, false positives, and misses count predictions with confidence of at least 0.5, matched to labels as the confusion matrix matches them. Brightness is the mean gray level from 0 to 255 and sharpness the variance of the Laplacian, both measured on the image stretched to 384 pixels square, as the model sees it. Their bins hold equal numbers of images, pooled over every test split, so a bin can hold few images of one dataset.

Brightness:

| Dataset | Brightness | Images | Labeled boxes | Hits | False positives | Misses | Precision | Recall | mAP50 | False positives per image |
|---|---|---|---|---|---|---|---|---|---|---|
| marswars | below 107.8 | 81 | 142 | 130 | 4 | 12 | 0.97 | 0.915 | 0.938 | 0.049 |
| marswars | 107.8 to 115.6 | 42 | 252 | 233 | 10 | 19 | 0.959 | 0.925 | 0.936 | 0.238 |
| marswars | 115.6 and above | 48 | 61 | 59 | 2 | 2 | 0.967 | 0.967 | 0.968 | 0.042 |
| robotzftp2 | below 107.8 | 96 | 100 | 91 | 8 | 9 | 0.919 | 0.91 | 0.987 | 0.083 |
| robotzftp2 | 107.8 to 115.6 | 62 | 106 | 105 | 5 | 1 | 0.955 | 0.991 | 0.994 | 0.081 |
| robotzftp2 | 115.6 and above | 128 | 170 | 170 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| scorekeeper | below 107.8 | 81 | 113 | 113 | 241 | 0 | 0.319 | 1.0 | 0.785 | 2.975 |
| scorekeeper | 107.8 to 115.6 | 149 | 105 | 105 | 581 | 0 | 0.153 | 1.0 | 0.931 | 3.899 |
| scorekeeper | 115.6 and above | 84 | 157 | 157 | 261 | 0 | 0.376 | 1.0 | 0.857 | 3.107 |

Sharpness:

| Dataset | Sharpness | Images | Labeled boxes | Hits | False positives | Misses | Precision | Recall | mAP50 | False positives per image |
|---|---|---|---|---|---|---|---|---|---|---|
| marswars | below 170.9 | 58 | 296 | 271 | 10 | 25 | 0.964 | 0.916 | 0.936 | 0.172 |
| marswars | 170.9 to 695.3 | 113 | 159 | 151 | 6 | 8 | 0.962 | 0.95 | 0.953 | 0.053 |
| robotzftp2 | below 170.9 | 160 | 180 | 176 | 4 | 4 | 0.978 | 0.978 | 0.998 | 0.025 |
| robotzftp2 | 170.9 to 695.3 | 94 | 136 | 132 | 4 | 4 | 0.971 | 0.971 | 0.995 | 0.043 |
| robotzftp2 | 695.3 and above | 32 | 60 | 58 | 5 | 2 | 0.921 | 0.967 | 0.989 | 0.156 |
| scorekeeper | below 170.9 | 38 | 375 | 375 | 74 | 0 | 0.835 | 1.0 | 0.886 | 1.947 |
| scorekeeper | 170.9 to 695.3 | 51 | 0 | 0 | 70 | 0 | 0.0 | n/a | n/a | 1.373 |
| scorekeeper | 695.3 and above | 225 | 0 | 0 | 939 | 0 | 0.0 | n/a | n/a | 4.173 |

Labeled fuel per image:

| Dataset | Labeled fuel per image | Images | Labeled boxes | Hits | False positives | Misses | Precision | Recall | mAP50 | False positives per image |
|---|---|---|---|---|---|---|---|---|---|---|
| marswars | 0 | 35 | 0 | 0 | 1 | 0 | 0.0 | n/a | n/a | 0.029 |
| marswars | 1 | 37 | 37 | 36 | 1 | 1 | 0.973 | 0.973 | 0.999 | 0.027 |
| marswars | 2 to 4 | 75 | 151 | 150 | 1 | 1 | 0.993 | 0.993 | 0.986 | 0.013 |
| marswars | 5 to 9 | 7 | 51 | 39 | 4 | 12 | 0.907 | 0.765 | 0.828 | 0.571 |
| marswars | 10 or more | 17 | 216 | 197 | 9 | 19 | 0.956 | 0.912 | 0.926 | 0.529 |
| robotzftp2 | 0 | 17 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0.0 |
| robotzftp2 | 1 | 162 | 162 | 158 | 7 | 4 | 0.958 | 0.975 | 0.996 | 0.043 |
| robotzftp2 | 2 to 4 | 107 | 214 | 208 | 6 | 6 | 0.972 | 0.972 | 0.994 | 0.056 |
| scorekeeper | 0 | 283 | 0 | 0 | 1012 | 0 | 0.0 | n/a | n/a | 3.576 |
| scorekeeper | 5 to 9 | 2 | 18 | 18 | 8 | 0 | 0.692 | 1.0 | 0.875 | 4.0 |
| scorekeeper | 10 or more | 29 | 357 | 357 | 63 | 0 | 0.85 | 1.0 | 0.903 | 2.172 |

Has labeled fuel:

| Dataset | Has labeled fuel | Images | Labeled boxes | Hits | False positives | Misses | Precision | Recall | mAP50 | False positives per image |
|---|---|---|---|---|---|---|---|---|---|---|
| marswars | yes | 136 | 455 | 422 | 15 | 33 | 0.966 | 0.927 | 0.936 | 0.11 |
| marswars | no | 35 | 0 | 0 | 1 | 0 | 0.0 | n/a | n/a | 0.029 |
| robotzftp2 | yes | 269 | 376 | 366 | 13 | 10 | 0.966 | 0.973 | 0.994 | 0.048 |
| robotzftp2 | no | 17 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0.0 |
| scorekeeper | yes | 31 | 375 | 375 | 71 | 0 | 0.841 | 1.0 | 0.889 | 2.29 |
| scorekeeper | no | 283 | 0 | 0 | 1012 | 0 | 0.0 | n/a | n/a | 3.576 |

Source:

| Dataset | Source | Images | Labeled boxes | Hits | False positives | Misses | Precision | Recall | mAP50 | False positives per image |
|---|---|---|---|---|---|---|---|---|---|---|
| marswars | Basler_daA1280-54uc__24770352__20260112_180305685 | 23 | 16 | 15 | 1 | 1 | 0.938 | 0.938 | 0.996 | 0.043 |
| marswars | Basler_daA1280-54uc__24770352__20260112_180602871 | 10 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0.0 |
| marswars | Basler_daA1280-54uc__24770352__20260112_180633579 | 21 | 32 | 32 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| marswars | Basler_daA1280-54uc__24770352__20260112_180745507 | 26 | 33 | 33 | 1 | 0 | 0.971 | 1.0 | 0.999 | 0.038 |
| marswars | Basler_daA1280-54uc__24770352__20260112_180938780 | 22 | 39 | 39 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| marswars | Basler_daA1280-54uc__24770352__20260112_181311364 | 69 | 335 | 303 | 14 | 32 | 0.956 | 0.904 | 0.922 | 0.203 |
| robotzftp2 | IMG_5833_MOV | 79 | 129 | 129 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| robotzftp2 | IMG_5834_MOV | 14 | 14 | 14 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| robotzftp2 | IMG_5835_MOV | 65 | 117 | 111 | 8 | 6 | 0.933 | 0.949 | 0.985 | 0.123 |
| robotzftp2 | IMG_5836_MOV | 35 | 35 | 31 | 1 | 4 | 0.969 | 0.886 | 0.994 | 0.029 |
| robotzftp2 | IMG_5837_MOV | 20 | 20 | 20 | 2 | 0 | 0.909 | 1.0 | 1.0 | 0.1 |
| robotzftp2 | IMG_5838_MOV | 29 | 25 | 25 | 2 | 0 | 0.926 | 1.0 | 1.0 | 0.069 |
| robotzftp2 | IMG_5839_MOV | 44 | 36 | 36 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0.0 |
| scorekeeper | fuel photos | 31 | 375 | 375 | 71 | 0 | 0.841 | 1.0 | 0.889 | 2.29 |
| scorekeeper | 2024 Milford broadcast | 159 | 0 | 0 | 632 | 0 | 0.0 | n/a | n/a | 3.975 |
| scorekeeper | frc_train photos | 60 | 0 | 0 | 111 | 0 | 0.0 | n/a | n/a | 1.85 |
| scorekeeper | YouTube frames | 34 | 0 | 0 | 217 | 0 | 0.0 | n/a | n/a | 6.382 |
| scorekeeper | other | 30 | 0 | 0 | 52 | 0 | 0.0 | n/a | n/a | 1.733 |

By relative box size. A box is small when it covers less than 0.0025 of the image area, medium below 0.0225, and large otherwise. Labeled boxes are sized by their label and false positives by their own box.

| Dataset | Size | Labeled boxes | Hits | Recall | False positives |
|---|---|---|---|---|---|
| marswars | small | 222 | 192 | 0.865 | 12 |
| marswars | medium | 233 | 230 | 0.987 | 3 |
| marswars | large | 0 | 0 | n/a | 1 |
| robotzftp2 | small | 13 | 7 | 0.538 | 11 |
| robotzftp2 | medium | 180 | 177 | 0.983 | 1 |
| robotzftp2 | large | 183 | 182 | 0.995 | 1 |
| scorekeeper | small | 32 | 32 | 1.0 | 870 |
| scorekeeper | medium | 335 | 335 | 1.0 | 186 |
| scorekeeper | large | 8 | 8 | 1.0 | 27 |

False positives by why they matched no label. A duplicate overlaps a label that another prediction already took. A localization error overlaps a label by more than 0.1 IoU, but not enough to count. One inside an unscored label has its center inside a box of a class the model is not scored on, such as a robot. The rest are background.

| Dataset | Duplicate | Localization | Inside an unscored label | Background |
|---|---|---|---|---|
| marswars | 1 | 12 | 0 | 3 |
| robotzftp2 | 0 | 1 | 0 | 12 |
| scorekeeper | 2 | 58 | 53 | 970 |

The model's training split next to each test split. Each cell gives the quantiles 0.1, 0.5, 0.9. Box side is the side of a square with the box's share of the image area, as a fraction of the image side.

| Split | Role | Images | Labeled boxes | Brightness | Sharpness | Box side | Boxes per image |
|---|---|---|---|---|---|---|---|
| marswars train | training | 942 | 8653 | 67.432 / 116.428 / 138.227 | 114.865 / 167.537 / 460.009 | 0.02 / 0.044 / 0.087 | 1.0 / 2.0 / 17.0 |
| marswars test | test | 171 | 455 | 92.645 / 108.514 / 131.201 | 121.96 / 192.29 / 254.477 | 0.028 / 0.051 / 0.077 | 0.0 / 2.0 / 9.0 |
| robotzftp2 test | test | 286 | 376 | 76.049 / 111.808 / 153.63 | 27.24 / 139.397 / 724.864 | 0.06 / 0.148 / 0.238 | 1.0 / 1.0 / 2.0 |
| scorekeeper test | test | 314 | 375 | 93.041 / 114.773 / 126.0 | 97.835 / 1995.108 / 2400.531 | 0.05 / 0.061 / 0.116 | 0.0 / 0.0 / 0.0 |

Errors judged by eye. Where a split has more errors than were reviewed, the reviewed ones are a seeded random sample.

| Verdict | marswars false positives | marswars misses | robotzftp2 false positives | robotzftp2 misses | scorekeeper false positives |
|---|---|---|---|---|---|
| Errors | 16 | 33 | 13 | 10 | 1083 |
| Reviewed | 16 | 33 | 13 | 10 | 40 |
| unlabeled fuel | 1 | 0 | 0 | 0 | 3 |
| person | 1 | 0 | 0 | 0 | 30 |
| wrong box on labeled fuel | 5 | 3 | 1 | 0 | 0 |
| loose or offset label | 1 | 2 | 0 | 0 | 0 |
| ball cut off at the image edge | 8 | 24 | 0 | 10 | 0 |
| barely visible fuel | 0 | 4 | 0 | 0 | 0 |
| other yellow object | 0 | 0 | 12 | 0 | 0 |
| ball from another game | 0 | 0 | 0 | 0 | 7 |

The failure gallery, tile by tile from the top left:

| Tile | Dataset | Error | Source |
|---|---|---|---|
| 1 | marswars | false positive | `Basler_daA1280-54uc__24770352__20260112_181311364` |
| 2 | marswars | miss | `Basler_daA1280-54uc__24770352__20260112_181311364` |
| 3 | marswars | false positive | `Basler_daA1280-54uc__24770352__20260112_180745507` |
| 4 | robotzftp2 | false positive | `IMG_5835_MOV` |
| 5 | robotzftp2 | miss | `IMG_5836_MOV` |
| 6 | robotzftp2 | false positive | `IMG_5837_MOV` |
| 7 | robotzftp2 | false positive | `IMG_5838_MOV` |
| 8 | scorekeeper | false positive | `other` |
| 9 | scorekeeper | false positive | `fuel photos` |
| 10 | scorekeeper | false positive | `frc_train photos` |
| 11 | scorekeeper | false positive | `YouTube frames` |
| 12 | scorekeeper | false positive | `other` |
| 13 | scorekeeper | false positive | `fuel photos` |
| 14 | scorekeeper | false positive | `frc_train photos` |
| 15 | scorekeeper | false positive | `YouTube frames` |
<!-- DIAGNOSIS:END -->
