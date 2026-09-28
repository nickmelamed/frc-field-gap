# Model card: baseline-a

baseline-a is the project's first fuel detector. It was trained on one
team's footage so that it could be scored on other teams' footage, and it
is the model behind every result in the v0.1.0 release. Scores below are
copied from the tables that `make report` generates, and
`docs/EVALUATION.md` has the full results and the diagnosis.

## Model details

| Field | Value |
|---|---|
| Architecture | RF-DETR (Nano) |
| Trained on | Roboflow's hosted training, 2026-09-27 |
| Roboflow project and version | `frc-rebuilt-fuel-a`, version 2 |
| Model ID | `frc-rebuilt-fuel-a-2-rfdetr-nano-t1` |
| Input | 384x384, each image stretched to fit |
| Classes | fuel |
| Augmentation | none |
| Epochs | 100 at most, with early stopping |

Training settings are recorded in `reports/models.yaml`, and the upload and
training steps are in `docs/RETRAINING.md`. The model was trained with
Roboflow's platform, which doesn't expose every setting or a seed, so a
retrain on the same data will not give the same weights (see Limits).

The model was never published as weights. It is called through Roboflow's
hosted API by project and version, which works only from the workspace
that trained it. Its cached predictions are committed under
`reports/runs/`, so anyone can rescore them (see "Reproducing the
scores").

## Intended use

baseline-a exists to measure how a detector trained on one team's data
does on other teams' data, and to find out why it fails. It is meant for
research and teaching, such as an FRC student comparing their own
dataset with it.

It is not meant to drive a robot. It has never been scored on footage from
a real 2026 match, because the project's own field test set is not
confirmed yet, so nothing here shows how it does on a REBUILT field. It
detects fuel only, and draws many false boxes on people and on balls from
earlier games (see "Known failure modes").

## Training data

The model was trained on `marswars` (the "2026 Rebuilt" dataset from the
`marswars-robotics-program` workspace, version 5, `CC BY 4.0`), called
Dataset A in this project. It is the only dataset shot from a camera
mounted on a robot. Its fuel labels (`game_piece`) were mapped to fuel,
and its hub-state labels were dropped (D-012).

The Roboflow splits were replaced, since adjacent frames of one recording
sat on both sides. Each recording was cut by frame order into train, valid,
and test, with a gap of dropped frames between them (D-013). The training
split holds 942 images and 8653 fuel boxes, valid 140 images, and test 171
images from 6 recordings. The platform version used for training adds only
the 384x384 stretch (D-016).

## Evaluation

The model was scored on the test splits of three datasets, each from a
different team. Precision and recall count predictions with a confidence
of at least 0.5. The intervals come from resampling whole recordings or
groups of related photos, as `docs/EVALUATION.md` explains, along with
what mAP50, precision, and recall mean.

| Dataset | Images | mAP50 | mAP50 interval | Precision | Recall |
|---|---|---|---|---|---|
| `marswars` (A, its own dataset) | 171 | 0.936 | 0.923 to 1.0 | 0.963 | 0.927 |
| `robotzftp2` (B) | 286 | 0.994 | 0.985 to 1.0 | 0.966 | 0.973 |
| `scorekeeper` (C) | 314 | 0.836 | 0.773 to 0.895 | 0.257 | 1.0 |

A's score is an optimistic reference point, since its test split holds
later frames of the recordings the model trained on. B scores as well as
A because B's balls are large and few. On C the model finds every labeled
ball but draws 1083 boxes that match no label.

Roboflow's model page shows its own scores on A's test split. They use a
confidence threshold the platform picked on that same split, so they are
kept in `reports/models.yaml` for reference and are not quoted as results.

## Known failure modes

These come from the diagnosis in `docs/EVALUATION.md`, where errors were
sliced by source and judged by eye.

The model draws fuel on people. Of 40 sampled false positives on C, 30 are
people, such as heads in a broadcast crowd or spectators in yellow shirts.
Seven are yellow, orange, and blue balls from earlier FRC games. Almost
all of C's false positives fall on pit photos and match broadcasts with no
fuel in them.

It boxes other yellow objects. Of B's 13 false positives, 12 are yellow
things in the room, such as the cap of a vacuum cleaner.

It disagrees with the labels on balls cut off by the edge of the frame. Of
A's 33 misses, 24 are edge balls, and all 10 of B's misses are too. The
labels box the visible sliver, and the model boxes it differently or not at
all.

## Limits

- No REBUILT match footage has been scored.
- Each test split rests on a few recordings (6 for A and 7 for B), so
  scores move with which recordings landed in test. The intervals show how
  much.
- C's numbers are bounds. Each image keeps its 25 most confident
  predictions (D-020), which on C dropped only false positives, so C's
  true mAP50 is at most 0.836 and its false positives at least 1083.
- C is scored on fuel only, and its robot labels are ignored (D-019).
- The eye review is one person's first pass, and C's verdicts rest on 40
  of its 1083 false positives.
- Platform training can't be repeated bit for bit, and the hosted service
  reported a TensorRT fp16 backend, so scores can differ slightly from
  those in the Roboflow app.

## Reproducing the scores

`notebooks/reproduce_baseline.ipynb` downloads and harmonizes the datasets
on Colab and rescores the committed predictions with
`frc-evaluate --from-cache`, which needs a Roboflow API key for the
download only. `docs/EVALUATION.md` covers scoring a model live.

## License

The code is under the MIT License. The training and test data are Roboflow
Universe datasets under `CC BY 4.0`, credited with their exact versions in
the README.
