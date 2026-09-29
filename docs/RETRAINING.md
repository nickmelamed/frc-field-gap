# Training on Roboflow

How to train a model on Roboflow from a harmonized dataset and record what
the run used. The baseline
(Task 5) is the first run. Later runs, such as the merged model in Task 10,
follow the same steps with a different dataset key.

Training runs on Roboflow's servers, not in this repo, so the commands here
only prepare the data, upload it, and check it. Settings you pick in the web
app are recorded by hand in `reports/models.yaml`, and the settings Roboflow
reports through its API are recorded by `frc-verify-upload`.

## Why the upload is checked

Every harmonized dataset has a train, valid, and test split chosen so that
no test image has a near duplicate in train (see D-013 in
`docs/DECISIONS.md`). Roboflow can change splits in three ways. The SDK
guesses an image's split from its path, and puts anything it cannot place in
train. The web app can rebalance splits when a version is generated. And an
image that fails to upload is only printed as an error, so the run goes on
without it. Any of these would mix test images into training or change what
the test score means. So the upload names each split explicitly, and the
generated version is downloaded again and compared image by image before any
training starts.

## Steps

### 1. Check your training credits

Open the workspace's billing or usage page in the Roboflow web app and check
that there are enough training credits for one run of the architecture you
plan to use. If there are not, use the fallback for short credits in step 6
and note the reason in `reports/models.yaml`.

### 2. Create the project

In the web app, create an object detection project in the workspace your API
key belongs to. Name it after the dataset's entry under `platform.projects`
in `configs/project.yaml`.
Pick the CC BY 4.0 license, since the source datasets are CC BY 4.0. The
upload command refuses to run if the project does not exist, because the SDK
would otherwise create one under an MIT license.

### 3. Build and check the dataset

```bash
make harmonize
make upload KEY=marswars ARGS=--dry-run
```

The dry run loads every split with supervision, checks that each split holds
as many images as `reports/splits.json` records, and checks the images
against the field test set in `data/field_test/`. It stops with an error if
anything differs. When `data/field_test/` does not exist yet, it logs a
warning and has nothing to compare against.

### 4. Upload

```bash
make upload KEY=marswars
```

This runs the same checks and then uploads train, valid, and test in three
separate calls, each tagged with its split and a batch name like
`harmonized-marswars-train`. Images without boxes are uploaded as
background (null) images. The SDK prints one line per image, and a line
starting with `[ERR]` means that image did not arrive.

In the web app, open the project's dataset page and compare the number of
images in each split with `reports/splits.json`.

### 5. Generate a version

Roboflow never trains on the uploaded images directly. It trains on a
version, which is a frozen copy of the dataset with its splits,
preprocessing, and augmentation applied. Versions are numbered from 1 within
a project and never change once made, so a model can always be traced back
to exactly the images it saw. Editing or adding images later only affects
versions made after that.

In the web app, open the project and choose Versions in the sidebar (or
Generate, depending on the layout), then create a new version. The page walks
through a few steps. Use these settings.

- Source images. Include every image. There should be no unannotated
  images, since background images were uploaded as null.
- Train/test split. Leave it as uploaded. Do not rebalance.
- Preprocessing. Keep auto-orient. Add a resize only if the training page
  for your architecture asks for one, and then use the size it names with
  "Stretch to". For RF-DETR Nano the page asks for 384x384 (D-016). If you
  only find out on the training page, generate a second version with the
  resize and train on that one. Add nothing else. In particular, do not add
  "Filter Null", which would drop the background images.
- Augmentation. None for the baseline.

Generating takes a minute or two. The version number shown on its page is the
`VERSION` below. Then download the version and check it:

```bash
make verify-upload KEY=marswars VERSION=<n>
```

This downloads the version in COCO format to `data/platform/`, matches every
image to its harmonized file by file name, ignoring the `.rf.<hash>` suffix
and `_jpg` tag that Roboflow drops, and writes
`reports/platform_upload_<key>_v<version>.json`. The baseline's report,
`reports/platform_upload_marswars.json`, was written before the version
became part of the name. It fails if an image is missing,
appears twice, sits in a different split, or has a different number of
boxes of any class. For a version with augmentation, add
`ARGS=--augmented`, which allows extra copies of train images as long as
they stay in train, and skips their box counts, since a crop can change
them. Valid and test are still checked in full. The report also holds the
preprocessing and augmentation Roboflow reports for the version. Do not
train on a version that fails this check. The command refuses to run from a working tree with uncommitted changes,
since the report records the commit it was made from.

### 6. Train

Start training from the version page. For the baseline, use RF-DETR Nano,
which is small enough for the Raspberry Pi in Task 12. If Nano is not
offered, use RF-DETR Small. If credits are short, use a small YOLO model.
Start from the
default pretrained (COCO) checkpoint. Leave any train-time augmentation the
platform offers switched off for the baseline.

Before you start, write down every setting the training page shows, such as
the architecture, checkpoint, input size, epochs, and any advanced options.
Screenshots are fine as notes, but do not commit them.

### 7. Record the run

When training finishes, fill in the model's entry in `reports/models.yaml`:
the model ID, the version, and every setting from step 6. If the workspace
part of the model URL is made from an email address, leave it out, since this
repo is public. Commit it together
with `reports/platform_upload_<key>_v<version>.json`. The evaluation in
Task 6 reads the model ID from there.

## The merged model (Task 10)

The merged model trains on `data/harmonized/merged/` (D-028) in its own
project, `frc-rebuilt-merged`, with the same architecture and input size as
the baseline, so that the data is the main thing that changes. There are
two runs from one upload (D-029). `merged-noaug` uses no augmentation, like
the baseline, and shows what the merged data alone does. `merged-aug` adds
augmentation aimed at the diagnosed failures. If credits allow only one
run, train `merged-aug` and note in `reports/models.yaml` that the data and
the augmentation cannot be told apart.

1. Check your credits. The merged train split is about seven times the
   size of the baseline's, and the training page shows an estimate before
   the run starts. Write it down.
2. Create `frc-rebuilt-merged` in the web app with the `CC BY 4.0` license.
3. Build, check, and upload:

   ```bash
   make harmonize merge
   make upload KEY=merged ARGS=--dry-run
   make upload KEY=merged
   ```

   The dry run compares the split sizes with `reports/merge.json`.
4. Generate version 1 with auto-orient and Resize 384x384 (Stretch to), no
   augmentation, and splits as uploaded. Check it and commit the report:

   ```bash
   make verify-upload KEY=merged VERSION=1
   ```

5. Generate version 2 from the same images, with the same preprocessing and
   these augmentations, each applied to train only:
   - Flip, horizontal only. Fuel and robots have no left or right.
   - Crop, 0 to 20 percent. It cuts balls at the frame edge, which is where
     A's and B's misses were.
   - Brightness, -25 to +25 percent. B's misses fell in its darkest images.
   - Leave out hue and saturation, since the color of fuel is part of what
     tells it apart from other balls, and leave out rotation, which
     loosens boxes (D-014).

   Take the output count the page offers (usually three per training
   image) and write it down. Check it and commit the report:

   ```bash
   make verify-upload KEY=merged VERSION=2 ARGS=--augmented
   ```

6. Train RF-DETR Nano from the default COCO checkpoint on each version, with
   the same settings as the baseline and no train-time augmentation beyond
   the version's own. Record both runs in `reports/models.yaml` as
   `merged-noaug` and `merged-aug`, with `dataset: merged`.

## What is not reproducible bit for bit

Rerunning these steps gives the same harmonized data and the same split of
every image, which `frc-verify-upload` checks. The trained model will still
differ from run to run, for these reasons.

- The training code, its random seed, and its default hyperparameters belong
  to Roboflow and can change without notice. We record the settings the web
  app shows, but not the code behind them.
- Roboflow re-encodes images when they are uploaded and when a version is
  generated, so the pixels the model trains on differ slightly from the
  harmonized files. That is why the check matches images by name rather than
  by hash.
- Uploads can partly fail and be retried, and images arrive in no fixed
  order.
- The trained weights stay on Roboflow's servers and are not committed. If
  the model is deleted there, the baseline can be retrained with these
  steps, but it will not be the same model.
