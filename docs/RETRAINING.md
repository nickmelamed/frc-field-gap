# Training on Roboflow

This page is the procedure for training a model on the Roboflow platform
from a harmonized dataset, and for recording what the run used. The baseline
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
plan to use. If there are not, pick a smaller architecture (see step 6) and
note the reason in `reports/models.yaml`.

### 2. Create the project

In the web app, create an object detection project in the workspace your API
key belongs to. Name it after `platform.project` in `configs/project.yaml`.
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
make verify-upload KEY=marswars VERSION=1
```

This downloads the version in COCO format to `data/platform/`, matches every
image to its harmonized file by the file name recorded at upload, and writes
`reports/platform_upload_marswars.json`. It fails if an image is missing,
appears twice, sits in a different split, or has a different number of
boxes. The report also holds the preprocessing and augmentation Roboflow
reports for the version. Do not train on a version that fails this check.
The command refuses to run from a working tree with uncommitted changes,
since the report records the commit it was made from.

### 6. Train

Start training from the version page. For the baseline, use RF-DETR Nano,
which is small enough for the Raspberry Pi in Task 12. RF-DETR Small, or a
small YOLO model if credits are short, are the fallbacks. Start from the
default pretrained (COCO) checkpoint. Leave any train-time augmentation the
platform offers switched off for the baseline.

Before you start, write down every setting the training page shows, such as
the architecture, checkpoint, input size, epochs, and any advanced options.
Screenshots are fine as notes, but do not commit them.

### 7. Record the run

When training finishes, fill in the model's entry in `reports/models.yaml`:
the model ID, the version, and every setting from step 6. Commit it together
with `reports/platform_upload_<key>.json`. The evaluation in Task 6 reads the
model ID from there.

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
- Uploads are not transactional. A retry can succeed on a second attempt,
  and the order images arrive in is not fixed.
- The trained weights stay on Roboflow's servers and are not committed. If
  the model is deleted there, the baseline can be retrained with these
  steps, but it will not be the same model.
