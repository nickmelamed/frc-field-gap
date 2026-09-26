---
name: platform-training
description: Prepare a Roboflow platform training run for Nick to execute, and record what it used. Use for Task 5 and any retraining.
---

Nick runs training on the Roboflow platform himself. Your job is to make the
run reproducible on paper.

1. Check that the harmonized dataset to upload exists and passes
   `sv.DetectionDataset.from_coco`, and list its class counts from
   `reports/class_coverage.json`.
2. Confirm the dataset contains nothing from `data/field_test/` (rule 2).
3. Write or update the exact upload and training steps in docs/RETRAINING.md.
   Include the project name, dataset version, architecture (for example
   RF-DETR nano or small), input size, preprocessing, and augmentation (none
   for the baseline). Tell Nick to check his training credits first.
4. After Nick reports the run, record the model ID, version, and every
   setting shown in the UI in `reports/models.yaml`.
5. State plainly in the docs what is not reproducible bit for bit.

Never read or print the API key. Anything that needs it runs through the
package's own config loading.
