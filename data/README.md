# data/

This folder holds the PAMAP2 dataset used to train and evaluate PulseNet. Only this README is tracked by git; everything else here is ignored on purpose (see `.gitignore`). Raw data never goes into the repo.

## Source

- **Dataset:** PAMAP2 Physical Activity Monitoring
- **Where:** UCI Machine Learning Repository, dataset 231 — https://archive.ics.uci.edu/dataset/231
- **File:** `PAMAP2_Dataset.zip` (656.3 MB)
- **License:** CC BY 4.0 (sharing and adaptation allowed with credit)

### Citation

Reiss, A. (2012). PAMAP2 Physical Activity Monitoring [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NW2H

A. Reiss and D. Stricker. "Introducing a New Benchmarked Dataset for Activity Monitoring." 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.

## Setup

1. Download `PAMAP2_Dataset.zip` from the UCI page above.
2. Unzip it into this `data/` folder.

The zip contains a `Protocol` folder (the 12-activity protocol recordings, one file per subject) and an `Optional` folder (extra activities some subjects performed).

## What's in the raw files

- One space-separated `.dat` file per subject per session; one line per timestamped sample.
- 54 columns: timestamp (s), activity ID, heart rate (bpm), then 17 columns each for the hand, chest and ankle IMUs.
- IMUs sample at 100 Hz; the heart-rate monitor at about 9 Hz.
- Missing values are `NaN`.
- Activity ID `0` marks transient periods between activities.

## How PulseNet uses it

- **Recordings:** protocol activities.
- **Subjects:** 101–108, one subject per Raspberry Pi.
- **Channels (19):** for each of the 3 IMUs, the ±16g accelerometer (x, y, z) and gyroscope (x, y, z), plus heart rate.
- **Sampling:** downsampled from 100 Hz to 50 Hz, with heart rate interpolated to 50 Hz.

## Getting data onto the Pis

The lab network has no internet, so download on a laptop, then copy each subject's data to its Pi with `scp`. Each Pi only receives its own subject's data.
