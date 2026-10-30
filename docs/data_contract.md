# Data Contract: Sensing Pipeline -> Local Training

What the training code expects on each Pi. Whatever produces the data (the stand-in script `scripts/prepare_pamap2.py` today, the cleaned PAMAP2 pipeline or the BLE receiver later) must write files in this format.

Working draft as of 10-24-27

## Files

One file per subject: `data/processed/subject_<id>.npz` (NumPy compressed archive). Subjects 101–108, one per Pi; a Pi only ever holds its own subject's file.

## Arrays

| Key | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float32 | [N, 19, 100] | Input windows: 19 channels × 100 steps (2 s at 50 Hz) |
| `y` | float32 | [N, 19, 10] | The next 10 steps after each window (the forecast target) |
| `activity` | int16 | [N] | PAMAP2 activity ID of the window |
| `split` | int8 | [N] | 0 = train, 1 = validation |
| `segment` | int32 | [N] | Clean segment the window came from |
| `start` | int32 | [N] | First step of the window inside its segment |
| `mean`, `std` | float32 | [19] | This subject's training-split channel statistics |
| `channels` | str | [19] | Channel names, in order |
| `subject`, `fs`, `input_len`, `horizon`, `stride`, `gap` | int32 | scalar | Metadata (fs = 50, input_len = 100, horizon = 10, stride = 50, gap = 110) |

`x` and `y` are already standardized: value_z = (value − mean) / std, using `mean` and `std` from this subject's training split only. To get physical units back: value = value_z × std + mean.

## Channels (in order)

1–3. hand accelerometer x, y, z (±16g sensor, m/s²)
4–6. hand gyroscope x, y, z (rad/s)
7–9. chest accelerometer x, y, z
10–12. chest gyroscope x, y, z
13–15. ankle accelerometer x, y, z
16–18. ankle gyroscope x, y, z
19. heart rate (bpm, linearly interpolated from about 9 Hz)

## How windows are made

- Protocol recordings only. Activity 0 (transient) is dropped.
- A bout is a run of one activity with no timestamp gap. IMU gaps up to 0.1 s are interpolated; longer gaps split the bout into separate segments.
- 100 Hz is downsampled to 50 Hz by averaging pairs of samples.
- Each segment is split in time: first ~80% for training, then 110 unused steps, then validation. Segments too short for both (under 330 steps, 6.6 s) go entirely to training.
- A sample is 110 consecutive steps (100 input + 10 target), taken every 50 steps.

## Training side guarantees

- It reads only `x`, `y` and `split` for training and evaluation; `activity` is used for reporting and the later classification task.
- Raw data never leaves the Pi. Only model weights and summary metrics are sent.
