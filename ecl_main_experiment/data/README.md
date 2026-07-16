# Data Setup

Raw benchmark datasets are not committed to this repository because of file
size and reproducibility considerations.

## ECL

Primary dataset used in this project:

- Dataset: ECL / Electricity Consuming Load
- Expected local path: `data/ECL/electricity.csv`
- Frequency: hourly
- Variables: 321 client electricity consumption series

Download the ECL electricity benchmark dataset from the public long-term
forecasting benchmark used by Informer/iTransformer-style experiments:

- iTransformer official repository: https://github.com/thuml/iTransformer
- Google Drive dataset archive: https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link
- Baidu Cloud mirror: https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr

After downloading and extracting the archive, place the ECL CSV at:

```text
data/ECL/electricity.csv
```

The training and analysis scripts expect this exact path.

## Notes

- `data/ECL/electricity.csv` is intentionally ignored by Git.
- Generated metrics and figures are stored under `results/`.
- The repository tracks code, experiment outputs, and documentation, but not
  large raw benchmark files.
