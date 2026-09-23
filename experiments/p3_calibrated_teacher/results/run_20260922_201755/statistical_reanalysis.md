# P3 main-split statistical reanalysis

No adaptation was rerun; this analysis reads the immutable main-run records.

- teacher accuracy delta: `+0.34790210`, 95% CI `[0.3304195804195804, 0.3660839160839161]`
- signed task delta: `+0.13840151`, 95% CI `[0.1296941092469684, 0.14698847064084106]`
- delta EER: `+0.00000000`, 95% CI `[-0.0017769739148454375, 0.0019133491057058156]`
- delta AUC: `+0.00018473`, 95% CI `[-0.0003985714381735506, 0.000897394051935285]`

- P3_TEACHER_MECHANISM_SUPPORTED: `True`
- P3_ADAPTATION_DIRECTION_SUPPORTED: `True`
- P3_TASK_GAIN_SUPPORTED: `False`
- P3_MECHANISM_PASS_LEGACY: `True`

Legacy PASS was over-broad because any favorable EER/AUC point estimate was sufficient; the hardened task-gain conclusion requires a paired-bootstrap CI to exclude zero.
