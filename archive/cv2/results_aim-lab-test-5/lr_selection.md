# lr selection (seed 0)

teacher: best-epoch val acc, lr ∈ {5e-5, 1e-4}.  student: CE final-epoch val acc, lr ∈ {5e-5, 1e-4, 3e-4}; the chosen student lr is used for CE and KD.  ties → smaller lr.

| dataset | run | lr=5e-05 | lr=0.0001 | lr=0.0003 | chosen |
|---|---|---|---|---|---|
| cub | teacher | 86.33 | 85.83 | – | **5e-05** |
| cub | student | 76.17 | 79.00 | 78.83 | **0.0001** |
| waterbirds | teacher | 93.99 | 93.33 | – | **5e-05** |
| waterbirds | student | 89.49 | 88.74 | 88.57 | **5e-05** |
