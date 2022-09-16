# PathHom
Python code of path homology and persistent path homology.

---

## Data

The source data used in this work is provided.

https://github.com/WeilabMSU/PathHom/tree/main/data


---

## Source code

https://github.com/WeilabMSU/PathHom/tree/main/code


### Requirments

Python Dependencies
  - python      (>=3.7)
  - numpy       (1.21.5)

### Tutorial

1. Basic examples, from **nodes** and **edges**.

```
import numpy as np
from pathhomology import PathHomology

nodes = np.array([1, 2, 3, 4, 5, 6])
edges = np.array(
    [
        [1, 2],
        [3, 4],
        [3, 2],
        [5, 2],
        [5, 1],
    ]
)
max_path = 2
betti_num = PathHomology().path_homology(edges, nodes, max_path)
print(f'Betti numbers for 0 to max {max_path} path: {betti_num}')
```

2. Persistent path homology

Using the command line.
```
# dis-based
python "./code/pathhomology.py" --input_type cloudpoints --input_data "./data/B12N12_cage.csv" --filtration_type distance --max_distance 5 --save_name 'betti_numbers_dppt.npy' --max_path 2
# angle-based
python "./code/pathhomology.py" --input_type cloudpoints --input_data "./data/B12N12_cage.csv" --filtration_type angle --save_name 'betti_numbers_dppt.npy' --max_path 2
```

- `--input_type`: ['cloudpoints', 'digraph', 'No']
- `--input_data`: If the input type is cloudpoints, the input data should be the csv file, which contians the cloudpoints and weights with the shape n*m, the n means the number of the points, (m-1) is the dimension of the points, the last column are treated as weights. For the digraph, the format of the file is .csv. The contents of the file is cloudpoints and edges. The last two columns are start point idx and end point idx of the edges. All indices are start from 0.'
- `--filtration_type`: ['angle', 'distance']. For angle-based filtration, only `m=6, k=12` are used in current version.
- `--max_distance`: Cutoff of the max distance, if `filtration_type` is angle, it will be ignored.
- `--save_name`: The savename of the results. To save in `.npy` format.
- `--max_path`: Maximum length of the path. 

Using python script.

```

# From cloudpoints
import numpy as np
from pathhomology import PathHomology
max_path = 2
cloudpoints = np.array([
    [-1.1, 0, 0],
    [1.2, 0, 0],
    [0, -1.3, 0],
    [0, 1.4, 0],
    [1.5, 0, 0],
    [-1.6, 0, 0]
])
points_weight = [1, 2, 9, 4, 5, 6]
betti_nums_dppt = PathHomology().persistent_path_homology(
        cloudpoints, points_weight, max_path, filtration=None)
betti_nums_appt = PathHomology().persistent_angle_path_homology(
        cloudpoints, points_weight, max_path)

# #############

# From digraph (predefined digraph)
all_edges = np.array([
    [1, 3, 4, 4, 5, 5, 3, 5, 6, 2, 3, 4],
    [2, 2, 3, 1, 1, 2, 5, 4, 1, 6, 6, 6],
]).T - 1
betti_num_appt_digraph = PathHomology().persistent_angle_path_homology_from_digraph(
    cloudpoints, all_edges, max_path, filtration_angle_step=30)

filtration_for_dis_filtration = np.arange(0, 5, 0.2)
betti_num_dppt_digraph = PathHomology().persistent_path_homology_from_digraph(
        cloudpoints, all_edges, max_path, filtration=filtration_for_dis_filtration)

```

### Citing

- Persistent path topology in molecular and materials sciences.

### Contributors
`Pathhomology` code was created by [Dong Chen](https://github.com/ChenDdon) and is maintained by [Dong Chen](https://github.com/ChenDdon), Jian Liu and [Weilab](https://github.com/msuweilab) at MSU Math.

### License
All codes released in this study is under the MIT License.

---


