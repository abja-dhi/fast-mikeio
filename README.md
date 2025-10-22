# Installation
1. Clone the repo
2. python setup.py sdist bdist_wheel
3. pip install .

# Notes
1. The package currently works only with DFSU3DSigmaLayered files

# Features
1. Export the mesh directly from 3D Dfsu file using the code below:
```python
import fastmikeio
dfsu = fastmikeio.read("3D dfsu.dfsu")
dfsu.geometry.to_mesh(output_fname)
```