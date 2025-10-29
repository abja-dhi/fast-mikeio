# Installation
```cmd
pip install fastmikeio
```

# Notes
1. The package currently works with the following dfsu formats:
- Dfsu 2D Horizontal
- Dfsu 3D Sigmal Layer
- Dfsu Vertical Plane

# Features
1. Export the mesh directly from 3D Dfsu file using the code below:
```python
import fastmikeio
dfsu = fastmikeio.read("3D dfsu.dfsu")
dfsu.geometry.to_mesh(output_fname)
```
2. Extract vertical profile from 3D dfsu file
```python
x = [2000, 2100]
y = [100, 234]
output_filename = "Vertical Profile.dfsu"
vertical = dfsu.vertical_extractor(x, y, output_filename)
```
3. Retrive data as numpy array
```python
print(dfsu)     # Helper statement to get item indices
print(dfsu.n_layers)
print(dfsu.n_timesteps)
dfsu.get_data(item_idx=1, time_idx=[0, dfsu.n_timesteps-1], layer_idx=2)    # (1, 2, 1, n_2d_elements)
dfsu.get_data(item_idx=1)       # (1, n_timesteps, n_layers, n_2d_elements)
dfsu.get_data(reshape=False)        # (n_items, n_timesteps, n_3d_elements)
dfsu.get_data()                     # (n_items, n_timesteps, n_layers, n_2d_elements)
```