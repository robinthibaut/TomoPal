# Mohinh package

Python toolbox in construction to assign values to a computational grid. I originaly developped this to build starting and reference models for Electrical Resistivity Tomography inversions.

### Instructions

Select points in the figure by enclosing them within a polygon.
The ending point as to be the starting point to enclose the shape.

Press the **esc** key to start a new polygon.
Hold the **shift** key to move all of the vertices.
Hold the **ctrl** key to move a single vertex.

You can use the tool ModelMaker from mohinh to interactively create prior models, and automatically save the results in a dat file if you provide a file name.
Otherwise, you can access the final results with (ModelMaker object).final_results and export it yourself.

Example with a backgroung value of 100:
```
mm = ModelMaker(blocks=blocks, values_log=1, bck=100)
```
```
my_model = rfwm.final_results
```

![Initialization](illu/blank.png)

![Assigning values to the grid](illu/illu.png)
