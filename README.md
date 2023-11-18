# INFOMMR2023
Multimedia retreival assignment instructions
## Installation

### Linux/MacOS
```sh
pip install -r requirements.txt
```
### Windows
```sh
python -m pip install -r requirements.txt
```

## Run

### Run Main Application
```sh
To run the main application, run 'src/main.py'.

The window will take 1-2 minutes to load since it first has to load all models, load examples of poorly-sampled shapes and prepare the ANN index.

By default, the average shape is rendered.
```

### Error Prevention
```sh
If you encounter a 'pyglet.window.NoSuchConfigException' Error, comment Line 22 in 'src/main.py'.
```

### Different Loading Options
```sh
By changing the '.head()' argument in Line 187 of 'src/scenes/query_scene.py' you can select how many poorly-sampled models will be loaded. We use 4 for loading speed purposes.

By changing the 'self.normalized_model_path' to 'self.less_class_model_path' in Lines 137 and 143 in 'src/render/query_scene.py' you can load the normalized shapes where some classes have either been combined or separated to less classes.
```

### Visualization Options
```sh
Click on the 'Show Wireframe' checkbox to render the shape in Wireframe mode.

Click on the 'Show Bounding Box' checkbox to render the bounding boxes for all rendered shapes.

Click on the 'Show Axes' checbkox to render the 3D axes on the world origin.
A 'Move Axes to Barycenter' option will then appear. Clicking on that checkbox will move the 3D axes to the shape barycenter instead.
```

### Visualize Poorly-Sampled Shape Examples
```sh
Clicking on the 'Show Poorly Sample Shapes' checkbox will show a porrly-sampled shape (right) example next to its resampled shape counterpart (left).
```

### Visualize Normalized Shapes
```sh
Clicking on the 'Show Normalized Shapes' checkbox will show the 1st shape of the 1st class of the normalized shape database.

2 Drop-down menus will then appear.
Clicking on the 1st one ('Classes') allows the user to select the class of which to render a shape. By default the 1st shape of that class will be rendered.
Clicking on the 2nd one ('Models') allows the user to select which shape to render from the selected class.
```

### Visualize Query Results
```sh
If the 'Show Normalized Shapes' checkbox is selected, the 'Query' menu will appear.

The user can select how many best-matching shapes ('n') to return. Be default, 1 shape is returned.

The user can also select what distance metric to use in the 'Custom' query, by clicking on the 'Select Distance Metric (Custom)' dropdown menu.

Clicking on the 'Get Best-Matching Shapes (Customs)' uses the 'Custom' query method with the selected distance metric.
Clicking on the 'Get Best-Matching Shapes (ANN)' uses the prepared 'ANN' query method with the 'braycurtis' distance.

The query shape always appears in the middle.
The 1st best-matching shape and its description appear on its left.
The 2nd best-matching shape and its description appear on its right.
The 3rd appears further to the left, the 4th further to the right, and so on.
```

### Visualize T-SNE
```sh
Clicking on the 'Run t-SNE' button will run the t-SNE algorithm on the normalized shape database.

After a few seconds, a new window will appear, displaying the scatterplot of the the resulting 2D shape features.

You can hover on any point to see its class and shape name.
```

### Evaluate Query
```sh
Clicking on the 'Evaluate CBSR System' checkbox allows the user to evaluate both query methods.

The user can once again select the number of best-matching shapes that will be returned, as well as the distance metric that will be used for the 'Custom' query.

Clicking on the 'Evaluate Custom Query' will evaluate the 'Custom' query. This can take up to 10 minutes depending on the selected distance metric.
Clicking on the 'Evaluate ANN Query' will evaluate the 'ANN' query. This will take a couple of seconds.

A new window which, by default, shows the average query precision, recall and f1-score will appear.
The use can use the 'Select Evaluation Subject' dropdown menu to visualize the query precision, recall and f1-score for a specific class.
```

### Generate Original Database CSV
```sh
To create a CSV file containing the statistics of the original database, run 'src/tools/save_statistics.py'.

This will generate the 'shape_data.csv' file already located inside src/tools/outputs.
```

### Generate Histograms
```sh
To generate the histograms with the average number of vertices or faces for the whole database as well as for each class separately, run 'src/tools/save_histograms.py'.

By default, the histograms are not displayed. To display them, change the 'show_histograms' argument to 'True' when calling 'save_histograms()' in Line 6 of 'src/tools/save_histograms.py'.

This will generate the histogram images already located inside 'src/tools/outputs/histograms'.
```

### Generate Normalized Database
```sh
To generate the normalized database CSV, run 'src/tools/database_write.py'

This will generate the 'database.CSV' already located inside 'src/tools/outputs'.

By changing the 'SAMPLE_SIZE' and 'BIN_SIZE' values in Lines 14 and 15 of 'src/tools/descriptor_extraction.py' you can generate different databases.
```
