1. Preprocessing creates the folders in a structure which can be consumed by domain transfer algorithms and calssification algorithms.

2. To execute preprocessing  data_classify.py needs to be executed. Following is how to execute it : ``` python3 data_classify.py /data/capstone/BigEarthNet-v1.0 output```. The first parameter points to the source folder with all Big Earth images. The second parameter is the output folder where the data is generated.

3. The program executes based on the configuration file *dataselect_config.json* . Following is the structure of the file:

```
{
"selection_type":"all",
"label_array": ["Airports","Construction sites", "Port areas"],
"category" : "Forest",
"class_type":"category"
}
```
*selection_type* : The parameter can take following values . 1) *all* : The program runs for all data. 2) *label* : The program runs for the labels defined in the label_array 3) *category* : The program runs for the category defined in the json

*label_array* : The parameter defines the list of labels for which the program runs in case the *selection_type* is *label*.

*category* : The parameter defines the category for which the program runs in case the *selection_type* is *category*.

*calss_type* : The parameter defines how the calssids are defined. In case it is *category* then it selects the classid definition from *category_label_all.json* and *category_id_all.json*. In this case the classids are based on 15 subcategories. In case it is *label* then it selects the classid definition from *category_label.json* and *category_id.json*. In this case lassids are based on 43 labels. The category_label and category_id json files can be adjusted to change the subcategories.

4. The program generates the following folder structure:
```
+-- output
|   +-- CUT
|       +-- dataset
|           +-- trainA
|           +-- trainB
|       +-- predict
|           +-- dataset
|               +-- trainA
|               +-- trainB
|       +-- test
|   +-- alldata
|   +-- model
|       +-- test
|       +-- train
|       +-- validate
|   +-- flatselected.csv
|   +-- labelall.csv
|   +-- labelselected.csv
|   +-- labelsummary.csv
```
*CUT* : Folder for all CUT data
*CUT/dataset/*: All dataset for CUT training
*CUT/predict* : Dataset to generate fake data from the trained CUT model
*CUT/test*: Dataset to validate the output from CUT model
*alldata* : All images used for the processing
*model* : Dataset used in RESNET and VGG models
*model/train* : Train dataset for RESNET and VGG models
*model/test* : Test dataset for RESNET and VGG models
*model/validate* : Validate dataset for RESNET and VGG models
*flatselected.csv* : Flattened list of images with classids
*labelall.csv* : Label information for all images
*labelselected* : Label information for selected images
*labelsummary.csv* : Summary label information for selected images

*Execution time : 4600 secs for all images*
