# This is the resource to execute the gap filling experiments for the tempeture chain


## Note

For the correct execution 

1) change the paths in folder_gap_filling.yaml

as well as 

2) change yaml location in the execution script here:

with open(r"/home/martinw/gapfill/notebooks/folder_gap_filling_giub.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)


--> All the data you need are in the folder data_input_for_gapfilling