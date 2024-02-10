# This is the resource to execute the gap filling experiments for the tempeture chain


## Note

For the correct execution change the paths in folder_gap_filling.yaml

as well as 

change yaml location in the execution script here:

with open(r"/home/martinw/gapfill/notebooks/folder_gap_filling_giub.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)
