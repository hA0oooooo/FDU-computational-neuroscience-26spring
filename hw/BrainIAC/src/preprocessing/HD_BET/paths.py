import os

# please refer to the readme on where to get the parameters. Save them in this folder:
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.dirname(current_dir)
folder_with_parameter_files = os.path.join(preprocessing_dir, 'hd-bet_params')
