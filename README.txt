Requirements:

Python 3.6
tensorflow 1.12
numpy
pandas
sklearn
cv2
tqdm
ggplot (NOTE: there is a bug in the current release, see https://stackoverflow.com/q/50591982/4138037 for fix)

Models variables are saved in "saves_*MODEL*" directories.

To get data to load correctly, set the global_address variable in load_data.py to point to a directory that contains the
attributes list, a directory of the provided labeled dataset called "dataset", and a directory of evaluation images called "testing_dataset".

All code is run from main.py. Uncomment the appropriate line to run code. Be cautious of trying to restore a model if save files are not present.