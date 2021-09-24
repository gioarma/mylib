# mylib

## Installing the library

1. From the Github page click the `Code` button and copy the URL. 
2. Go to your GitHub Desktop app, click `File`->`Clone Repository`. 
3. Paste the URL
4. Select a folder where to save locally the library.

**Warning:** for importing the library inside your python code with `import mylib`, the local path you choose now must be inside the PYTHON PATH, i.e. listed in the sys.path list.
To check which folders are contained in sys.path, open a python console and type:
```python
import sys
print(sys.path)
```

5. Click `Clone`


If the folder you chose is not listed here, please add it to the python path.
On Linux/Mac using `conda` you should just add the following line to the `~/.bashrc` or `~/.zshrc` file (inserting the correct folder path):
```python
export PYTHONPATH="/Path/to/folder"
```
