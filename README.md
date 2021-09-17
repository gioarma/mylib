# mylib

## Installing the library

1. From the Github page click the `Code` button and copy the URL. 
2. Go to your GitHub Desktop app, click `File`->`Clone Repository`. 
3. Paste the URL
4. Select a folder where to save locally the library.
5. Click `Clone`

**Warning:** for importing the library inside your python code with `import mylib`, the local path you choose now must be inside sys.path.
To check which folders are contained in sys.path, open a python console and type:
```python
import sys
print(sys.path)
```
This will list all the folders in your sys.path. You should save the Repository locally in one of these folders. If you save it to another folder, you should add this folder to the `sys.path` list by running in a python console:
```python
import sys
sys.path.insert(0,'/path/to/mod_directory')
```

