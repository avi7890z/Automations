import os , hashlib
from pathlib import Path

path = input("Please enter the path to find duplicates : ")

file_list = os.walk(path)

unique = dict()
for root,folders,files in file_list:
    for file in files:
        path = Path(os.path.join(root,file))
        fileHash = hashlib.md5(open(path,'rb').read()).hexdigest()
        if fileHash not in unique:
            unique[fileHash] = path
            
        else:
            os.remove(path)
            print(f"{path} has been deleted")