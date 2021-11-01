import argparse
import os
import random
import shutil 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_in",
        type=str,
    )
    parser.add_argument(
        "--path_out",
        type=str,
    )

    args = parser.parse_args()
    path_in = args.path_in
    folders = os.listdir(path_in)
    random.shuffle(folders)

    path_out = args.path_out
    os.mkdir(path_out)
    for folder in range(len(folders)//2): 
        dir_to_move = os.path.join(path_in, folders[folder])
        shutil.move(dir_to_move, path_out) 

    testfolder = os.listdir(path_in)
    valfolder = os.listdir(path_out)
    i, j = 0, 0
    for test in testfolder:
        os.rename(os.path.join(path_in, test), os.path.join(path_in, str(i)+'a'))
        i += 1
    for val in valfolder:
        os.rename(os.path.join(path_out, val), os.path.join(path_out, str(j)+'a'))
        j += 1
    
    testfolder = os.listdir(path_in)
    valfolder = os.listdir(path_out)
    i, j = 0, 0
    for test in testfolder:
        os.rename(os.path.join(path_in, test), os.path.join(path_in, str(i)))
        i += 1
    for val in valfolder:
        os.rename(os.path.join(path_out, val), os.path.join(path_out, str(j)))
        j += 1
if __name__ == '__main__':
    main()
# python test.py --path_in='femnist/test' --path_out='femnist/val'
# python test.py --path_in='sent140/test' --path_out='sent140/val'
# python test.py --path_in='shakespeare/test' --path_out='shakespeare/val'
