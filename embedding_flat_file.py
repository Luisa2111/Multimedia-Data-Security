import numpy as np

def embedding_flat(block, wat):
    if not hasattr(wat, "__len__"):
        if wat==1:
            if block[0,1]<=0:
                block[0,1]=1
            if block[1,1]<=0:
                block[1,1]=1
            if block[1,0]<=0:
                block[1,0]=1
        elif wat==-1:
            if block[0,1]>=0:
                block[0,1]=-1
            if block[1,1]>=0:
                block[1,1]=-1
            if block[1,0]>=0:
                block[1,0]=-1
        else:
            print("the wat value must be -1 or 1. Not", wat)
    else:
        print("wat must be an integer not a list")
