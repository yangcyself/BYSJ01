import os
import pickle as pkl
from contextlib import contextmanager
import traceback

class DataLogger:
    def __init__(self,dir = "./data",name = "tmp"):
        self.dirname = os.path.join(dir,name)
        os.makedirs(self.dirname,exist_ok=True)
        self.dumpFileCount = 0
        self.dumpLengh = 30
        self.dumpIterCount = self.dumpLengh
        self.toSave = []
    
    def dump(self):
        pkl.dump(self.toSave, open(os.path.join(self.dirname,"%04d.pkl"%self.dumpFileCount), "wb"))

    def checkDump(self):
        if(not self.dumpIterCount):
            self.dump()
            self.dumpIterCount = self.dumpLengh + 1
            self.dumpFileCount += 1
            self.toSave = []
        self.dumpIterCount -= 1

    def add(self,testPoint,fvalue):
        self.checkDump()
        self.toSave.append({
            "testPoint" : testPoint,
            "fvalue" : fvalue,
            "classifiedPoint" : []
        })

    def amend(self, classifiedPoint, side, clcu):
        self.toSave[-1]["classifiedPoint"].append({
            "classifiedPoint" : classifiedPoint,
            "side" : side,
            "clcu": clcu
        })

    def __enter__(self):
        return self

    def __exit__(self,*ex):
        self.dump()
        if (ex[0] is None):
            return 
        traceback.print_exception(*ex)
        



# @contextmanager
# def DataLogger(dir = "./data",name = "tmp"):
#     try:
#         logger = DataLogger_t(dir,name)
#         yield logger
#     finally:
#         logger.dump()
