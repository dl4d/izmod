import pickle

def iz_load(filename):
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)
