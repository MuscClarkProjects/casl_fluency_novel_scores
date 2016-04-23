import json
import pickle

def convert_ppmi(ppmi_src, ppmi_dest=None):
    with open(ppmi_src) as f:
        print("loading pickle")
        data = pickle.load(f)
        print("pickle loaded")

    if ppmi_dest is not None:
        with open(ppmi_dest, 'w') as f:
            json.dump(data, f)

    return data
