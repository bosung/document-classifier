import json
import numpy as np


def prepro(config):
    documents = []
    mal = []
    normal = []
    with open(config.data_input_file, "r", encoding='utf-8') as f:
        source = json.load(f)
        for data in source["data"]:
            documents.append(data)
            if data["type"] == "mal":
                mal.append(data)
            else:
                normal.append(data)

    print("total {} documents. {} mals / {} normals".format(
        len(documents), len(mal), len(normal)))

    # return documents
    return normalize(documents)


def normalize(data):
    feature_size = len(data[0]["features"])
    vectors = np.array([d["features"] for d in data])
    max_list = np.max(vectors, axis=0)
    for i in range(feature_size):
        if max_list[i] > 0:
            vectors[:, i] /= max_list[i]

    for i, d in enumerate(data):
        d["features"] = vectors[i]

    return data


def csv_parser(config):
    documents = []
    with open("pure_bit.csv", "r") as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            tokens = line.split(",")
            temp = dict()
            temp["type"] = "mal"
            temp["features"] = [float(f) for f in tokens[2:]]
            """
            temp["f1"] = tokens[2]
            temp["f2"] = tokens[3]
            temp["f3"] = tokens[4]
            temp["f4"] = tokens[5]
            temp["f5"] = tokens[6]
            temp["f6"] = tokens[7]
            temp["f7"] = tokens[8]
            temp["f8"] = tokens[9]
            temp["f9"] = tokens[10]
            temp["f10"] = tokens[11]
            temp["f11"] = tokens[12]
            temp["f12"] = tokens[13]
            temp["f13"] = tokens[14]
            temp["f14"] = tokens[15]
            temp["f15"] = tokens[16]
            temp["f16"] = tokens[17]
            temp["f17"] = tokens[18]
            temp["f18"] = tokens[19]
            """
            documents.append(temp)

    # return documents
    return normalize(documents)
