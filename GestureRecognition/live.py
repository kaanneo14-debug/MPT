from labeling import data_labeling,dataset_building
from hmmclassifier import HMMClassifier

if __name__ == "__main__":
    data_labeling(10, "hut"),
    dataset_building(),
    HMMClassifier().fit()