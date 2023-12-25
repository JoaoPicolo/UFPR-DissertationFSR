import math

import pandas as pd
import numpy as np
from scipy.stats import t

def paired_t_test(sample1, sample2, alpha=0.05):
    # Calculate differences between paired observations
    differences = [x - y for x, y in zip(sample2, sample1)]

    # Calculate mean and standard deviation of the differences
    mean_diff = sum(differences) / len(differences)
    std_dev_diff = math.sqrt(sum((x - mean_diff) ** 2 for x in differences) / (len(differences) - 1))

    # Calculate t-statistic
    t_statistic = mean_diff / (std_dev_diff / math.sqrt(len(differences)))

    # Degrees of freedom
    degrees_of_freedom = len(differences) - 1

    # Calculate critical t-value for one-tailed test
    critical_t_value = abs(t.ppf(alpha, degrees_of_freedom))
    return t_statistic > critical_t_value

def main():
    dataframe = pd.read_csv("./results.csv", names=["method", "accuracy"])

    types = []
    sizes = []
    datasets = []
    methods = []
    for _, row in dataframe.iterrows():
        if "32" in row["method"]:
            sizes.append("256")
        elif "64" in row["method"]:
            sizes.append("512")
        elif "512" in row["method"]:
            sizes.append("512")

        if "open-set" in row["method"]:
            types.append("open-set")
        elif "closed-set" in row["method"]:
            types.append("closed-set")

        infos = row["method"].split('/')
        datasets.append(infos[0].split('-')[0])
        method_infos = infos[1].split('-')

        method = f"{method_infos[0]}_{method_infos[1]}"
        method = method.removesuffix("_quiscampi").removesuffix("_scface")
        methods.append(method)

    dataframe["size"] = sizes
    dataframe["type"] = types
    dataframe["dataset"] = datasets
    dataframe["method"] = methods

    base_method = "downgrade_64"
    methods = dataframe["method"].unique()
    protocols = dataframe["type"].unique()
    sizes = dataframe["size"].unique()
    datasets = dataframe["dataset"].unique()

    accept_null = 0
    reject_null = 0

    for protocol in protocols:
        for size in sizes:
            for dataset in datasets:
                for method in methods:
                    print(f"Protocol {protocol}, size {size}, dataset {dataset}, method {method}, ", end='')

                    before_accuracies = dataframe[(
                                        (dataframe["dataset"] == dataset)
                                        & (dataframe["type"] == protocol)
                                        & (dataframe["size"] == size)
                                        & (dataframe["method"] == base_method)
                                    )]["accuracy"].values

                    after_accuracies = dataframe[(
                                        (dataframe["dataset"] == dataset)
                                        & (dataframe["type"] == protocol)
                                        & (dataframe["size"] == size)
                                        & (dataframe["method"] == method)
                                    )]["accuracy"].values
                    
                    if len(before_accuracies) and len(after_accuracies):
                        res = paired_t_test(before_accuracies, after_accuracies)
                        if res:
                            print(f"mean: {np.mean(after_accuracies)}, std: {np.std(after_accuracies)}, Alternative hypothesis")
                            reject_null += 1
                        else:
                            print(f"mean: {np.mean(after_accuracies)}, std: {np.std(after_accuracies)}, Null hypothesis")
                            accept_null += 1
                    else:
                        print("None")

    print(reject_null, accept_null)

if __name__ == "__main__":
    main()