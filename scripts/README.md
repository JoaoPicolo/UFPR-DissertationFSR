# Findings scripts

This directoty stores the implementation of the scripts used to explore the findings of this work.

All the necessary packages are installable by using ***pip*** over the *requirements.txt* file and by installing the [insight-face](https://github.com/littletomatodonkey/insight-face-paddle) library.

The description of each step is defined as follow:

- The *datasets* directory stores the code used to pre-process the original datasets by detecting faces, cropping and aligning each image. The outputs will be 512x512 images saved in a structured (all images to one individual are in a subdirectory with the name of the individual) or unstructured (all images saved into a single directory) way. This is done both, for the LR and HR images.

- The *methodoloy* directory stores the code used to ground the proposed approach by spliting the HR and LR sets into the open and closed sets protocols. Is also creates the setup files used by *insightface* to determine the galery set. This directory has a "execution.sh" file to serve as an example on how to execute the code.

- The *recognition* directory stores the code used for final recognition evaluation. This is done by saving the results of a SR set into a *.csv* file named "results.csv". This directory has a "execution.sh" file to serve as an example on how to execute the code.

The "pipeline.py" file stores the code used to run the entire pipeline proposed by creating 30 random samples for each dataset. The final "results.csv" file composed by the sum of all the recognition evaluation steps is analysed by the "students_evaluation.py" file.

Finally the "data.zip" file contains, for each dataset and reconition protocol, the images randomly generated in each sample of the described text in the thesis. It also stores for each one of the experiments the final "results.csv" file.