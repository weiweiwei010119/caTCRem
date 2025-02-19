# caTCRem

## 🧬Project Overview

Monitoring the dynamic changes of T-cell receptor (TCR) immune repertoire in the peripheral blood that are elicited by tumorigenesis offers an attractive strategy for early cancer detection. However, current TCR-based cancer detection methods focus on limited high frequency TCRs instead of repertorie-scale representation, overlooking numerous crucial cancer-associated immune characteristics. Here, we proposed caTCRem, a repertorie-scale representation method of cancer associated TCRs, which enables precise cancer detection by amplifying the immune characteristic signals without discarding any TCR. The caTCRem exhibits remarkable precision in distinguishing cancer patients from non-cancer individuals using TCR repertories in peripheral blood. Furthermore, analysis of early-onset cancer progression and T cell function demonstrated the interpretability of caTCRem in capturing cancer-associated immune characteristics. Collectively, the work provides a repertorie-scale representation approach for non-invasive early cancer detection.

## 🚀Quick Experience

If you just want a quick experience without installing and configuring everything yourself, you can directly visit our [online site](http://www.jianglab.org.cn/caTCRem). There, you can experience the project's features and effects without downloading or installing any software.

## 📁Directory Structure

* **data/**: Contains datasets and sample data required for the project. Users can find example datasets for testing  in this directory.
* **model/**: Stores model files used by the project. Users can find pre-trained models for specific analysis tasks in this directory.
* **code/**: Contains the source code of the project. These codes include data preprocessing scripts, model training scripts, and more. Users can find all the source code implementing the project's functionality in this directory and modify or extend it as needed.

## 🔧Installation & Dependencies

To run this project, you need to ensure that the following Python libraries are installed. You can use `pip` to install these libraries.

1. Clone the project repository to your local environment.
2. In the project's root directory, run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

**Note**:

* For the installation of PaddlePaddle, you may need to choose the appropriate installation command based on your environment (CPU or GPU) and Python version. You can find detailed installation guides on the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick).
* If you encounter any installation issues, please check if your Python version and operating system support the library versions you want to install.

After installation, you should be able to run the code in the project. If you encounter any problems, please refer to the project's README file or contact the project maintainer for help.

👀 Additionally, if you want to execute the `01_process_reference_dataset.r` code, you need to ensure that **R** is installed on your system.

## 📚Usage Guide

### Data Preparation

* Place your dataset in the `data/` directory or specify the dataset location according to the paths in the code.
* Ensure that the dataset format is consistent with the format expected in the code.

### Running Analysis

* Find the corresponding script files in the `code/` directory and modify the parameters and configurations in the scripts as needed.
* Use the command line or an IDE to run the scripts to perform data preprocessing, model training, and result analysis tasks.
