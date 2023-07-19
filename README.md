# README.md

## Classification Task on High Dimensional Data
This project is a python-based solution to perform a classification task on high dimensional data. The main goal is to perform classification using the K-Nearest Neighbors algorithm. We also visualize the high dimensional data using t-SNE for dimensionality reduction, perform cross-validation to validate our model, and record the results using different metrics.

The tasks performed in this project are:

1. **Environment Setup:** We begin by importing necessary Python libraries like numpy, pandas, matplotlib, sklearn, pickle, etc.

2. **Load the Data:** The data is loaded from a pickle file using pickle.load().

3. **Data Exploration:** We visualize high dimensional data using t-SNE for dimensionality reduction and matplotlib for plotting.

4. **Prepare Data for Cross Validation:** We prepare our data for k-fold cross validation using sklearn's KFold class.

5. **Calculate Distances:** For each split in cross validation, we calculate the cosine and euclidean distances from each test set vector to the vectors in the training set (gallery vectors).

6. **K-Nearest Neighbors Classification:** We perform K-Nearest Neighbors (KNN) classification for both cosine and euclidean distances using sklearn's KNeighborsClassifier class.

7. **Record and Compare Results:** We record the top-k accuracy, AUC, and other relevant metrics for both algorithms (euclidean and cosine), create tables using pandas, and export the results.

8. **Plot ROC AUC Graph:** We create an ROC AUC graph for both algorithms to compare their performance visually.

9. **Write Unit Tests:** We also write a couple of unit tests for our code using the pytest testing framework.

The whole process involves exploration and experimentation, adjusting parameters as needed, and interpreting the results at each step to inform our choices in the next steps.

## Results

The following table shows the average results achieved by the classification model using both Cosine and Euclidean distances:

| Metric         | Cosine Distance | Euclidean Distance |
|----------------|-----------------|--------------------|
| AUC            | 0.8951          | 0.8696             |
| Accuracy       | 0.7168          | 0.6227             |
| F1-Score       | 0.7113          | 0.6157             |
| Precision      | 0.7279          | 0.6714             |
| Recall         | 0.7168          | 0.6227             |
| Top-3 Accuracy | 0.8871          | 0.8458             |

These results indicate that the model performs better with the Cosine distance metric, achieving higher accuracy, precision, recall, and AUC values than the Euclidean distance. 

## Running the project

This project can be run in two ways: using Docker Compose or by creating a local virtual environment. 
The orchestration of this project is done by the Makefile, you could run ``make help`` to check the available rules.

there you have:

```bash
clean               Delete all .png, .csv, .txt and cache files 
help                Shows this help text 
run_classifier      Plot a 2D data visualizatiom, calculate distances, perform cross validation,handle data 
                    preparation and classifie the data. 
test                Run unit tests. 
```

This project generates a significant amount of .csv, .txt and .png files which are used to generate the metrics of the tested models. A strong recommendation is to use make clean to clean the directory after using the used artifacts.

### Using Docker Compose:

1. Build and run the project using the following command: 
    ```
    docker-compose up
    ```
2. To run tests, use:
    ```
    docker-compose run app make test
    ```
3. To run classifier, use:
    ```
    docker-compose run app make run_classifier
    ```
4. To run all targets in the Makefile, use:
    ```
    docker-compose run app make all
    ```
### Using local virtual environment:

1. Create a virtual environment:
    ```
    python -m venv venv
    ```
2. Activate the virtual environment:
    - On Windows, use:
        ```
        .\venv\Scripts\activate
        ```
    - On Unix or MacOS, use:
        ```
        source venv/bin/activate
        ```
3. Install the requirements:
    ```
    pip install -r requirements.txt
    ```
4. Run the commands as in the Docker Compose section, but without the "docker-compose run app" part. For example, to run all targets, use:
    ```
    make all
    ```
Remember to navigate to the project directory before running these commands.
