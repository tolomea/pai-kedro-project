# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited (“QuantumBlack”) name and logo
# (either separately or in combination, “QuantumBlack Trademarks”) are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

"""
# pylint: disable=invalid-name

from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pai
from pai_lmpt.detect import ContinuousTest


def generate_plot():
    """
    util to generate plot
    """

    def f(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)

    t1 = np.arange(0.0 * random.random(), 5.0 * random.random(), 0.1 * random.random())
    f = plt.plot(t1, f(t1), "bo")
    return f


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Node for splitting the classical Iris data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """
    test_data_ratio = parameters["example_test_data_ratio"]
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    data["target"] = pd.factorize(data["target"])[0]

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
    train_data_y = training_data["target"]
    test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
    test_data_y = test_data["target"]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )


def train_model(
    train_data_x: pd.DataFrame, train_data_y: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """

    # PAI experiment tracking begins here:

    local_path = Path(parameters["MY_LOCAL_PAI_DIR"]).absolute()

    pai.set_config(experiment=parameters["PAI_EXPERIMENT"], local_path=str(local_path))

    run_name = "Model Run at %s" % datetime.now().strftime("%H:%M:%S")
    # run_name = None
    pai.start_run(run_name=run_name)

    random_state = parameters["random_state"]
    n_estimators = parameters["n_estimators"]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    clf.fit(train_data_x, train_data_y)

    # 1. Use the default logger to save the model object, and model parameters
    pai.log(clf)

    # 2. Save other parameters which may be relevant
    pai.log_params({"master_table_ver": "abc", "seed": 0})

    # 3. Save features
    pai.log_features(list(train_data_x.columns), clf.feature_importances_)

    current_run_uuid = pai.current_run_uuid()

    pai.end_run()

    return dict(fitted_model=clf, run_id=current_run_uuid)


def predict(
    fitted_model, test_data_x: pd.DataFrame, test_data_y: pd.DataFrame, run_id: str
) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    preds = fitted_model.predict(test_data_x)
    conf_mat = pd.crosstab(
        test_data_y, preds, rownames=["Actual Species"], colnames=["Predicted Species"]
    )

    pai.start_run(run_id=run_id)
    # 5. Save performance metric
    pai.log_metrics({"accuracy": accuracy_score(test_data_y, preds)})
    pai.log_metrics({"f1": f1_score(test_data_y, preds, average="macro")})
    pai.log_metrics({"precision": precision_score(test_data_y, preds, average="macro")})
    pai.log_metrics({"recall": recall_score(test_data_y, preds, average="macro")})

    # 6. Save other artifacts, e.g. the confusion matrix
    pai.log_artifacts({"confusion_matrix": conf_mat})
    pai.log_artifacts({"plot": generate_plot()})

    pai.end_run()

    return preds


def data_shift_eval(
    eval_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    run_id: str,
    parameters: Dict[str, Any],
):
    """Node for doing covariate shift analysis on the new data
    """

    tests = parameters["covariate_shift_tests"]
    features = ["sepal_length", "petal_length"]

    with pai.start_run(run_id=run_id):
        # run hypothesis tests on each feature
        for f in features:
            result = (
                ContinuousTest(calls=tests)
                .run(evaluation=eval_df[f].values, reference=reference_df[f].values)
                .model_stats
            )

            pai.log_metrics(
                {
                    "%s_ks_test" % f: result["ks_test"][0],
                    "%s_levene_test" % f: result["levene_test"][0],
                }
            )
