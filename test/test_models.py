import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from src.models import models, featureExtraction, eval, training, requiresPositionalArgument
import inspect

# Test class for model-related functions
class TestModel:
    """
    Tests for model-related functions.
    """

    def test_regression_models(self):
        """
        Test if there are regression models available.
        """
        assert len(models("Regression")) > 0

    def test_classification_models(self):
        """
        Test if there are classification models available.
        """
        assert len(models("Classification")) > 0

# Test class for feature extraction
class TestFeatureExtraction:
    """
    Tests for the feature extraction function.
    """

    def test_input(self):
        """
        Test if the feature extraction function accepts the correct number of inputs.
        """
        len_input = len(inspect.signature(featureExtraction).parameters)
        assert len_input == 3

    def test_input_type(self):
        """
        Test if the feature extraction function correctly raises TypeError for invalid input types.
        """
        with pytest.raises(TypeError):
            featureExtraction(1, 2, 3)
        with pytest.raises(TypeError):
            featureExtraction("a", "b", "c")
        with pytest.raises(TypeError):
            featureExtraction([1, 2], [3, 4], 2)

    def test_input_dimension(self):
        """
        Test if the feature extraction function raises ValueError for invalid input dimensions.
        """
        with pytest.raises(ValueError):
            featureExtraction(np.array([1, 2]), np.array([3, 4]), 2)
        with pytest.raises(ValueError):
            featureExtraction(np.array([[1, 2], [3, 4]]), np.array([3, 4]), 2)
        with pytest.raises(ValueError):
            featureExtraction(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [5, 6]]), 3)

    def test_valid_input(self):
        """
        Test if the feature extraction function works with valid input.
        """
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[3, 4], [5, 6]])
        slider = 2
        assert featureExtraction(X_train, X_test, slider)[0].shape[1] == slider
        assert featureExtraction(X_train, X_test, slider)[1].shape[1] == slider


    def test_invalid_3rd_dimension(self):
        """
        Test if the feature extraction function raises ValueError for invalid third dimension.
        """
        with pytest.raises(ValueError):
            featureExtraction(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [5, 6]]), 3)
        with pytest.raises(ValueError):
            featureExtraction(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [5, 6]]), 4)
        with pytest.raises(ValueError):
            featureExtraction(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [5, 6]]), 5)

# Test class for Evaluation-related functions
class TestEval:
    """
    Tests for Evaluation-related functions.
    """

    def setup_method(self):
        self.X_train = np.array([[1, 2], [3, 4]])
        self.y_train = np.array([5, 6])
        self.X_test = np.array([[5, 6], [7, 8]])
        self.y_test = np.array([9, 10])

    def test_eval_valid(self):
        """
        Test evaluation with a valid model instance and valid evaluation metrics.
        """
        model = LinearRegression()
        eval_metrics = ["Accuracy"]
        y_pred, evaluation_results = eval(self.X_train, self.y_train, self.X_test, self.y_test, model, eval_metrics)
        assert isinstance(y_pred, np.ndarray)
        assert "Accuracy" in evaluation_results

    def test_eval_invalid_model(self):
        """
        Test evaluation with an invalid model.
        """
        model = "invalid_model"
        eval_metrics = ["Accuracy"]
        with pytest.raises(ValueError):
            eval(self.X_train, self.y_train, self.X_test, self.y_test, model, eval_metrics)

    def test_eval_invalid_metric(self):
        """
        Test evaluation with an invalid evaluation metric.
        """
        model = LinearRegression()
        eval_metrics = ["InvalidMetric"]
        with pytest.raises(ValueError):
            eval(self.X_train, self.y_train, self.X_test, self.y_test, model, eval_metrics)

    def test_eval_no_model_instance(self):
        """
        Test evaluation without a model instance (using a class).
        """
        model = LinearRegression
        eval_metrics = ["Accuracy"]
        y_pred, evaluation_results = eval(self.X_train, self.y_train, self.X_test, self.y_test, model, eval_metrics)
        assert isinstance(y_pred, np.ndarray)
        assert "Accuracy" in evaluation_results

    def test_eval_invalid_inputs(self):
        """
        Test evaluation with invalid input data.
        """
        model = LinearRegression()
        eval_metrics = ["Accuracy"]
        X_train_invalid = "invalid_data"
        with pytest.raises(ValueError):
            eval(X_train_invalid, self.y_train, self.X_test, self.y_test, model, eval_metrics)

    def test_eval_with_nan_values(self):
        """
        Test evaluation with NaN values in the input data; it should raise a ValueError.
        """
        X_train_nan = np.array([[1, np.inf], [3, np.nan]])
        X_test_nan = np.array([[5, 6], [7, np.nan]])
        model = LinearRegression()
        eval_metrics = ["Accuracy"]
        with pytest.raises(ValueError):
            eval(X_train_nan, self.y_train, X_test_nan, self.y_test, model, eval_metrics)


# Test class for training-related functions
class TestTraining:
    """
    Test for training-related functions.
    """

    def setup_method(self):
        # Set up common data for the test cases
        self.X_train = np.array([[1, 2], [3, 4]])
        self.y_train = np.array([5, 6])
        self.X_test = np.array([[5, 6], [7, 8]])
        self.y_test = np.array([9, 10])

    def test_valid_regression_model(self):
        """
        Test with a valid regression model, ensuring it does not return an error (-1).
        """
        model_type = "Regression"
        model_selection_radio = "Comparative Analysis"
        model_selector = None
        eval_metrics = ["MSE"]
        comparison_metrics = "MSE"
        slider = None
        result = training(
            model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
            self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
        )
        assert result != -1  # Should not return -1 (error)

    def test_valid_classification_model(self):
        """
        Test with a valid classification model, ensuring it does not return an error (-1).
        """
        model_type = "Classification"
        model_selection_radio = "Comparative Analysis"
        model_selector = None
        eval_metrics = ["Accuracy"]
        comparison_metrics = "Accuracy"
        slider = None
        result = training(
            model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
            self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
        )
        assert result != -1  # Should not return -1 (error)

    def test_invalid_slider_value(self):
        """
        Test with an invalid slider value. It should raise a ValueError.
        """
        with pytest.raises(ValueError):
            model_type = "Regression"
            model_selection_radio = "Comparative Analysis"
            model_selector = None
            eval_metrics = ["MSE"]
            comparison_metrics = "MSE"
            slider = -1
            training(
                model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
                self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
            )

        with pytest.raises(ValueError):
            model_type = "Regression"
            model_selection_radio = "Comparative Analysis"
            model_selector = None
            eval_metrics = ["MSE"]
            comparison_metrics = "MSE"
            slider = 3
            training(
                model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
                self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
            )

    def test_valid_slider_value(self):
        """
        Test with a valid slider value. It should not raise a ValueError.
        """
        model_type = "Regression"
        model_selection_radio = "Comparative Analysis"
        model_selector = None
        eval_metrics = ["MSE"]
        comparison_metrics = "MSE"
        slider = 1
        training(
            model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
            self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
        )

    def test_single_model_selection(self):
        """
        Test with single model selection. It should not raise a ValueError.
        """
        model_type = "Regression"
        model_selection_radio = "Individual Model Selection"
        model_selector = "LinearRegression"
        eval_metrics = ["MSE"]
        comparison_metrics = "MSE"
        slider = None
        training(
            model_type, model_selection_radio, model_selector, self.X_train, self.y_train,
            self.X_test, self.y_test, eval_metrics, comparison_metrics, slider
        )

# Test class for positional argument
class TestPositionalArgument:
    """
    Test for positional argument decorator.
    """

    def test_requires_positional_argument(self):
        """
        Test the `requiresPositionalArgument` decorator to check if it correctly identifies
        whether a scikit-learn estimator requires positional arguments.

        - For LinearRegression, it should return False since it does not require positional arguments.
        - For OneVsRestClassifier, it should return True since it requires positional arguments.
        """
        assert requiresPositionalArgument(LinearRegression) == False
        assert requiresPositionalArgument(OneVsRestClassifier) == True

