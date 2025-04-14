import os
import unittest

import numpy as np
from keras.models import load_model

from tests.test_tools import safe_remove

from mcfly import modelgen, find_architecture


class IntegrationSuite(unittest.TestCase):

    def run_integration(self):
        """
        Does most of the operations in the tutorial and uses many of mcfly's functionalities consecutively.
        """
        X_train, X_val, y_train, y_val, num_classes, x_shape = self.generate_random_data_sets()

        metric = "accuracy"
        models = modelgen.generate_models(
            x_shape,
            number_of_output_dimensions=num_classes,
            number_of_models=2,
            metrics=[metric],
            model_type="CNN",  # Because CNNs are quick to train.
        )
        histories, val_accuracies, _ = find_architecture.train_models_on_samples(
            X_train, y_train,
            X_val, y_val,
            models, nr_epochs=5,
            verbose=True,
            outputfile=self.outputfile,
        )
        best_model_index = np.argmax(val_accuracies[metric])
        best_model, _, _ = models[best_model_index]
        _ = best_model.fit(
            X_train, y_train,
            epochs=2,
            validation_data=X_val if y_val is None else (X_val, y_val)
        )
        best_model.save(self.modelfile)
        model_reloaded = load_model(self.modelfile, safe_mode=False)
        assert model_reloaded is not None, "Expected model"  # TODO: check if it's a real model
        assert len(histories) == 2, "Expected two models in histories"
        assert os.path.exists(self.outputfile)
        assert os.path.exists(self.modelfile)

    def generate_random_data_sets(self):
        X_train = np.random.randn(200, 512, 9)  # (11397, 512, 9)
        y_train = np.random.randn(200, 7)       # (11397, 512, 9)
        X_val = np.random.randn(100, 512, 9)
        y_val = np.random.randn(100, 7)
        num_classes = y_train.shape[1]
        x_shape = X_train.shape
        return X_train, X_val, y_train, y_val, num_classes, x_shape

    def setUp(self):
        np.random.seed(1234)
        self.outputfile = ".generated_model_comparison_for_integration_test.json"
        self.modelfile = ".generated_model_for_integration_test.keras"
        safe_remove(self.outputfile)
        safe_remove(self.modelfile)

    def tearDown(self):
        safe_remove(self.outputfile)
        safe_remove(self.modelfile)
