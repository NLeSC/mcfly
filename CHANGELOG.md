# Change Log

## v3.1.0
- Added support for training with Datasets, generators, and other data types supported by Keras [#211](https://github.com/NLeSC/mcfly/issues/211)
- Added separate classes for each model type, also allowing easier extension by own custom models [#239](https://github.com/NLeSC/mcfly/pull/239)
- Dropped support for Python 3.5 (supported are: Python 3.6, 3.7 and 3.8)
- Extend and update documentation in readthedocs [#250](https://github.com/NLeSC/mcfly/pull/250)
- Fix broken model visualization [#256](https://github.com/NLeSC/mcfly/pull/256)

## v3.0.0
- Add ResNet architecture
- Add InceptionTime architecture
- Tensorflow dependency to 2.0
- Dropped support for Python 2.7 and added support for Python 3.7
- Early_stopping argument for train_models_on_samples() changed to early_stopping_patience
- Fix metric name issue in visualization
- Lower level functions arguments have changed by using keyword arguments dic

## v2.1.0
- Add class weight support

## v2.0.1
- Fix documentation inconsistency

## v2.0.0
- Using Tensorflow.keras instead of Keras
- Using keyword 'accuracy' instead of 'acc' in logs like latest keras versions do

## v1.0.5
- Requirements change (keras<2.3.0)

## v1.0.4
- Fixed Zenodo configuration

## v1.0.3
- Requirements change (six>=1.10.0)
- Fixed Zenodo configuration

## v1.0.2
- Small bug fixes
- Added metric option to model training
- Extended documentation
- Removed redundant dependency on Pandas

## v1.0.1
- Small bug fixes
- Compatible with Keras v2.x (no longer compatible with Keras < v2.0.0)
- Tutorial is moved to separate repository (https://github.com/NLeSC/mcfly-tutorial)

## v1.0.0
First major release.
