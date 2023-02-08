#
# mcfly
#
# Copyright 2017 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from .modelgen import generate_models
from .find_architecture import train_models_on_samples, find_best_architecture, kNN_performance
from ._version import __version__


__author__ = "Netherlands eScience Center"
__all__ = [
    "__version__",
    "find_best_architecture",
    "generate_models",
    "train_models_on_samples",
    "kNN_performance",
]
