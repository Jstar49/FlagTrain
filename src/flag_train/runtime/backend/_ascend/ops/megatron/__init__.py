# Copyright 2026 FlagOS Contributors
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
"""Backend ``megatron`` operators.

Mirrors ``flag_train.megatron``: a specialised operator goes in the same-named
module here as the generic one it replaces, so which generic module an override
belongs to is readable from the path.

To specialise an operator, define it under this package, re-export it here
(``from .<name> import <name>``) and list it in ``__all__`` -- that is what
``SpecOpRegistrar`` registers, overriding the same-named generic implementation.
``__all__`` is therefore the source of truth for which operators this backend
overrides.
"""

from .fused_indices_to_multihot import fused_indices_to_multihot

__all__ = [
    "fused_indices_to_multihot",
]
