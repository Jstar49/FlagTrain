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

"""Moore Threads MUSA Backend Configuration

The device is reached through ``torch.musa``, the module its ``torch_musa``
package registers, and ``MUSA_VISIBLE_DEVICES`` selects the cards. Its operator
dispatch is a private-use key rather than an upstream one -- MUSA is not a
device torch ships -- so ``dispatch_key`` has to be spelled out instead of
defaulting to the upper-cased device name.

``mthreads-gmi`` is the device-management tool (the vendor's nvidia-smi), which
is what both the presence check and ``tools/gpu_check_mthreads.sh`` read.

``fp64_enabled`` is off because MUSA has no fp64: declaring it here is what lets
``tests/accuracy_utils.py`` pick its fallback dtypes rather than have a test
discover the gap at run time. ``tle_enabled`` turns on the triton language
extensions the MUSA flagtree compiler implements.
"""

from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="mthreads",
    device_name="musa",
    dispatch_key="PrivateUse1",
    device_query_cmd="mthreads-gmi",
    fp64_enabled=False,
    tle_enabled=True,
)

# Operators that should use PyTorch native implementation. Empty: nothing in
# flag_train is known to be both unsupported and worth not registering yet.
CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["vendor_info"]
