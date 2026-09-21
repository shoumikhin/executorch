# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# make_test.py is a helper that builds models, so it imports torch at module scope,
# and its name matches what pytest collects. Collecting it fails the whole run on a
# machine without torch, which is the one configuration these tests exist to cover.
collect_ignore = ["make_test.py"]
