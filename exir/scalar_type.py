# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from enum import IntEnum


class ScalarType(IntEnum):
    BYTE = 0
    CHAR = 1
    SHORT = 2
    INT = 3
    LONG = 4
    HALF = 5
    FLOAT = 6
    DOUBLE = 7
    COMPLEX32 = 8
    COMPLEX64 = 9
    COMPLEX128 = 10
    BOOL = 11
    QINT8 = 12
    QUINT8 = 13
    QINT32 = 14
    BFLOAT16 = 15
    QUINT4x2 = 16
    QUINT2x4 = 17
    BITS16 = 22
    FLOAT8E5M2 = 23
    FLOAT8E4M3FN = 24
    FLOAT8E5M2FNUZ = 25
    FLOAT8E4M3FNUZ = 26
    UINT16 = 27
    UINT32 = 28
    UINT64 = 29
