# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import textwrap

from test_numel import NumelAPIBase

obj = NumelAPIBase("torch.Tensor.numel")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.randn(1, 2, 3, 4, 5)
        result = a.numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(4,4)
        result = a.numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.zeros(4,4).numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.empty(0, 3).numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        count = a.numel()
        result = count * 3 + 1
        is_python_int = type(count) is int
        """
    )
    obj.run(pytorch_code, ["result", "is_python_int"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
        result = a.numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(7, dtype=torch.int32)
        result = a.numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([[0, 1], [1, 2]])
        values = torch.tensor([1.0, 2.0])
        a = torch.sparse_coo_tensor(indices, values, [2, 3])
        result = a.numel()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        args = ()
        result = a.numel(*args)
        """
    )
    obj.run(pytorch_code, ["result"])
