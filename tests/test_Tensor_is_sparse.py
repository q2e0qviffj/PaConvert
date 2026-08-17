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

from apibase import APIBase

obj = APIBase("torch.Tensor.is_sparse")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9254, -0.6213]])
        result = a.is_sparse
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        a = torch.sparse_coo_tensor(indices, values, [2, 2])
        result = a.is_sparse
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = torch.tensor([0, 1, 1])
        cols = torch.tensor([0])
        values = torch.tensor([1.0])
        a = torch.sparse_csr_tensor(crows, cols, values, [2, 2])
        result = a.is_sparse
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.empty([0, 3], dtype=torch.float64)
        result = a.is_sparse
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
        result = a.is_sparse
        result_type = type(result).__name__
        """
    )
    obj.run(pytorch_code, ["result", "result_type"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([[0, 1], [1, 2]])
        values = torch.tensor([1, 2], dtype=torch.int64)
        a = torch.sparse_coo_tensor(indices, values, [2, 3])
        result = a.is_sparse
        result_type = type(result).__name__
        """
    )
    obj.run(pytorch_code, ["result", "result_type"])
