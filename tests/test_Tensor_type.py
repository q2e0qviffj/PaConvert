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

obj = APIBase("torch.Tensor.type")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a =torch.ones(2, 3)
        result = a.type(torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a =torch.ones((10,10))
        result = a.type(torch.int8, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([1,2,3])
        result = a.type(dtype=torch.float64, non_blocking=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([1,2,3])
        result = a.type().split(".", 1)[1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a =torch.ones(2, 3)
        result = a.type(non_blocking=True, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        result = a.type("torch.DoubleTensor")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3, device="cpu")
        result = a.type(dtype=torch.float64, non_blocking=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        args = (torch.float64, True)
        result = a.type(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        kwargs = {"dtype": torch.float64, "non_blocking": True}
        result = a.type(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        result = a.type(torch.DoubleTensor)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3, device="cpu")
        kwargs = {"dtype": torch.float64, "async": True}
        result = a.type(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        a = torch.sparse_coo_tensor(
            indices, values, [2, 2], device="cpu"
        )
        result = a.type().split(".", 1)[1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = [
            t.type().split(".", 1)[1]
            for t in (
                torch.ones(1, dtype=torch.bool, device="cpu"),
                torch.ones(1, dtype=torch.int32, device="cpu"),
                torch.ones(1, dtype=torch.float64, device="cpu"),
                torch.ones(1, dtype=torch.bfloat16, device="cpu"),
                torch.ones(1, dtype=torch.complex64, device="cpu"),
            )
        ]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = a.type(torch.float64)
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3, device="cpu")
        result = (
            a.type(torch.float32) is a,
            a.type("torch.FloatTensor") is a,
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(2, 3)
        result = [
            a.type("torch.Float8_e4m3fnTensor").type().split(".", 1)[1],
            a.type("torch.Float8_e5m2Tensor").type().split(".", 1)[1],
        ]
        """
    )
    obj.run(pytorch_code, ["result"])
