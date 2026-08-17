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

from dist_apibase import DistributionAPIBase

obj = DistributionAPIBase("torch.distributions.categorical.Categorical")


def test_case_1():
    """Categorical with positional probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(torch.tensor([0.3, 0.3, 0.4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """Categorical with probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Categorical with logits"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Categorical with logits and validate_args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]), validate_args=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Categorical with probs=None, logits provided"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=None, logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Categorical with probs and logits=None"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]), logits=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Categorical with keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(validate_args=False, logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.categorical.Categorical(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """float64 dtype input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.distributions.categorical.Categorical(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Expression as argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Categorical with sample"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """All positional, mixed, and variadic constructor arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        logits = torch.tensor([[0.2, 0.3, 0.5], [1.0, -1.0, 0.5]])
        result1 = torch.distributions.categorical.Categorical(None, logits, False)
        result2 = torch.distributions.categorical.Categorical(None, logits=logits, validate_args=False)
        args = (None, logits, False)
        result3 = torch.distributions.categorical.Categorical(*args)
        """
    )
    obj.run(pytorch_code, ["result1", "result2", "result3"])


def test_case_13():
    """Distribution properties and probability methods with batched input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        probs = torch.tensor([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
        dist = torch.distributions.categorical.Categorical(probs=probs)
        value = torch.tensor([2, 0])
        log_prob = dist.log_prob(value)
        entropy = dist.entropy()
        perplexity = dist.perplexity()
        mode = dist.mode
        mean_is_nan = torch.isnan(dist.mean)
        variance_is_nan = torch.isnan(dist.variance)
        param_shape = dist.param_shape
        batch_shape = dist.batch_shape
        event_shape = dist.event_shape
        has_enumerate_support = dist.has_enumerate_support
        has_rsample = dist.has_rsample
        """
    )
    obj.run(
        pytorch_code,
        [
            "log_prob",
            "entropy",
            "perplexity",
            "mode",
            "mean_is_nan",
            "variance_is_nan",
            "param_shape",
            "batch_shape",
            "event_shape",
            "has_enumerate_support",
            "has_rsample",
        ],
    )


def test_case_14():
    """Zero probability keeps finite logits and entropy semantics"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        dist = torch.distributions.categorical.Categorical(
            probs=torch.tensor([0.0, 0.25, 0.75])
        )
        logits = dist.logits
        entropy = dist.entropy()
        log_prob = dist.log_prob(torch.tensor([0, 2]))
        """
    )
    obj.run(pytorch_code, ["logits", "entropy", "log_prob"])


def test_case_15():
    """Support checks and support enumeration"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        dist = torch.distributions.categorical.Categorical(probs=probs)
        support_check = dist.support.check(torch.tensor([0, 2, -1, 3]))
        support_expanded = dist.enumerate_support(expand=True)
        support_unexpanded = dist.enumerate_support(expand=False)
        """
    )
    obj.run(
        pytorch_code,
        ["support_check", "support_expanded", "support_unexpanded"],
    )


def test_case_16():
    """Expand preserves normalized parameters and distribution shapes"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        logits = torch.tensor([[0.2, 0.3, 0.5], [1.0, -1.0, 0.5]])
        dist = torch.distributions.categorical.Categorical(logits=logits)
        result = dist.expand((4, 2))
        expanded_log_prob = result.log_prob(torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]]))
        """
    )
    obj.run(pytorch_code, ["result", "expanded_log_prob"])


def test_case_17():
    """Sample shape, dtype, and support"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        dist = torch.distributions.categorical.Categorical(
            probs=torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        )
        samples = dist.sample((2, 3))
        sample_shape = samples.shape
        sample_dtype_is_int64 = samples.dtype == torch.int64
        samples_in_support = dist.support.check(samples).all()
        """
    )
    obj.run(
        pytorch_code,
        ["sample_shape", "sample_dtype_is_int64", "samples_in_support"],
    )


def test_case_18():
    """Gradient through normalized logits, log_prob, and entropy"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        logits = torch.tensor([0.2, -0.1, 0.7], requires_grad=True)
        dist = torch.distributions.categorical.Categorical(logits=logits)
        loss = dist.log_prob(torch.tensor(2)) + dist.entropy()
        loss.backward()
        logits_grad = logits.grad
        """
    )
    obj.run(
        pytorch_code,
        ["loss", "logits_grad"],
        check_stop_gradient=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_case_19():
    """Constructor and sample validation errors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        errors = []
        try:
            torch.distributions.categorical.Categorical()
        except ValueError:
            errors.append("missing")
        try:
            torch.distributions.categorical.Categorical(
                probs=torch.tensor([0.2, 0.8]),
                logits=torch.tensor([0.2, 0.8]),
            )
        except ValueError:
            errors.append("both")
        try:
            torch.distributions.categorical.Categorical(torch.tensor(1.0))
        except ValueError:
            errors.append("scalar")
        dist = torch.distributions.categorical.Categorical(
            probs=torch.tensor([0.2, 0.3, 0.5]),
            validate_args=True,
        )
        try:
            dist.log_prob(torch.tensor([3]))
        except ValueError:
            errors.append("out_of_support")
        try:
            dist.log_prob(torch.tensor([1.5]))
        except ValueError:
            errors.append("non_integer")
        try:
            torch.distributions.categorical.Categorical(
                probs=torch.tensor([-0.1, 1.1]),
                validate_args=True,
            )
        except ValueError:
            errors.append("invalid_probs")
        try:
            torch.distributions.categorical.Categorical(
                logits=torch.tensor([float("nan"), 0.0]),
                validate_args=True,
            )
        except ValueError:
            errors.append("invalid_logits")
        try:
            torch.distributions.categorical.Categorical(torch.empty((0,)))
        except ValueError:
            errors.append("empty")
        """
    )
    obj.run(pytorch_code, ["errors"])


def test_case_20():
    """Integer probabilities and empty categories with validation disabled"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        integer_result = torch.distributions.categorical.Categorical(
            probs=torch.tensor([1, 2, 3])
        )
        empty_result = torch.distributions.categorical.Categorical(
            probs=torch.empty(0),
            validate_args=False,
        )
        """
    )
    obj.run(pytorch_code, ["integer_result", "empty_result"])
