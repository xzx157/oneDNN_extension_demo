import os

os.environ.setdefault("ODNN_ENABLE_ATEN_HIJACK", "1")

import torch

import oneDNN_extension_demo as odnn
from oneDNN_extension_demo.cpp_extension import hijack_extension_status


def _assert_relu_values(actual, values):
    actual_values = actual.detach().cpu().flatten().tolist()
    expected_values = [max(float(value), 0.0) for value in values]
    assert len(actual_values) == len(expected_values)
    for actual_value, expected_value in zip(actual_values, expected_values):
        assert abs(float(actual_value) - expected_value) <= 1e-6, (
            actual_value,
            expected_value,
        )


def main():
    assert callable(odnn.enable_aten_hijack)
    status = hijack_extension_status()
    assert status == {"loaded": True, "source": "prebuilt-wheel"}, status

    values = [-2.0, -0.5, 0.0, 1.25, 3.5]
    _assert_relu_values(torch.relu(torch.tensor(values, dtype=torch.float32)), values)
    _assert_relu_values(torch.relu(torch.tensor(values, dtype=torch.float64)), values)

    image = torch.tensor(values * 16, dtype=torch.float32).reshape(1, 4, 4, 5)
    image = image.contiguous(memory_format=torch.channels_last)
    _assert_relu_values(torch.relu(image), image.flatten().tolist())

    print("aten hijack smoke test passed:", status)


if __name__ == "__main__":
    main()
