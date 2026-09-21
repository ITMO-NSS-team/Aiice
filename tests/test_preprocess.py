import numpy as np
import pytest
import torch

from aiice.preprocess import SlidingWindowDataset, apply_downsample, apply_threshold


class TestSlidingWindowDataset:
    @pytest.mark.parametrize(
        "data, pre, forecast, expected_len, expected_x0, expected_y0",
        [
            # 1d time series
            (
                [1, 2, 3, 4, 5],
                2,
                2,
                2,  # windows: [1,2]->[3,4], [2,3]->[4,5]
                [[1], [2]],  # x[0]
                [[3], [4]],  # y[0]
            ),
            # 2d time series
            (
                torch.Tensor([[1, 10], [2, 20], [3, 30], [4, 40]]),
                2,
                1,
                2,  # windows: 4 - 2 - 1 + 1 = 2
                [[1, 10], [2, 20]],
                [[3, 30]],
            ),
            # 3d time series
            (
                np.array(
                    [
                        [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]],
                        [[9, 10], [11, 12]],
                    ]
                ),  # T=3, shape [3,2,2]
                1,
                1,
                2,
                [[[1, 2], [3, 4]]],
                [[[5, 6], [7, 8]]],
            ),
        ],
    )
    def test_ok(self, data, pre, forecast, expected_len, expected_x0, expected_y0):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=pre,
            forecast_len=forecast,
        )

        assert len(dataset) == expected_len

        x0, y0 = dataset[0]

        assert torch.is_tensor(x0)
        assert torch.is_tensor(y0)

        assert x0.shape[0] == pre
        assert y0.shape[0] == forecast

        np.testing.assert_array_equal(x0.cpu().numpy(), np.array(expected_x0))
        np.testing.assert_array_equal(y0.cpu().numpy(), np.array(expected_y0))

    @pytest.mark.parametrize(
        "data, pre, forecast",
        [
            ([1, 2, 3], 3, 1),  # T = 3, needs minimum 4
            ([1, 2, 3], 1, 3),  # T = 3, needs minimum 4
            ([1, 2], 1, 2),  # T = 2, needs minimum 3
            ([1], 1, 1),  # T = 1, needs minimum 2
        ],
    )
    def test_not_enough_data_raise(self, data, pre, forecast):
        with pytest.raises(ValueError) as exc:
            SlidingWindowDataset(
                data=data,
                pre_history_len=pre,
                forecast_len=forecast,
            )

        assert "Not enough data" in str(exc.value)

    @pytest.mark.parametrize(
        "threshold, x_binarize, expected_x0, expected_y0",
        [
            (
                None,
                False,
                [[1], [2]],
                [[3], [4]],
            ),
            (
                2.5,
                False,
                [[1], [2]],
                [[1], [1]],
            ),
            (
                2.5,
                True,
                [[0], [0]],
                [[1], [1]],
            ),
            (
                0.0,
                True,
                [[1], [1]],
                [[1], [1]],
            ),
        ],
    )
    def test_ok_threshold_binarize(
        self, threshold, x_binarize, expected_x0, expected_y0
    ):
        dataset = SlidingWindowDataset(
            data=[1, 2, 3, 4, 5],
            pre_history_len=2,
            forecast_len=2,
            threshold=threshold,
            x_binarize=x_binarize,
        )

        x0, y0 = dataset[0]

        np.testing.assert_array_equal(x0.numpy(), np.array(expected_x0))
        np.testing.assert_array_equal(y0.numpy(), np.array(expected_y0))

    def test_getitem_non_int_index_raises(self):
        dataset = SlidingWindowDataset(
            data=[1, 2, 3, 4], pre_history_len=2, forecast_len=1
        )
        with pytest.raises(TypeError) as exc:
            _ = dataset["not_an_int"]
        assert "index must be int" in str(exc.value)

    @pytest.mark.parametrize("id", [-1, 2, 3, 10])
    def test_getitem_index_out_of_range_raises(self, id):
        # T=4, pre_history_len=2, forecast_len=1 => len=2, valid indices = 0,1
        dataset = SlidingWindowDataset(
            data=[1, 2, 3, 4], pre_history_len=2, forecast_len=1
        )
        with pytest.raises(IndexError) as exc:
            _ = dataset[id]
            assert "index out of range" in str(exc.value)

    @pytest.mark.parametrize(
        "data, idx, pre, forecast, expected_idx0, expected_x0, expected_y0",
        [
            (
                [1, 2, 3, 4, 5],
                ["a", "b", "c", "d", "e"],
                2,
                2,
                ["a", "b", "c", "d"],
                [[1], [2]],
                [[3], [4]],
            ),
            # 2d time series with numeric indices
            (
                [[1, 10], [2, 20], [3, 30], [4, 40]],
                [100, 101, 102, 103],
                2,
                1,
                [100, 101, 102],
                [[1, 10], [2, 20]],
                [[3, 30]],
            ),
        ],
    )
    def test_ok_idx(
        self, data, idx, pre, forecast, expected_idx0, expected_x0, expected_y0
    ):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=pre,
            forecast_len=forecast,
            idx=idx,
        )

        idx_slice, x0, y0 = dataset[0]

        assert idx_slice == expected_idx0
        np.testing.assert_array_equal(x0.numpy(), np.array(expected_x0))
        np.testing.assert_array_equal(y0.numpy(), np.array(expected_y0))

    @pytest.mark.parametrize(
        "data, idx",
        [
            (
                [-1, 2, 3, 10],
                [1, 2, 3],
            ),
            (
                [[1, 10], [2, 20]],
                ["a"],
            ),
        ],
    )
    def test_idx_length_vs_data_length_raises(self, data, idx):
        with pytest.raises(ValueError) as exc:
            _ = SlidingWindowDataset(
                data=data, idx=idx, pre_history_len=2, forecast_len=1
            )
            assert (
                f"Data length (got {len(data)}) should be equal to indices length (got {len(idx)})"
                in str(exc.value)
            )


class TestSlidingWindowDatasetChannels:
    """
    Data shaped [T, C, H, W]: channel 0 is the target, 1 and 2 are predictors.
    """

    @pytest.fixture
    def data(self):
        # T=5, C=3, H=2, W=2; value encodes (time, channel) as t*10 + c
        return torch.stack(
            [
                torch.stack([torch.full((2, 2), float(t * 10 + c)) for c in range(3)])
                for t in range(5)
            ]
        )

    def test_default_keeps_every_channel(self, data):
        dataset = SlidingWindowDataset(data=data, pre_history_len=2, forecast_len=2)
        x0, y0 = dataset[0]

        assert x0.shape == (2, 3, 2, 2)
        assert y0.shape == (2, 3, 2, 2)

    def test_y_channels_narrows_target_only(self, data):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=2,
            forecast_len=2,
            y_channels=[0],
        )
        x0, y0 = dataset[0]

        assert x0.shape == (2, 3, 2, 2)  # model still sees every predictor
        assert y0.shape == (2, 1, 2, 2)  # but forecasts the target only

        # y[0] is t=2, channel 0 => 20
        np.testing.assert_array_equal(y0[0, 0].numpy(), np.full((2, 2), 20.0))

    def test_x_and_y_channels(self, data):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=2,
            forecast_len=1,
            x_channels=[0, 2],
            y_channels=[0],
        )
        x0, y0 = dataset[0]

        assert x0.shape == (2, 2, 2, 2)
        assert y0.shape == (1, 1, 2, 2)

        # x[0] is t=0 => channels 0 and 2 hold 0 and 2
        np.testing.assert_array_equal(x0[0, 0].numpy(), np.full((2, 2), 0.0))
        np.testing.assert_array_equal(x0[0, 1].numpy(), np.full((2, 2), 2.0))

    def test_negative_channel_index(self, data):
        dataset = SlidingWindowDataset(
            data=data, pre_history_len=2, forecast_len=1, y_channels=[-1]
        )
        _, y0 = dataset[0]

        assert y0.shape == (1, 1, 2, 2)
        # last channel of t=2 => 22
        np.testing.assert_array_equal(y0[0, 0].numpy(), np.full((2, 2), 22.0))

    def test_threshold_applies_after_selection(self, data):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=2,
            forecast_len=1,
            y_channels=[0],
            threshold=15.0,
        )
        _, y0 = dataset[0]

        # only the kept channel is binarized: t=2 channel 0 is 20 > 15
        assert y0.shape == (1, 1, 2, 2)
        np.testing.assert_array_equal(y0[0, 0].numpy(), np.ones((2, 2)))

    def test_channels_with_idx(self, data):
        dataset = SlidingWindowDataset(
            data=data,
            pre_history_len=2,
            forecast_len=1,
            idx=["a", "b", "c", "d", "e"],
            y_channels=[0],
        )
        idx_slice, x0, y0 = dataset[0]

        assert idx_slice == ["a", "b", "c"]
        assert x0.shape == (2, 3, 2, 2)
        assert y0.shape == (1, 1, 2, 2)

    @pytest.mark.parametrize("channels", [[3], [-4], [0, 7]])
    def test_channel_out_of_range_raises(self, data, channels):
        with pytest.raises(ValueError) as exc:
            SlidingWindowDataset(
                data=data,
                pre_history_len=2,
                forecast_len=1,
                y_channels=channels,
            )
        assert "out of range" in str(exc.value)

    @pytest.mark.parametrize("name", ["x_channels", "y_channels"])
    def test_empty_channels_raises(self, data, name):
        with pytest.raises(ValueError) as exc:
            SlidingWindowDataset(
                data=data,
                pre_history_len=2,
                forecast_len=1,
                **{name: []},
            )
        assert f"{name} must not be empty" in str(exc.value)


class TestApply:
    @pytest.mark.parametrize(
        "tensor, threshold, expected",
        [
            (
                torch.tensor([0.2, 0.6, 0.5]),
                0.5,
                [0, 1, 0],
            ),
            (
                torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
                0.4,
                [[0, 1], [1, 0]],
            ),
            (
                torch.tensor([1.0, 2.0, 3.0]),
                2.5,
                [0, 0, 1],
            ),
        ],
    )
    def test_apply_threshold(self, tensor, threshold, expected):
        out = apply_threshold(tensor, threshold)
        assert torch.is_tensor(out)
        np.testing.assert_array_equal(out.cpu().numpy(), np.array(expected))

    @pytest.mark.parametrize(
        "tensor, i, axes, expected_shape, expected_values",
        [
            # 1D, axis=-1 (default)
            (
                torch.arange(6),
                2,
                (-1,),
                (3,),
                [0, 2, 4],
            ),
            # 2D, downsample first axis
            (
                torch.arange(12).reshape(3, 4),
                2,
                (0,),
                (2, 4),
                [[0, 1, 2, 3], [8, 9, 10, 11]],
            ),
            # 3D, downsample last axis
            (
                torch.arange(24).reshape(2, 3, 4),
                3,
                (-1,),
                (2, 3, 2),
                [[[0, 3], [4, 7], [8, 11]], [[12, 15], [16, 19], [20, 23]]],
            ),
            # 3D, downsample axes 1 and 2
            (
                torch.arange(24).reshape(2, 3, 4),
                2,
                (1, 2),
                (2, 2, 2),
                [[[0, 2], [8, 10]], [[12, 14], [20, 22]]],
            ),
        ],
    )
    def test_apply_downsample_ok(
        self, tensor, i, axes, expected_shape, expected_values
    ):
        out = apply_downsample(tensor, i, axes=axes)
        assert torch.is_tensor(out)
        assert tuple(out.shape) == expected_shape
        np.testing.assert_array_equal(out.cpu().numpy(), np.array(expected_values))

    @pytest.mark.parametrize("i", [0, -1])
    def test_apply_downsample_raise(self, i):
        t = torch.arange(10)
        with pytest.raises(ValueError) as err:
            apply_downsample(t, i)
        assert "i must be > 0" in str(err.value)
