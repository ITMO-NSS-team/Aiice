# Aiice

[![uv](https://img.shields.io/badge/uv-F0DB4F?style=flat&logo=uv&logoColor=black)](https://uv.io)
[![Hugging Face](https://img.shields.io/badge/huggingface-FF9900?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/pytorch-CB2C31?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

---

**AIICE** is an open-source Python framework designed as a standardized benchmark for spatio-temporal forecasting of Arctic sea ice concentration. It provides reproducible pipelines for loading, preprocessing, and evaluating satellite-derived OSI-SAF data, supporting both short- and long-term prediction horizons

## Installation

The simplest way to install framework with `pip`:
```shell
pip install aiice-bench
```

## Quickstart

The AIICE class provides a simple interface for loading Arctic ice data, preparing datasets, and benchmarking PyTorch models:

![image](.doc/media/aiice-flow.png)

```python
from aiice import AIICE

# Initialize AIICE with a sliding window 
# of past 30 days and forecast of 7 days
aiice = AIICE(
    pre_history_len=30,
    forecast_len=7,
    batch_size=32,
    start="2022-01-01",
    end="2022-12-31"
)

# Define your PyTorch model
model = MyModel()

# Run benchmarking to compute metrics on the dataset
report = aiice.bench(model)
print(report)
```

Check **[package doc](https://itmo-nss-team.github.io/Aiice/aiice.html)** and see more **[usage examples](https://github.com/ITMO-NSS-team/Aiice/tree/main/scripts/examples)**. You can also explore the **[raw dataset](https://huggingface.co/datasets/ITMO-NSS/Aiice)** and work with it independently via Hugging Face

## Leaderboard

The leaderboard reports the mean performance of each model across the evaluation dataset. You can check models' setup in [examples](./scripts/experiments).

<!-- benchmark -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>baseline_mean</th>
      <th>baseline_repeat</th>
      <th>conv2d</th>
      <th>conv3d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">Barents Sea</th>
      <th>bin_accuracy</th>
      <td>0.874963</td>
      <td>0.848936</td>
      <td><b>0.937071</b></td>
      <td>0.891255</td>
    </tr>
    <tr>
      <th>iou</th>
      <td>0.185126</td>
      <td>0.331170</td>
      <td><b>0.647688</b></td>
      <td>0.420801</td>
    </tr>
    <tr>
      <th>mae</th>
      <td>0.130236</td>
      <td>0.151377</td>
      <td><b>0.067575</b></td>
      <td>0.113846</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>0.053554</td>
      <td>0.106431</td>
      <td><b>0.028444</b></td>
      <td>0.064654</td>
    </tr>
    <tr>
      <th>psnr</th>
      <td>12.712070</td>
      <td>9.729317</td>
      <td><b>15.460110</b></td>
      <td>11.894089</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.231418</td>
      <td>0.326238</td>
      <td><b>0.168653</b></td>
      <td>0.254271</td>
    </tr>
    <tr>
      <th>ssim</th>
      <td>0.540464</td>
      <td>0.609196</td>
      <td><b>0.696043</b></td>
      <td>0.618139</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Chukchi Sea</th>
      <th>bin_accuracy</th>
      <td>0.656515</td>
      <td>0.675528</td>
      <td><b>0.947459</b></td>
      <td>0.789110</td>
    </tr>
    <tr>
      <th>iou</th>
      <td>0.126601</td>
      <td>0.364351</td>
      <td><b>0.865943</b></td>
      <td>0.585862</td>
    </tr>
    <tr>
      <th>mae</th>
      <td>0.269926</td>
      <td>0.300754</td>
      <td><b>0.069100</b></td>
      <td>0.198657</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>0.124306</td>
      <td>0.246038</td>
      <td><b>0.023475</b></td>
      <td>0.125997</td>
    </tr>
    <tr>
      <th>psnr</th>
      <td>9.055069</td>
      <td>6.089983</td>
      <td><b>16.293947</b></td>
      <td>8.996499</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.352571</td>
      <td>0.496022</td>
      <td><b>0.153215</b></td>
      <td>0.354958</td>
    </tr>
    <tr>
      <th>ssim</th>
      <td>0.405798</td>
      <td>0.385161</td>
      <td><b>0.651510</b></td>
      <td>0.449680</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Kara Sea</th>
      <th>bin_accuracy</th>
      <td>0.801598</td>
      <td>0.797711</td>
      <td><b>0.939550</b></td>
      <td>0.844245</td>
    </tr>
    <tr>
      <th>iou</th>
      <td>0.282630</td>
      <td>0.412451</td>
      <td><b>0.785852</b></td>
      <td>0.559398</td>
    </tr>
    <tr>
      <th>mae</th>
      <td>0.162785</td>
      <td>0.185920</td>
      <td><b>0.065702</b></td>
      <td>0.149524</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>0.070723</td>
      <td>0.136968</td>
      <td><b>0.025262</b></td>
      <td>0.092185</td>
    </tr>
    <tr>
      <th>psnr</th>
      <td>11.504373</td>
      <td>8.633821</td>
      <td><b>15.975368</b></td>
      <td>10.358038</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.265939</td>
      <td>0.370091</td>
      <td><b>0.158939</b></td>
      <td>0.303539</td>
    </tr>
    <tr>
      <th>ssim</th>
      <td>0.604080</td>
      <td>0.590542</td>
      <td><b>0.725831</b></td>
      <td>0.589535</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Laptev Sea</th>
      <th>bin_accuracy</th>
      <td>0.839829</td>
      <td>0.863018</td>
      <td><b>0.964288</b></td>
      <td>0.897629</td>
    </tr>
    <tr>
      <th>iou</th>
      <td>0.387533</td>
      <td>0.534633</td>
      <td><b>0.859092</b></td>
      <td>0.683309</td>
    </tr>
    <tr>
      <th>mae</th>
      <td>0.115111</td>
      <td>0.122237</td>
      <td><b>0.043340</b></td>
      <td>0.094628</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>0.051770</td>
      <td>0.094377</td>
      <td><b>0.015273</b></td>
      <td>0.066438</td>
    </tr>
    <tr>
      <th>psnr</th>
      <td>12.859248</td>
      <td>10.251351</td>
      <td><b>18.160892</b></td>
      <td>11.784326</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.227529</td>
      <td>0.307208</td>
      <td><b>0.123582</b></td>
      <td>0.257630</td>
    </tr>
    <tr>
      <th>ssim</th>
      <td>0.782073</td>
      <td>0.746823</td>
      <td><b>0.837163</b></td>
      <td>0.802543</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Sea of Japan</th>
      <th>bin_accuracy</th>
      <td>0.994356</td>
      <td>0.989473</td>
      <td>0.994356</td>
      <td><b>0.995731</b></td>
    </tr>
    <tr>
      <th>iou</th>
      <td>0.000000</td>
      <td><b>0.035046</b></td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mae</th>
      <td>0.013824</td>
      <td>0.016332</td>
      <td>0.009841</td>
      <td><b>0.008582</b></td>
    </tr>
    <tr>
      <th>mse</th>
      <td><b>0.004467</b></td>
      <td>0.009577</td>
      <td>0.005990</td>
      <td>0.004908</td>
    </tr>
    <tr>
      <th>psnr</th>
      <td><b>23.499943</b></td>
      <td>20.187490</td>
      <td>22.225956</td>
      <td>22.945065</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td><b>0.066835</b></td>
      <td>0.097865</td>
      <td>0.077393</td>
      <td>0.069567</td>
    </tr>
    <tr>
      <th>ssim</th>
      <td>0.841847</td>
      <td>0.879064</td>
      <td><b>0.922021</b></td>
      <td>0.919443</td>
    </tr>
  </tbody>
</table>
<!-- benchmark -->
