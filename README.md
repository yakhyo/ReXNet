# ReXNet
Rethinking Channel Dimensions for Efficient Model Design | [Paper](https://arxiv.org/abs/2007.00992) | [Official](https://github.com/clovaai/rexnet)

<!---
Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license
--->

### Results:

<table>
  <tr>
    <td></td>
    <td colspan="4" align="center">This Repo.</td>
    <td colspan="2" align="center">Official</td>
  </tr>
  <tr>
    <td>Model</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Params</td>
    <td>Weights</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
  </tr>
  <tr>
    <td>ReXNet_1.0</td>
    <td><strong>72.4</strong></td>
    <td><strong>90.8</strong></td>
    <td>4.8M</td>
    <td><a href='https://'>soon</a></td>
    <td>77.9</td>
    <td>93.9</td>
  </tr>
  <tr>
    <td>ReXNet_1.3</td>
    <td><strong>74.2</strong></td>
    <td><strong>92.0</strong></td>
    <td>7.6M</td>
    <td><a href='https://'>soon</a></td>
    <td>79.5</td>
    <td>94.7</td>
  </tr>
  <tr>
    <td>ReXNet_1.5</td>
    <td><strong>75.05</strong></td>
    <td><strong>92.37</strong></td>
    <td>7.6M</td>
    <td><a href='https://'>soon</a></td>
    <td>80.3</td>
    <td>95.2</td>
  </tr>
  <tr>
    <td>ReXNet_2.0</td>
    <td><strong>...</strong></td>
    <td><strong>...</strong></td>
    <td>16M</td>
    <td><a href='https://'>soon</a></td>
    <td>81.6</td>
    <td>95.7</td>
  </tr>
  <tr>
    <td>ReXNet_2.2</td>
    <td><strong>...</strong></td>
    <td><strong>...</strong></td>
    <td>19M</td>
    <td><a href='https://'>soon</a></td>
    <td>81.7</td>
    <td>95.8</td>
  </tr>
  <tr>
    <td>ReXNet_3.0</td>
    <td><strong>...</strong></td>
    <td><strong>...</strong></td>
    <td>34M</td>
    <td><a href='https://'>soon</a></td>
    <td>82.8</td>
    <td>96.2</td>
  </tr>
</table>


Trained on ImageNet (90 epochs)
- GPU: Tesla V100
- Input size: 3x224x224

Dataset structure:

```
├── IMAGENET 
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ....
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
```



### Reference
- Clova AI



