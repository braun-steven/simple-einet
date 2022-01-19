# Inference/Backward Time Comparison

The following lists different forward pass and backward pass results for this library (`simple-einet`) in comparison to the official EinsumNetworks implementation ([`EinsumNetworks`](https://github.com/cambridge-mlg/EinsumNetworks)).

The benchmark code can be found in [benchmark.py](./benchmark.py).

The default values for different hyperparameters are as follows:

```python
batch_size = 64
num_features = 512
depth = 5
num_sums = 16
num_leaves = 16
num_channels = 1
num_repetitions = 16
num_classes = 1
```

## Results

Note: All times are in milliseconds (ms).

Summary: It's a tie (?). For some hyper-parameters `EinsumNetworks` is slower than `simple-einet` and for some others its faster. Generally, it seems that the backward pass scales better in `simple-einet`.

### Batch Size

```
[--- batch_size-forward ---]
            |   si   |   og
        --------------------
         1  |   3.5  |   3.9
         2  |   4.2  |   3.5
         4  |   3.7  |   3.2
         8  |   3.8  |   3.5
        16  |   3.8  |   3.7
        32  |   5.2  |   3.9
        64  |   5.4  |   3.8
       128  |   7.3  |   3.9
       256  |  11.1  |   7.1
       512  |  19.7  |  18.2
      1024  |  33.9  |  38.2
      2048  |  67.1  |  74.1


[--- batch_size-backward ----]
            |    si   |    og
        ----------------------
         1  |   10.3  |   10.7
         2  |   10.0  |   11.6
         4  |   10.1  |   11.5
         8  |   10.0  |   12.0
        16  |   11.1  |   13.2
        32  |   11.5  |   13.7
        64  |   12.2  |   12.6
       128  |   17.6  |   13.5
       256  |   25.5  |   26.7
       512  |   45.5  |   63.5
      1024  |   90.2  |  132.7
      2048  |  182.6  |  271.4

```

### Number of Features/Random Variables

```
[-- num_features-forward -]
            |   si   |   og
        -------------------
         4  |   2.2  |  2.1
         8  |   2.7  |  2.7
        16  |   3.1  |  3.6
        32  |   3.7  |  3.8
        64  |   3.6  |  4.8
       128  |   3.8  |  3.8
       256  |   4.3  |  3.9
       512  |   5.4  |  3.8
      1024  |   7.0  |  3.8
      2048  |  11.2  |  5.1


[- num_features-backward --]
            |   si   |   og
        --------------------
         4  |   5.6  |   6.0
         8  |   6.2  |   7.3
        16  |   7.6  |   9.0
        32  |   9.6  |  11.4
        64  |   8.9  |  11.0
       128  |   9.5  |  12.3
       256  |  10.2  |  10.3
       512  |  10.8  |  10.8
      1024  |  14.2  |  12.2
      2048  |  18.9  |  16.9

```

### Depth

```

[----- depth-forward ------]
            |   si   |   og
        --------------------
         1  |   3.2  |   1.5
         2  |   3.7  |   2.2
         3  |   4.1  |   2.6
         4  |   4.6  |   3.2
         5  |   5.2  |   3.4
         6  |   6.1  |   4.3
         7  |   7.4  |   5.6
         8  |   8.8  |   9.2
         9  |  13.5  |  17.3


[----- depth-backward -----]
            |   si   |   og
        --------------------
         1  |   5.4  |   3.8
         2  |   7.3  |   5.7
         3  |   8.7  |   6.9
         4  |  10.6  |  10.2
         5  |  10.6  |  10.4
         6  |  12.6  |  14.0
         7  |  15.6  |  22.8
         8  |  23.0  |  39.9
         9  |  32.3  |  73.9

```

### Number of Sum Nodes per Einsumlayer

```

[---- num_sums-forward ---]
            |   si  |   og
        -------------------
         1  |  5.2  |   3.3
         2  |  6.1  |   3.8
         4  |  5.5  |   3.6
         8  |  5.3  |   4.6
        16  |  5.0  |   3.8
        32  |  5.9  |   3.8
        64  |  8.7  |  11.3


[--- num_sums-backward ----]
            |   si   |   og
        --------------------
         1  |  11.0  |  12.4
         2  |  10.9  |  11.2
         4  |  11.2  |  11.5
         8  |  13.3  |  12.0
        16  |  12.6  |  11.2
        32  |  11.8  |  14.7
        64  |  20.4  |  41.6

```

### Number of Distributions per Feature/Random Variable

```

[--- num_leaves-forward ---]
            |   si   |   og
        --------------------
         1  |   3.6  |   3.2
         2  |   3.7  |   4.6
         4  |   3.6  |   3.6
         8  |   4.8  |   3.7
        16  |   4.9  |   3.7
        32  |   7.5  |   4.7
        64  |  12.7  |  19.7


[-- num_leaves-backward ---]
            |   si   |   og
        --------------------
         1  |   8.8  |  10.5
         2  |  10.8  |  10.2
         4  |   9.1  |  10.9
         8  |   9.7  |  11.3
        16  |  10.6  |  10.5
        32  |  14.9  |  16.8
        64  |  28.9  |  52.6

```

### Number of Input Channels per Feature/Random Variable
(e.g. image color channels: RGB)

```

[-- num_channels-forward --]
            |   si   |   og
        --------------------
         1  |   5.3  |   3.4
         2  |   6.0  |   4.1
         4  |   7.4  |   4.0
         8  |  10.6  |   3.5
        16  |  17.0  |   5.2
        32  |  30.3  |   8.7
        64  |  55.2  |  12.8


[-- num_channels-backward ---]
            |    si   |    og
        ----------------------
         1  |   10.5  |   11.7
         2  |   13.5  |   11.8
         4  |   16.3  |   12.7
         8  |   24.7  |   17.8
        16  |   36.2  |   30.4
        32  |   65.8  |   57.3
        64  |  120.6  |  109.1

```

### Number of Network Repetitions

```

[ num_repetitions-forward -]
            |   si   |   og
        --------------------
         1  |   3.2  |   3.3
         2  |   3.6  |   3.8
         4  |   3.7  |   3.7
         8  |   4.0  |   4.6
        16  |   5.6  |   3.4
        32  |   7.3  |   4.6
        64  |  13.4  |  12.5


[ num_repetitions-backward ]
            |   si   |   og
        --------------------
         1  |   8.9  |  10.6
         2  |  10.3  |  12.4
         4  |  10.8  |  11.1
         8  |  10.7  |  10.7
        16  |  11.2  |  12.4
        32  |  17.0  |  16.5
        64  |  26.5  |  38.4

```

### Number of Classes

```

[- num_classes-forward --]
            |   si  |   og
        ------------------
         1  |  5.2  |  3.7
         2  |  5.3  |  4.0
         4  |  5.7  |  4.1
         8  |  5.2  |  3.7
        16  |  5.2  |  3.4
        32  |  6.2  |  4.3
        64  |  5.1  |  3.7


[-- num_classes-backward --]
            |   si   |   og
        --------------------
         1  |  11.9  |  13.5
         2  |  11.9  |  12.1
         4  |  12.2  |  12.0
         8  |  11.5  |  12.1
        16  |  10.8  |  13.0
        32  |  10.8  |  12.2
        64  |  10.7  |  11.8

```
