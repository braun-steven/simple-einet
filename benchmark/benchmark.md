# Inference/Backward Time Comparison

The following lists different forward pass and backward pass results for this library (`simple-einet`) in comparison to the official EinsumNetworks implementation ([`EinsumNetworks`](https://github.com/cambridge-mlg/EinsumNetworks)).

The benchmark code can be found in [benchmark.py](./benchmark.py).

The default values for different hyperparameters are as follows:

```python
batch_size = 256
num_features = 512
depth = 5
num_sums = 32
num_leaves = 32
num_repetitions = 32
num_channels = 1
num_classes = 1
```

## Results

The `simple-einet` implementation is 1.5x - 3.0x faster almost everywhere but scales similar to the official `EinsumNetworks` implementation

`OOM` indicates an `OutOfMemory` runtime exception.

```
[------------ batch_size-forward ------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |       2.5      |        3.4     
         2  |       2.3      |        3.3     
         4  |       2.4      |        3.1     
         8  |       2.9      |        3.1     
        16  |       4.7      |        3.8     
        32  |       8.6      |        6.7     
        64  |      14.4      |       14.5     
       128  |      27.3      |       36.8     
       256  |      54.2      |       75.3     
       512  |     106.0      |      146.1     
      1024  |     211.7      |      292.5     
      2048  |     418.7      |      575.9     

Times are in milliseconds (ms).

[----------- batch_size-backward ------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |       7.1      |       10.8     
         2  |       6.9      |       10.5     
         4  |       7.4      |       11.3     
         8  |       7.7      |       12.1     
        16  |      10.6      |       15.1     
        32  |      14.9      |       22.9     
        64  |      27.7      |       43.1     
       128  |      58.3      |       99.9     
       256  |     119.7      |      218.8     
       512  |     240.2      |      435.8     
      1024  |     481.2      |      873.1     

Times are in milliseconds (ms).

[----------- num_features-forward -----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         4  |       1.9      |        2.8     
         8  |       3.5      |        7.6     
        16  |       8.0      |       22.8     
        32  |      20.3      |       53.1     
        64  |      22.7      |       53.7     
       128  |      26.8      |       56.6     
       256  |      34.7      |       62.7     
       512  |      53.7      |       74.2     
      1024  |      91.9      |      100.0     
      2048  |     167.3      |      146.2     
      4096  |     313.5      |      253.5     

Times are in milliseconds (ms).

[---------- num_features-backward -----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         4  |       4.3      |       12.5     
         8  |       8.6      |       27.7     
        16  |      18.7      |       65.5     
        32  |      43.2      |      143.7     
        64  |      47.8      |      145.5     
       128  |      57.4      |      155.0     
       256  |      77.8      |      177.4     
       512  |     119.6      |      218.2     
      1024  |     202.4      |      302.3     
      2048  |     370.9      |      472.1     
      4096  |     628.7      |      729.1     

Times are in milliseconds (ms).

[-------------- depth-forward ---------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      36.9      |       10.9     
         2  |      38.0      |       12.5     
         3  |      39.2      |       19.2     
         4  |      43.3      |       38.1     
         5  |      53.8      |       75.3     
         6  |      71.3      |      151.0     
         7  |     107.5      |      301.9     
         8  |     217.8      |        OOM     
         9  |     526.7      |        OOM     

Times are in milliseconds (ms).

[-------------- depth-backward --------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      82.9      |       55.7     
         2  |      84.7      |       63.8     
         3  |      89.4      |       83.6     
         4  |      97.7      |      129.6     
         5  |     120.4      |      220.0     
         6  |     158.3      |      401.5     
         7  |     237.9      |      765.7     

Times are in milliseconds (ms).

[------------- num_sums-forward -------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      49.1      |       50.9     
         2  |      50.0      |       52.5     
         4  |      49.8      |       52.9     
         8  |      50.1      |       53.0     
        16  |      50.7      |       54.9     
        32  |      53.6      |       74.4     
        64  |      65.9      |      139.9     
       128  |     156.5      |        OOM     

Times are in milliseconds (ms).

[------------ num_sums-backward -------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |     102.7      |      152.9     
         2  |     106.4      |      157.8     
         8  |     106.9      |      158.5     
        16  |     110.1      |      166.6     
        32  |     120.4      |      219.7     
        64  |     164.1      |      404.4     

Times are in milliseconds (ms).

[------------ num_leaves-forward ------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      10.1      |       23.5     
         2  |       6.4      |       24.1     
         4  |       8.0      |       25.5     
         8  |      14.4      |       28.7     
        16  |      26.0      |       38.7     
        32  |      53.2      |       75.2     
        64  |     130.1      |      181.6     
       128  |     363.7      |        OOM     

Times are in milliseconds (ms).

[----------- num_leaves-backward ------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      20.7      |       68.4     
         2  |      17.0      |       68.9     
         4  |      19.6      |       73.0     
         8  |      29.7      |       83.2     
        16  |      57.2      |      116.9     
        32  |     119.6      |      218.8     
        64  |     274.8      |      504.4     

Times are in milliseconds (ms).

[----------- num_channels-forward -----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      54.4      |       74.1     
         2  |      65.8      |       78.4     
         4  |      89.9      |       85.1     
         8  |     138.1      |       97.5     
        16  |     235.1      |      125.8     

Times are in milliseconds (ms).

[---------- num_channels-backward -----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |     120.5      |      219.8     
         2  |     175.1      |      249.2     
         4  |     288.4      |      303.8     
         8  |     452.0      |      391.3     

Times are in milliseconds (ms).

[--------- num_repetitions-forward ----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |       2.2      |        2.8     
         2  |       3.2      |        3.0     
         4  |       5.4      |        5.1     
         8  |      10.7      |       11.3     
        16  |      22.8      |       30.4     
        32  |      53.7      |       75.2     
        64  |     109.4      |      192.2     
       128  |     224.1      |      520.7     

Times are in milliseconds (ms).

[--------- num_repetitions-backward ---------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |       5.8      |        10.3    
         2  |       6.3      |        11.4    
         4  |       9.8      |        18.1    
         8  |      21.6      |        39.5    
        16  |      51.4      |        95.4    
        32  |     119.1      |       220.2    
        64  |     250.6      |       520.8    
       128  |     504.6      |      1316.4    

Times are in milliseconds (ms).

[----------- num_classes-forward ------------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |      53.9      |       75.1     
         2  |      53.9      |       75.0     
         4  |      54.0      |       75.1     
         8  |      54.1      |       75.5     
        16  |      53.5      |       75.5     
        32  |      54.0      |       75.2     
        64  |      53.7      |       74.7     
       128  |      54.3      |       75.6     

Times are in milliseconds (ms).

[----------- num_classes-backward -----------]
            |  simple-einet  |  EinsumNetworks
1 threads: -----------------------------------
         1  |     119.8      |      218.5     
         2  |     120.6      |      220.7     
         4  |     120.2      |      220.3     
         8  |     119.9      |      221.4     
        16  |     119.8      |      221.2     
        32  |     120.4      |      217.7     
        64  |     120.4      |      221.2     
       128  |     121.0      |      219.4     

Times are in milliseconds (ms).
```
