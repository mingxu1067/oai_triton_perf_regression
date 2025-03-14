# OAI Triton Performance Regression Reproducer

## Reproduce Steps
0. Please use [JAX-Toolbox Nightly Container at 25-03-10], ghcr.io/nvidia/jax:jax-2025-02-01

1. Clone and build [OAI Triton](https://github.com/triton-lang/triton)
```
$ git clone https://github.com/triton-lang/triton.git
$ cd triton
# checkout to the target commit if need
$ pip install ninja cmake wheel pybind11
$ pip install -e python 
```

2. Clone [this reproducer](https://github.com/mingxu1067/oai_triton_perf_regression)
```
$ git clone https://github.com/mingxu1067/oai_triton_perf_regression
```

3. Run `run.sh` and check the performance of `kernel_1`
```
$ bash run.sh
...
[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ------------------------
     95.5           872982         20   43649.1   43295.5     42463     46624       1021.1  kernel_1
...

```