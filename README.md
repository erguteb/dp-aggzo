# AE submission for USENIX Security 2025 Cycle 2 p186

This is for testing the funtionality and reproducing the performance (i.e., test accuracy) of DP-AggZO with differential privacy (DP) as constraints.

## Recommended Hardware Spec
Linux machine with Ubuntu 22.04.4 and RTX 4090 GPU 24GB, or above (larger GPU memory is needed if run larger models, e.g., OPT 6.7B).

Abstract: The main idea of our proposed method, DP-AggZO, is to aggregate *multiple zeroth-order estimates* for the exact gradients, computed over independent perturbation vectors (random Gaussian vectors), before enforcing differential privacy (i.e., artificial clipping, taking the average, and then injecting random DP noises). Compared with the vanilla DPZO (or DPZero), which is effectively a degenerated version of DP-AggZO with only *one* zeroth-order estimate, our DP-AggZO achieves much better utility under the same privacy constraints. Our DP-AggZO also outperforms the state-of-the-art DP-AdamW in some cases.

We refer to the number of random directions in DP-AggZO as `K` (`K>1` is an integer). DPZero/DPZO is effectively, DP-AggZO with `K=1`.

## Differential Privacy (DP)

We consider the classic (epsilon,delta)-DP framework, specified by two parameters, `eps` and `delta`. 
Smaller `eps` and `delta` correspond to more restrictive privacy,
and hence, lower utility. 

In our experiments, we fix `delta` to `1e-5`, with `eps` varying in `{0.5,1,2,6}`.

Our privacy analysis is based on the Opacus library for the [Subsampled Gaussian Mechanism](https://arxiv.org/pdf/1908.10530).

## Installation and Setup

Our implementation is based on prior work [DPZero](https://github.com/liang137/dpzero) and [MeZO](https://github.com/princeton-nlp/MeZO/tree/main/medium_models).

This code is tested on `python 3.9.18`, with
`torch==2.4.0+cu121`, `transformers==4.28.1`, and `opacus==1.4.0`.

More on enviroments can be found in `environments.yml`. You can also create one using commands below.
```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero
```

We focus on the RoBERTA(355M) and OPT model family (1.3B and 6.7B). 

We highly recommend to test on RoBERTA(355M), which takes less time than OPT.

In what follows, we will **focus on RoBERTa(355M)**.

## Experiments on RoBERTA(355M)
Go to folder roberta and install the dataset as follows. 

The datasets can be found [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Alternatively (if the server is down), try the following two links from [dropbox](https://www.dropbox.com/scl/fi/s4tf8m0t40k85mhsybaru/datasets.tar?rlkey=55mq7r1s2lvs1l30ut08pbxe5&st=5yjtt4ba&dl=0) or [Google drive](https://drive.google.com/file/d/1dq6MGyGyOdPTKLlhc1yEdr8tgjyuruxT/view?usp=sharing).

Please download it to `data/` folder, 
and then run the following commands to prepare the datasets 

(un-comment the first line to wget the dataset first if it is not downloaded beforehand):

```bash
cd data
bash prepare_datasets.sh
cd ..
```

## Functionality Test
This is for testing functionality of our implementation of DP-AggZO.

To **save the computation time** for DP-AggZO, you can use a batch size of 32 (in expectation) and 500 iterations.

Run the following in bash.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=2 DP_SAMPLE_RATE=0.0208 STEP=500 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=6e-4 DPZERO_THRESHOLD=1  TASK="MNLI" bash examples/dpaggzo.sh
```
If huggingface is not accessible in your region, try adding `HF_ENDPOINT=https://hf-mirror.com` to the very beginning of the above bash command.

This script takes around **80 minutes** to run on a RTX 4090 GPU and gives a test accuracy around `72%`, which is much better than the utility of the original DPZO/DPZero under the same privacy constraint (around`65%`, or refer to Table 2 on [Page 9](https://arxiv.org/pdf/2310.09639)). 

Results on other datasets can be obtained by changing `TASK` to `trec`, `SNLI`, `RTE`, and `sst-5` and `SST-2`. 

The privacy constraints is specified under (eps,delta)-DP with `delta=1e-5` by default, following prior work [DPZero](https://github.com/liang137/dpzero).
You can change the value of `eps` via `DPZERO_PRIVACY_EPS`, e.g.,
```
DPZERO_PRIVACY_EPS=2
```

`DP_SAMPLE_RATE` specifies the Poisson sampling rate in each iteration, e.g., 
```
DP_SAMPLE_RATE=0.0416
```

`STEP` specifies the total number of training steps/iterations, e.g., 
```
STEP=1000
```

`NUM_DIRECTION` specifies the number of random directions/perturbation vectors, (in the paper, we refer to it as `K`) in DP-AggZO, e.g.,
```
NUM_DIRECTION=64
```
We recommend using `NUM_DIRECTION=64` on RoBERTA for a good trade-off of privacy, utility, and computation cost.

`RANDOM_DIRECTION_SEED` specifies the random seed used for generating random directions.

`DPZERO_THRESHOLD` is the clipping norm for the `K`-dimensional aggregator vector.

`Seed` corresponds to the seed for obtaining the `512`-shot dataset. Seeds `13,21,42,87,100` are supported; the default seed is `42`.

`LR` is the learning rate. 

## Reproducibility
The key message in our paper is that **our DP-AggZO can outperform the vanilla DPZero/DPZO (i.e., DP-AggZO with `NUM_DIRECTION=1`) and sometimes DP-AdamW.**

In what follows, we provide some scripts to validate this. 

(Note that the results may be different on different GPUs)

To reproduce the results on dataset **MNLI** of our paper (Table 1 Page 11), 
you can run them directly with the following command.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=2 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=8e-5 DPZERO_THRESHOLD=5  TASK="MNLI" bash examples/dpaggzo.sh
```
or
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=2 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=5e-4 DPZERO_THRESHOLD=1  TASK="MNLI" bash examples/dpaggzo.sh
```
You will observe the result of DP-AggZO (`K = 64`) for `epsilon=2` (third row in Table 1). 
These scripts take around 5 hours to run on a RTX 4090 GPU, and 14 hours on a RTX A5000 GPU. 
The test accuracy may also vary -- on H20 and A5000 GPUs, the test accuracy is around `74%`; on 4090 GPU, the test accuracy is around `71%`.

Nevertheless, the utility should be much better than the original DPZero/DPZO (see Table 2 on [Page 9](https://arxiv.org/pdf/2310.09639)). You can also reproduce that result using the following script of DP-AggZO with `K=1`.
``` bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=2 DP_SAMPLE_RATE=0.0416 STEP=5000 SEED=42 NUM_DIRECTION=1 RANDOM_DIRECTION_SEED=100 LR=2e-6 DPZERO_THRESHOLD=200  TASK="MNLI" bash examples/dpaggzo.sh
```
This should give you a much lower test accuracy (only around `65%` on 4090 GPU). 
This script takes around 40 minutes to run on a RTX 4090 GPU. 

Recall that with around 80 minutes, we can run DP-AggZO (`K=64`) to get a test accuracy around `72%` (refer to the very beginning script). So we believe paying twice the computation time is worth the utility improvement (under the same privacy constraint).

To test DP-AggZO (`K = 64`) for for `epsilon=6` (seven-th row in Table 1), run the following.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=3.2e-5 DPZERO_THRESHOLD=25  TASK="MNLI" bash examples/dpaggzo.sh
```
The test accuracy is around `76%` on A5000 and H20 GPUs.

You can compare the utility with DPZero/DPZO (i.e., DP-AggZO with `K=1`) using the following.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0416 STEP=5000 SEED=42 NUM_DIRECTION=1 RANDOM_DIRECTION_SEED=100 LR=2e-6 DPZERO_THRESHOLD=200  TASK="MNLI" bash examples/dpaggzo.sh
```
Similarly, the utility (should be below `70%`) is much worse than DP-AggZO.

### More examples for RoBERTA(355M)

**MNLI** with `eps=0.5` (i.e., the small epsilon regime, Table 4 in Page 13)

The following script is for DP-AggZO.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=0.5 DP_SAMPLE_RATE=0.0416 STEP=500 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=2e-4 DPZERO_THRESHOLD=1  TASK="MNLI" bash examples/dpaggzo.sh
```
You can verify that the above result utility (around `63.5%` on H20 and A5000 GPUs) is better than the following that runs DP-AdamW.
```bash
CUDA_VISIBLE_DEVICES=0 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 LR=1e-4 DPSGD_THRESHOLD=10 DPSGD_PRIVACY_EPS=0.5 DPSGD_PRIVACY_DELTA=1e-5 TASK="MNLI" bash examples/dpsgd.sh
```
The result utility should be around `62%`, aligning with Table 4 in Page 13, and showing that DP-AggZO sometimes outperforms DP-AdamW.

With `eps=1`, the result utility for DP-AggZO (`K=64`) is can be higher (around `69%` on A5000 GPU).

```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=1 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=4e-5 DPZERO_THRESHOLD=5  TASK="MNLI" bash examples/dpaggzo.sh
```

### More examples for other tasks.

**RTE** with `eps=6` (Table 1 in Page 11)
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0625 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=4e-5 DPZERO_THRESHOLD=15  TASK="RTE" bash examples/dpaggzo.sh
```
should give around `75%` test accuracy.

As a comparison, you can run the original DPZero/DPZO using the following script. The utility should be much worse.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0625 STEP=1000 SEED=42 NUM_DIRECTION=1 RANDOM_DIRECTION_SEED=100 LR=1e-6 DPZERO_THRESHOLD=200  TASK="RTE" bash examples/dpaggzo.sh
```
should give around `61%` test accracy.

**More examples...**

**TREC** with `eps=6` (Table 1 in Page 11)

```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0242 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=3.2e-5 DPZERO_THRESHOLD=25  TASK="trec" bash examples/dpaggzo.sh
```
should give around `93%` test accuracy.

**SST-5** with `eps=6` (Table 1 in Page 11)

```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.025 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=1e-5 DPZERO_THRESHOLD=25  TASK="sst-5" bash examples/dpaggzo.sh
```
should give around `51.3%` test accuracy.

You can try larger `K`, i.e., larger `NUM_DIRECTION` to increase the privacy-utility trade-off, at the cost of more computation. 
E.g., to test DP-AggZO (`K = 256`) for `epsilon=6` (eight-th row in Table 1), run the following.
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=256 RANDOM_DIRECTION_SEED=100 LR=1e-4 DPZERO_THRESHOLD=2.5  TASK="MNLI" bash examples/dpaggzo.sh
```
You can even try `K=512`
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=512 RANDOM_DIRECTION_SEED=100 LR=2e-4 DPZERO_THRESHOLD=1  TASK="MNLI" bash examples/dpaggzo.sh
```
and with a smaller clipping threshold
```bash
CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0416 STEP=1000 SEED=42 NUM_DIRECTION=512 RANDOM_DIRECTION_SEED=100 LR=2e-3 DPZERO_THRESHOLD=0.1  TASK="MNLI" bash examples/dpaggzo.sh
```
You will notice a slight increase in the utility (compared with `K=64`), which does not worth the computation overhead.

Feel free to modify the parameters. 

In general, we recommend 1) smaller clipping threshold (i.e., `DPZERO_THRESHOLD`) and 2) larger learning rate (i.e., `LR`) for larger `K`; and 1) smaller clipping threshold (otherwise the additive Gaussian noise can be overwhelming) and 2) perhaps fewer iterations (i.e., `STEP=500` for `epsilon<1`) for smaller `epsilon`. 

### Minor note on Poisson sampling
When using Poisson sampling the batch size may be 0 when sampling rate is low, causing runtime errors.

We provide an ad-hoc fix.

add `if len(features)==0: return None`

to Line 305 of file ~/anaconda3/envs/env_name/lib/python3.9/site-packages/transformers/data/data_collator.py

and add `if not features: return {}`

to Line 225 of file /src/utils.py the `call` function of `DataCollatorWithPaddingAndNesting`

## Experiments on OPT models (Table 2 in Page 11)
Go to opt folder and run the following commands on bash (no need to download the dataset this time).

For **OPT-1.3B** and SQuAD dataset, for DP-AggZO (`K=16`) under `eps=2`, please run
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-1.3b TASK=SQuAD MODE=ft LR=1e-5 EPS=1e-3 EVAL_STEPS=250 DP_SAMPLE_RATE=0.064 DP_EPS=2.0 STEPS=1000 N=16 DP_CLIP=7.5 bash examples/dpaggzo.sh
```
For DP-AggZO (`K=64`) under `eps=2`, please run
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-1.3b TASK=SQuAD MODE=ft LR=6e-5 EPS=1e-3 EVAL_STEPS=250 DP_SAMPLE_RATE=0.064 DP_EPS=2.0 STEPS=1000 N=64 DP_CLIP=2 bash examples/dpaggzo.sh
```
You will be likely to observe test accuracies around `76%` (for `K=16`) and `78%` (for `K=64`); much better than the test accuracy for DPZero/DPZO (around `72%`).

Also, feel free to change from `opt-1.3b` to `opt-6.7b` if you have a large GPU. You will observe better utility, at the cost of more memory and longer computation time. You can try a different task by setting `TASK=your_task`.

### Memory consumption of DP-AggZO
Finally, feel free to change `NUM_DIRECTIONS` but you will notice that the memory consumed is roughly the same. As we have claimed, DP-AggZO maintains the memory consumption as DPZero. 

Thanks!
