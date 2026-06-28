# Unified Stability Bounds for Structured World Models: Geometry, Equivariance, and Identifiability as Sufficient Conditions

This paper introduces a unified stability bound to evaluate representations in model-based reinforcement learning without retraining. It breaks down performance loss into three factors: geometric distortion, identifiability, and equivariance error. The authors provide a lightweight diagnostic tool to test these properties on existing models, offering a practical way to audit and interpret learned representations. 


# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+. The following commands can be copied and pasted directly into your terminal.

## 1. Clone the source code 
First, clone this repository to your local machine.


## 2. Create a virtual environment 
It's recommended to work in a dedicated virtual environment.
```
conda create -n spwm python=3.11 
conda activate spwm
```

## 3. Install dependencies 
Navigate into the cloned directory and install the required packages.
```sh
cd ./SPWM
pip install -U -r requirements.txt
```

Optional: To train on the Crafter environment, also install:

```sh
pip install crafter
```

## 4. Training the DreamerV3 model 
You can start training the model with the following command:
```sh
python dreamerv3/main.py \
  --logdir logdir/{timestamp} \
  --configs crafter \
  --run.train_ratio 256
```

You can modify the `--run.steps` argument to set the number of training steps. The paper uses a default of `500000` steps.

## 5. Reproduce the results 
First, install the additional packages needed for analysis:
```sh
pip install pandas torch torchvision torchaudio
```

Next, we provide a script to test the analysis pipeline. Run it with mock data to ensure everything is set up correctly. Remember to replace `Your own checkpoints path` with the actual path to your saved checkpoints:

```sh
python spwm_analysis/spwm_metrics_system.py --mode analyze --adapter mock --ckpt_dir ./logdir/Your own checkpoints path --scores_path ./logdir/Your own checkpoints path/scores.jsonl --eval_set_path ./spwm_analysis/eval_set_crafter.npz --output_dir ./results_demo --device cuda
```

Once the test passes, you can run the full analysis on your trained DreamerV3 model:
```sh
python spwm_analysis/spwm_metrics_system.py --mode analyze --adapter dreamer --ckpt_dir /hy-tmp/SPWM/logdir/crafter_5M_20250914_103133 --scores_path /hy-tmp/SPWM/logdir/crafter_5M_20250914_103133/scores.jsonl --eval_set_path /hy-tmp/SPWM/spwm_analysis/eval_set_crafter.npz --output_dir /hy-tmp/SPWM/results_real_run --device cuda
```



