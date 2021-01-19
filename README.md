# CLR research framework
## Overview
**Goal**: Facilitate and speed up the research on bargaining in MARL. 

**Philosophy**: Implement lazily.
Improve at each new use. Keep it simple. Keep it flexible.  

**Support**: We will actively support researchers by adding tools that they see relevant for research on bargaining 
in MARL.  

**Concrete value of this framework**:  
- **with 1h of practice:**   
  track your experiments, 
  log easily in TensorBoard, run hyperparameter search, 
  use the provided environments and run the provided algorithms, 
  mostly agnostic to the deep learning framework, 
  create custom algorithms using the `Tune` API 
- **with 10h of practice:**  
  use some of the components of `RLLib` 
  (like using a PPO agent in your custom algorithms), use checkpoints, 
  use the experimentation tools provided here, create new environments, 
  create simple custom algorithm with the `RLLib` API
- **with more than 10h of practice:**  
  build custom distributed algorithms,
  use all of the components of `RLLib`, 
  use the fully customizable training pipeline of `RLLib`,
  create complex custom algorithm with the `RLLib` API  
  
**Components:** We rely on two main components: 
- The `Ray` framework: specifically [`Tune` and `RLLib` v1.0.0](https://docs.ray.io/en/master/rllib.html) 
- A toolbox: `marltoolbox` this repository. Which has a `master` branch and 
    an `experimental` branch with more experimental elements.


# Get started
`Tune` is built on top of `Ray` and 
`RLLib` is built on top of `Tune`.  
This toolbox `marltoolbox` is built to work with `RLLib` 
but also to allow to fallback to `Tune` only if needed, 
at the cost of some functionalities.   


### Quick intro to Tune
You should read this quick introduction to 
[Tune's key concepts](https://docs.ray.io/en/master/tune/key-concepts.html)  

Later, to explore `Tune` further, please refer to the 
[user guide](https://docs.ray.io/en/latest/tune/user-guide.html) and the
[examples provided by `Tune`](https://docs.ray.io/en/master/tune/examples/index.html#tune-general-examples)


### Quick intro to RLLib
You should read this quick introduction to 
[RLlib](https://docs.ray.io/en/master/rllib.html).  
    
And later too, to explore `RLLib` further, please refer to the 
[`RLLib` documentation](https://docs.ray.io/en/master/rllib-toc.html).
Finally, you also can find many 
[examples provided by `RLLib`](https://github.com/ray-project/ray/tree/master/rllib/examples). 

### Toolbox installation

The installation is tested with Ubuntu 18.04 LTS (preferred) and 20.04 LTS.  
It requires less than 20 Go of space including all the dependencies like PyTorch, etc.

**Optional**: Connect to your virtual machine(VM) on Google Cloud Platform(GCP) with:
```bash
gcloud compute ssh {replace-by-instance-name}
```

**Optional**: Do some basic upgrade and install some basic requirements (e.g. needed on a new VM)
```bash
sudo apt update
sudo apt upgrade
sudo apt-get install build-essential
# Run this command another time (especially needed with Ubuntu 20.04 LTS)
sudo apt-get install build-essential
```

**Optional**: use a virtual environment
```bash
# If needed, install conda:
## Follow instruction at
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
## Like that:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	# Enter. Enter... yes. Enter. yes.
	exit  
	# Connect again to the VM or open a new terminal
        gcloud compute ssh {replace-by-instance-name} 
	# Check your conda installation  
	conda list

# Create a virtual environment:
conda create -n marltoolbox python=3.8.5
conda activate marltoolbox
pip install --upgrade pip
```

**Install the toolbox: `marltoolbox`**
```bash
## Install dependencies
### For RLLib
conda install psutil
### (optional) To be able to use most of the gym environments
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
## Install marltoolbox
git clone https://github.com/longtermrisk/marltoolbox.git
## Here, you may need to provide github authentication (email and password)
cd marltoolbox
pip install -e .
```
**Test the installation**
```bash
# Check that RLLib is working
## Use RLLib built-in training functionalities
rllib train --run=PPO --env=CartPole-v0 --torch 
## Ctrl+C to stop the training 

# Check that the toolbox is working
python ./marltoolbox/examples/rllib_api/pg_ipd.py
## You should get the status TERMINATED

# Visualize the logs
tensorboard --logdir ~/ray_results
## If working on GCP: forward the connection from a Virtual Machine(VM) to your machine
## Run this command on your local machine from another terminal (not in the VM)
gcloud compute ssh {replace-by-instance-name} -- -NfL 6006:localhost:6006
## Go to your browser to visualize the url http://localhost:6006/
```
**Optional**: Install additional deep learning libraries (PyTorch CPU only is installed by default)
```bash
# Install PyTorch with GPU
# Check cuda version
nvidia-smi
# Look for "CUDA Version: XX.X"
# With the right cuda version:
conda install pytorch torchvision cudatoolkit=[cuda version like 10.2] -c pytorch
# Check PyTorch installation and if your GPU is available to PyTorch
python
    import torch
    torch.__version__
    torch.cuda.is_available()
    exit()

# Install Tensorflow
pip install tensorflow
```


## Training models with the framework

You can use the provided environments, policies and some components 
of `marltoolbox` and `RLLib` (like a PPO agent) 
anywhere (e.g. without `Tune` nor `RLLib`).  

Yet we recommend to use `Tune` and if possible `RLLib`. 
There are mainly 3 ways to run experiments with this framework. 
They support increasing functionalities 
but also use a more and more constrained API. 
 

**<ins>Tune function API</ins>** (the less constrained, not recommended) 
- **Constraints:** With the `Tune` function API, you only need to provide the training 
  function. [See the `Tune` documentation](https://docs.ray.io/en/master/tune/key-concepts.html).     
- **Best used:** If you want to very quickly run some code from an external repository.
- **Functionalities:** Running several seeds in parallel and comparing their results. 
  Easily plot values to TensorBoard and visualizing the plots in live. 
  Tracking your experiments and hyperparameters. Hyperparameter search.

**<ins>Tune class API</ins>** (very few constraints, recommended)  
- **Constraints:** You need to provide a Trainer class with at minimum a setup method and a 
  step method. [See the `Tune` documentation](https://docs.ray.io/en/master/tune/key-concepts.html).    
- **Best used:** If you want to run some code from an external repository 
  and you need checkpoints. Helpers in this toolbox (`marltoolbox.utils.policy.get_tune_policy_class`)
  will also allow you transform this class (already trained) into frozen `RLLib` policies. 
  This is useful to produce evaluation against other `RLLib` algorithms or
  when using experimentation tools in `marltoolbox.utils`.
- **<ins>Additional</ins> functionalities:** Checkpoints. Allow conversion to the `RLLib` policy API.   
  The trained agents can be converted to the `RLLib` policy API for evaluation only.
  This allows you to use functionalities which rely on the `RLLib` API (but not training). 

**<ins>RLLib API</ins>** (quite constrained, recommended)  
- **Constraints:** You need to use the `RLLib` API (trainer, policy, callbacks, etc.). 
  For information, `RLLib` trainer classes are specific implementations of the `Tune` class API 
  (just above). [See the `RLLib` documentation](https://docs.ray.io/en/master/rllib-toc.html).  
- **Best used:** If you are creating a new training setup or policy from 
  scratch, and you don't want to manage boilerplate code. 
  Or you want a seamless integration with `RLLib` components. 
  Or if you need distributed training.  
- **<ins>Additional</ins> functionalities:** Easily using all components from `RLLib` 
  (models, environments, algorithms, exploration, schedulers, preprocessing, etc.).
  Using the customizable trainer and policy factories from `RLLib`.


# Main content of the toolbox
- envs
    - various matrix social dilemmas
    - various coin games
- algos
    - AMD ([Adaptive Mechanism Design](https://arxiv.org/abs/1806.04067))
    - amTFT ([Approximate Markov Tit-For-Tat](https://arxiv.org/abs/1707.01068))
    - L-TFT ([Learning Tit-For-Tat](https://longtermrisk.org/files/toward_cooperation_learning_games_oct_2020.pdf), 
      **simplified version**)
    - LOLA-DICE ([paper](https://arxiv.org/pdf/1802.05098.pdf), 
      [unofficial](https://github.com/alexis-jacq/LOLA_DiCE), 
      in the `experimental` branch, **WIP: 
      not working properly**)  
    - supervised learning
    - hierarchical
        - This is a base policy class which allows to use nested algorithms 
- utils  
    - exploration
        - SoftQ with temperature schedule
    - log
        - callbacks to log values from environments and policies
    - lvl1_best_response 
        - helper functions to train level 1 exploiters
    - policy
        - helper to transform a trained Tune Trainer
        into frozen RLLib policies
    - postprocessing
        - helpers to compute welfare functions 
        and add this data in the evaluation batch 
        (the batches sampled by the evaluation workers)
    - restore
        - helpers to load a checkpoint only for 
        a chosen policy (instead of for all existing policies as RLLib does) 
    - rollout
        - a rollout runner function which can be called from 
        inside a RLLib policy
    - same_and_cross_perf
        - a helper to evaluate the performance
        in same and cross-play.    
        "same-play": playing against agents from the same training run.  
        "cross-play": playing against agents from different training runs.  
- examples
    - Tune function API
        - LOLA-PG (official with slight modifications, 
        **WIP: hyperparameters not well tuned**)
        - LOLA-DICE (official with slight modifications, 
        **WIP: not working properly**)
    - Tune class API
        - LOLA-PG (official with various improvements) 
        - L1BR LOLA-PG
        - LOLA-Exact (official) 
        - LOLA-DICE (official with slight modifications)
        - AMD (official with various improvements)
        - AMD using RLLib agents (official with various improvements)
    - RLLib API
        - Inequity aversion
        - L-TFT (Learning Tit-for-tat) 
        - amTFT
        - L1BR amTFT
        - LOLA-DICE (unofficial, in the `experimental` branch, **WIP: Not working properly**)


# Some usages:

#### 0) Fall back to the `Tune` APIs when using the `RLLib` API is too costly
You can find examples in `marltoolbox.examples.tune_class_api` and in `marltoolbox.examples.tune_function_api`.  

If the setup you want to train already exist, has a training loop 
and if the cost to convert it into `RLLib` is too expensive,
then with minimum changes you can use `Tune`.


**When is the conversion cost to `RLLib` too high?**  
- If the algorithm has a complex unusual dataflow 
- If the algorithm has an unusual training process 
    - like LOLA: performing "virtual" opponent updates
    - like L-TFT: nested algorithms
- If you don't need to change the algorithm
- If you don't plan to run the algorithm against policies from `RLLib`
- If you do not plan to work much with the algorithm. 
And thus, you do not want to invest time in the conversion to `RLLib`.
- Some points above and you are only starting to use `RLLib`  
- etc.

#### 1) Using components directly provided by `RLLib`  

Examples using the `Tune` class API:
- Using an A3C policy: amd_using_rllib_agents.py (toolbox example)
- Using (custom or not) environments: 
    - IPD environment: pg_ipd.py (toolbox example)
    - Asymmetric coin game: ppo_asymmetric_coin_game.py (toolbox example)

Examples using the `RLLib` API:
- APEX_DDPG and the water world environment:
[`multi_agent_independent_learning.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py)
- MADDPG and the two step game environment:
[`two_step_game.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py)
- Policy Gradient (PG) and the rock paper scissors environment:
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)
(in the `run_same_policy` function)

#### 2) Customizing existing algorithms from `RLLib`
Examples:  
- Customize policy's postprocessing (processing after env.step) and trainer:
inequity_aversion.py (toolbox example)
- Change the loss function of the Policy Gradient (PG) Policy:
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_with_custom_entropy_loss` function) 

#### 3) Creating and using new custom policies in `RLLib`
In RLLib, customizing a policy allows to change its training and evaluation logics.    

Examples:  
- Hardcoded random Policy:
[`multi_agent_custom_policy.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_custom_policy.py)
- Hardcoded fixed Policy:
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_heuristic_vs_learned` function)
- Policy with nested Policies: `ltft_ipd.py` (toolbox example)

#### 4) Using custom dataflows in `RLLib` (custom Trainer or Trainer's execution_plan)
Examples:
- Training 2 different policies with 2 different Trainers 
(less complex but less sample efficient than the 2nd method below):
[`multi_agent_two_trainers.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py)
- Training 2 different policies with a custom Trainer (more complex, more sample efficient):
[`two_trainer_workflow.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/two_trainer_workflow.py)

#### 5) Using experimentation tools from the toolbox
Examples:
- Training a level 1 best response: `l1br_amtft.py` (toolbox example)
- Evaluating same-play and cross-play performances: `amtft_various_env.py` (toolbox example)


# TODO / Wishlist
- Improvements:
    - Add more unit tests
    - Use the logger everywhere
    - Add and improve docstrings
    - Check performance of coin game with and without reporting the additional info
    - Set good hyper-parameters in the custom examples 
    - Report all results directly in Weights&Biases (saving download time from VM)
- Add new algorithms:
    - Multi-agent adversarial IRL
    - Multi-agent generative adversarial imitation learning
    - Model-based RL like PETS, MPC
    - Opponent modeling like k-level
    - Capability to use algorithms from OpenSpiel like MCTS
- Add new functionalities:
    - Reward uncertainty
    - Full / partial observability of opponent actions
    - (partial) Parameter transparency
    - Easy benchmarking with metrics specific to MARL
    - (more on) Exploitability evaluation 
    - Performance against a suite of other MARL algorithms
- Add new environments:
    - Capability to use environments from OpenSpiel
    - (iterated) Ultimatum game (including variants)
