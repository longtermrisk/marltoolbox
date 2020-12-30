# CLR research framework
**Goal**: Facilitate and speed up the research on 
bargaining in multi-agent systems.  
**Philosophy**: Implement lazily (only when needed). 
Improve at each new use. Keep it simple.  

We rely on two components: 
- [`RLLib 1.0.0` (Ray / Tune / RLLib)](https://docs.ray.io/en/master/rllib.html) 
- A toolbox: `marltoolbox`.   
    This repository, which has a `master` branch and an `experimental` branch.


# Starting
`Tune` is built on top of `Ray` and 
`RLLib` is built on top of `Tune`.  
This toolbox (`marltoolbox`) is built to work with `RLLib` 
but also to allow to fallback to `Tune` (only) if needed, 
at the cost of some functionalities.   


### Quick intro to Tune
[Tune Key Concepts](https://docs.ray.io/en/master/tune/key-concepts.html)  

To explore further `Tune` please refer to the 
[examples provided by `Tune`](https://docs.ray.io/en/master/tune/examples/index.html#tune-general-examples)


### Quick intro to RLLib
[RLlib in 60 seconds](https://docs.ray.io/en/master/rllib.html).  
    
To explore further `RLLib` please refer to the 
[RLLib documentation](https://docs.ray.io/en/master/rllib-toc.html).
Finally, you also can find many 
[examples provided by `RLLib`](https://github.com/ray-project/ray/tree/master/rllib/examples). 

### Toolbox installation

```bash
# Create a virtual environment
conda create -n marltoolbox python=3.8.5
conda activate marltoolbox

# Install the toolbox: marltoolbox
conda install psutil
git clone https://github.com/Manuscrit/temp_research_framework.git
cd temp_research_framework
pip install -e .

# Check that RLLib is working
# Use RLLib built-in training fonctionnalities
rllib train --run=PPO --env=CartPole-v0 --torch 

# Check that the toolbox is working
python ./marltoolbox/examples/rllib_api/pg_ipd.py

# Visualize logs
tensorboard --logdir ~/ray_results
# If working on GCP: forward the connection from a Virtual Machine(VM) to your machine
gcloud compute ssh {replace-by-instance-name} --zone={replace-by-instance-zone} -- -NfL 6006:localhost:6006

# Commands below are optionals: 

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
pip install --upgrade pip
pip install tensorflow
```


## Training models with the framework

There are mainly 3 ways to run experiments. 
They support increasing functionalities 
but also use a more and more complex API:

**<ins>Tune function API</ins>** 
- With the Tune function API, you only need to provide the training 
function. [See the Tune documentation](https://docs.ray.io/en/master/tune/key-concepts.html).     
- Best used if you very quickly want to run some code from an external repository.
- **Functionalities:** Running several seeds in parallel and comparing their results. 
Easily plot values to Tensorboard (visualizing the plots in live). 
Tracking your experiments and hyperparameters. Hyperparameter search.  

**<ins>Tune class API</ins>**  
- You need to provide a Trainer class with at minimum a setup method and a 
step method. [See the Tune documentation](https://docs.ray.io/en/master/tune/key-concepts.html).    
- Best used if you want to run some code from an external repository 
and you need checkpoints. Helpers in this toolbox (`marltoolbox.utils.policy.get_tune_policy_class`)
 will also allow you to evaluate it against other RLLib algorithms or
 with some experimentation tools in `marltoolbox.utils`.
- **Additional functionalities:** Checkpoints.  
The trained agents can be converted to the RLLib Policy format for evaluation only.
This allows you to use functionalities which rely on the RLLib API. 

**<ins>RLLib Trainer class</ins>**  
- You need to use the RLLib Trainer and Policy classes APIs. 
The RLLib Trainer class is a specific implementation of the Tune class API 
(just above). [See the RLLib documentation](https://docs.ray.io/en/master/rllib-toc.html).  
- Best used if you are creating a new training setup or policy from scratch or 
if you need to train agents using both algorithms from 
RLLib and an external repository.  
- **Additional functionalities:** Using components from RLLib 
(models, environments, algorithms, exploration, etc.)


# Main content of the toolbox
- envs
    - various matrix social dilemmas
    - various coin games
- algos
    - amTFT [(approximate  Markov tit-for-tat)](https://arxiv.org/abs/1707.01068)
    - L-TFT (Learning Tit-for-tat or Learning Equilibrium, **simplified version**)
    - LOLA-DICE (unofficial, in the `experimental` branch, **WIP: not working properly**)  
    - supervised learning
    - hierarchical
        - Base Policy class which allow to use nested algorithms 
- utils  
    - exploration
        - SoftQ with temperature schedule
    - log
        - callbacks to log values from environments and policies
    - lvl1_best_response 
        - helper functions to train level 1 exploiters
    - policy
        - helper to transform a trained Tune Trainer
        into a frozen RLLib policy
    - postprocessing
        - helpers to compute welfare functions 
        and add this data in the evaluation batch
    - restore
        - helpers to load a checkpoint only for 
        a chosen policy
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
        **WIP: hyperparameters not well tunes**)
        - LOLA-DICE (official with slight modifications, 
        **WIP: not working properly**)
    - Tune class API
        - LOLA-PG (official with various improvements) 
        - L1BR LOLA-PG
        - LOLA-Exact (official) 
        - LOLA-DICE (official with slight modifications) 
    - RLLib API
        - Inequity aversion
        - L-TFT (Learning Tit-for-tat or Learning Equilibrium) 
        - amTFT
        - L1BR amTFT
        - LOLA-DICE (unofficial, in the `experimental` branch, **WIP: Not working properly**)


# Some usages:

#### 0) Fall back to the `Tune` APIs when using the `RLLib` API is too costly
You can find examples in `marltoolbox.examples.tune_class_api` and in `marltoolbox.examples.tune_function_api`.  

If the setup you want to train already exist and has a training loop 
and if the cost to convert it into `RLLib` is too expensive,
then with minimum changes you can use `Tune`.


**When is the conversion cost to `RLLib` too high?**  
- If the algorithm has a complex unusual dataflow 
- If the algorithm has an unusual training process 
    - like LOLA: performing virtual opponent updates
    - like L-TFT(LE): nested algorithms
- If you don't need to change the algorithm
- If you don't plan to run the algorithm against Policies from RLLib
- If you do not plan to work much with the algorithm. 
And thus, you do not want to invest time in the conversion to `RLLib`.
- Above points and you are only starting to use `RLLib`  
- etc.

#### 1) Using components directly provided by `RLLib`
Examples:
- APEX_DDPG and the water world environment:
[`multi_agent_independent_learning.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py)
- MADDPG and the two step game environment:
[`two_step_game.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py)
- Policy Gradient (PG) and the rock paper scissors environment (function `run_same_policy`):
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)

#### 2) Using custom `RLLib` environments
Examples:  
- IPD environment: pg_ipd.py (toolbox example)
- (Asymmetric) Coin Game: ppo_asymmetric_coin_game.py (toolbox example)

#### 3) Customizing existing algorithms from `RLLib`
Examples:  
- Customize policy's postprocessing (processing after env.step) and trainer:
inequity_aversion.py (toolbox example)
- Change the loss function of the Policy Gradient (PG) Policy:
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_with_custom_entropy_loss` function) 

#### 4) Using custom agents in `RLLib`
In RLLib, customizing a policy allows to change its training and evaluation logics.    

Examples:  
- Hardcoded random Policy:
[`multi_agent_custom_policy.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_custom_policy.py)
- Hardcoded fixed Policy:
[`rock_paper_scissors_multiagent.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_heuristic_vs_learned` function)
- Policy with nested Policies: `le_ipd.py` (toolbox example)

#### 5) Using custom dataflows in `RLLib` (custom Trainer or Trainer's execution_plan)
Examples:
- Training 2 different policies with 2 different Trainers 
(less complex but less sample efficient than the 2nd method below):
[`multi_agent_two_trainers.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py)
- Training 2 different policies with a custom Trainer (more complex, more sample efficient):
[`two_trainer_workflow.py`](https://github.com/ray-project/ray/blob/master/rllib/examples/two_trainer_workflow.py)

#### 6) Using experimentation tools from the toolbox
Examples:
- Training a level 1 best response: `l1br_amtft.py` (toolbox example)
- Evaluating same-play and cross-play performances: `amtft_various_env.py` (toolbox example)


# TODO / Wishlist
- See the google doc "Survey for MARL framework"
- See the google doc "MARL benchmark paper"
- add unit tests
- add a simple coin game env (not vectorized and more readable)
- check performance of coin game with and without reporting the additional info
- set good hyper-parameters for our custom examples 
    - like 2nd conv without padding for Coin Game?
- RLLib install helper
- report all results directly in Weights&Biases (saving download time from VM)
- MARL RLLib tutorial
- give some guidelines for this toolbox