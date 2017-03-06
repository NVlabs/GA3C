# GA3C: Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU

A hybrid CPU/GPU version of the Asynchronous Advantage Actor-Critic (A3C) algorithm, currently the state-of-the-art method in reinforcement learning for various gaming tasks. This CPU/GPU implementation, based on TensorFlow, achieves a significant speed up compared to a similar CPU implementation.

## How do I get set up? ###

* Install [Python > 3.0](https://www.python.org/)
* Install [TensorFlow 1.0](https://www.tensorflow.org/install/install_linux) 
* Install [OpenAI Gym](https://github.com/openai/gym)
* Clone the repo.
* That's it folks!

## How to Train a model from scratch? ###

Run `sh _clean.sh` first, and then `sh _train.sh`.
The script `_clean.sh` cleans the checkpoints folder, which contains the network models saved during the training process, as well as removing `results.txt`, which is a log of the scores achieved during training.

> Remember to save your trained models and scores in a different folder if needed before cleaning.

`_train.sh` launches the training procedure, following the parameters in `Config.py`.
You can modify the training parameters directly in `Config.py`, or pass them as argument to `_train.sh`.
E.g., launching `sh _train.sh LEARNING_RATE_START=0.001` overwrites the starting value of the learning rate in `Config.py` with the one passed as argument (see below).
You may want to modify `_train.sh` for your particular needs. 

The output should look like below:

...  
[Time:       33] [Episode:       26 Score:   -19.0000] [RScore:   -20.5000 RPPS:   822] [PPS:   823 TPS:   183] [NT:  2 NP:  2 NA: 32]  
[Time:       33] [Episode:       27 Score:   -20.0000] [RScore:   -20.4815 RPPS:   855] [PPS:   856 TPS:   183] [NT:  2 NP:  2 NA: 32]  
[Time:       35] [Episode:       28 Score:   -20.0000] [RScore:   -20.4643 RPPS:   854] [PPS:   855 TPS:   185] [NT:  2 NP:  2 NA: 32]  
[Time:       35] [Episode:       29 Score:   -19.0000] [RScore:   -20.4138 RPPS:   877] [PPS:   878 TPS:   185] [NT:  2 NP:  2 NA: 32]  
[Time:       36] [Episode:       30 Score:   -20.0000] [RScore:   -20.4000 RPPS:   899] [PPS:   900 TPS:   186] [NT:  2 NP:  2 NA: 32]  
...  

**PPS** (predictions per second) demonstrates the speed of processing frames, while **Score** shows the achieved score.  
**RPPS** and **RScore** are the rolling average of the above values.

To stop the training procedure, adjuts `EPISODES` in `Config.py` propoerly, or simply use ctrl + c.

## How to continue training a model? ###

If you want to continue training a model, set `LOAD_CHECKPOINTS=True` in `Config.py`, and set `LOAD_EPISODE` to the episode number you want to load.
Be sure that the corresponding model has been saved in the checkpoints folder (the model name includes the number of the episode).

> Be sure not to use `_clean.sh` if you want to stop and then continue training! 

## How to play a game with a trained agent? ###

Run `_play.sh`
You may want to modify this script for your particular needs.

## How to change the game, configurations, etc.? ###
All the configurations are in `Config.py`  
As mentioned before, one useful way of modifying a config is to pass it as an argument to `_train.sh`. For example, to save the models while training, just run: `train.sh TRAINERS=4`.

## Sample learning curves
Typical learning curves for Pong and Boxing are shown here. These are easily obtained from the results.txt file.
![Convergence Curves](http://mb2.web.engr.illinois.edu/images/pong_boxing.png)

### References ###

If you use this code, please refer to our [ICLR 2017 paper](https://openreview.net/forum?id=r1VGvBcxl):

```
@conference{babaeizadeh2017ga3c,
  title={Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU},
  author={Babaeizadeh, Mohammad and Frosio, Iuri and Tyree, Stephen and Clemons, Jason and Kautz, Jan},
  booktitle={ICLR},
  biurl={https://openreview.net/forum?id=r1VGvBcxl},
  year={2017}
}
```
This work was first presented in an oral talk at the [The 1st International Workshop on Efficient Methods for Deep Neural Networks](http://allenai.org/plato/emdnn/papers.html), NIPS Workshop, Barcelona (Spain), Dec. 9, 2016:

```
@article{babaeizadeh2016ga3c,
  title={{GA3C:} {GPU}-based {A3C} for Deep Reinforcement Learning},
  author={Babaeizadeh, Mohammad and Frosio, Iuri and Tyree, Stephen and Clemons, Jason and Kautz, Jan},
  journal={NIPS Workshop},
  biurl={arXiv preprint arXiv:1611.06256},
  year={2016}
}
```
