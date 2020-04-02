from train import create_arg_dict
from deep_rl_torch import Trainer

def setup_params(parameters):
    parameters["initial_steps"] = 100
    parameters["save_path"] = ".tmp"
    parameters["save_percentage"] = 0
    parameters["steps"] = 100
    
def run(env):
    parameters, env = create_arg_dict([], env=env)
    setup_params(parameters)
    trainer = Trainer(env, parameters, log=False, verbose=False)
    trainer.run(total_steps=parameters["steps"], render=False, verbose=False)
    
def test_cart():
    env = "CartPole-v1"
    run(env)
    
def test_pong():
    env = "Pong-v0"
    run(env)
