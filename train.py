import argparse
import yaml
from ray import tune
from ray.tune import RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from agents.marl_wrapper import ConstructionParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

def env_creator(config):
    env = ConstructionParallelEnv(render=config.get("render", False), num_robots=config.get("num_robots", 4))
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    register_env("construction_v0", lambda cfg: env_creator(config["env_config"]))

    # Configure PPO (MAPPO) with Multi-Agent Policy
    ppo_config = (
        PPOConfig()
        .environment("construction_v0")
        .framework(config["training_config"]["framework"])
        .resources(num_cpus_for_main_process=1)
        .env_runners(num_env_runners=config["training_config"]["num_workers"])
        .training(
            train_batch_size=config["training_config"]["train_batch_size"],
            lr=config["training_config"]["lr"],
            gamma=config["training_config"]["gamma"],
        )
        .multi_agent(
            policies={f"robot_{i}": (None, None, None, {}) for i in range(config["env_config"]["num_robots"])},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )

    tune.Tuner(
        "PPO",
        run_config=RunConfig(stop={"training_iteration": 10}),
        param_space=ppo_config.to_dict(),
    ).fit()
