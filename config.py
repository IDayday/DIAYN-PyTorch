import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="HalfCheetah-v3", type=str, help="Name of the environment. \
                                BipedalWalker-v3/Hopper-v3/HalfCheetah-v3/MountainCarContinuous-v0")
    parser.add_argument("--interval", default=20, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--mem_size", default=int(1e+6), type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=20, type=int, help="The number of skills to learn.")
    parser.add_argument("--num_processes", default=10, type=int, help="The number of multiprocessing.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--forward", default='CPU', type=str, help="forward on CPU or GPU when sampling")    
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")

    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {"lr": 3e-4,
                      "batch_size": 256,
                      "max_n_episodes": 5000,
                      "max_episode_len": 1000,
                      "warmup_size": 1000,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 300
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
