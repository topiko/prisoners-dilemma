from env_utils.env import get_pd_env

if __name__ == "__main__":
    env = get_pd_env(max_cycles=10, render_mode="human").reset()
    print(env)
