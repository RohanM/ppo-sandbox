import pytest
import gym

@pytest.fixture
def env_v0():
    env = gym.make('SimpleEnv-v0')
    env.reset()
    return env

@pytest.fixture
def env_v1():
    env = gym.make('SimpleEnv-v1')
    return env

def test_simple_env_v0_differing(env_v0):
    for i in range(5):
        obs, reward, terminated, truncated, info = env_v0.step(i % 2)
        assert reward == -1

def test_simple_env_v0_same(env_v0):
    for i in range(4):
        obs, reward, terminated, truncated, info = env_v0.step(0)
        assert reward == -1
    obs, reward, terminated, truncated, info = env_v0.step(0)
    assert reward == 20

def test_simple_env_v1_copy(env_v1):
    obs, info = env_v1.reset()
    for i in range(5):
        obs, reward, terminated, truncated, info = env_v1.step(obs)
        assert reward == 1

def test_simple_env_v1_diff(env_v1):
    obs, info = env_v1.reset()
    move = (obs + 1) % 5
    for i in range(50):
        obs, reward, terminated, truncated, info = env_v1.step(move)
        assert reward == -1
        assert not terminated

    obs, reward, terminated, truncated, info = env_v1.step(move)
    assert reward == -1
    assert terminated
