import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, yaml

def load_cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)

def test_terrain_generator():
    print("\n-- Terrain generator")
    import pybullet as p, pybullet_data
    from envs.terrain_generator import TerrainGenerator, TERRAIN_REGISTRY
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    gen = TerrainGenerator(client)
    for name in TERRAIN_REGISTRY:
        tid = gen.load(name, seed=42)
        fv = gen.as_feature_vector()
        print(f"  {name:12s}  id={tid}  fv={fv.round(3)}")
    gen.load("rock", seed=7)
    probes = gen.sample_probe_heights(0.0, 0.0, num_probes=16)
    assert probes.shape == (16,)
    p.disconnect(client)
    print("  terrain generator: OK")

def test_env_basic():
    print("\n-- A1 environment")
    from envs.a1_env import A1Env
    cfg = load_cfg()
    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])
    assert env.observation_space.shape == (49,)
    assert env.action_space.shape == (12,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (49,) and not np.any(np.isnan(obs))
    print(f"  reset obs shape={obs.shape} mean={obs.mean():.4f}  OK")
    obs, reward, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
    print(f"  step(zeros) reward={reward:.4f} terminated={term}  OK")
    env.reset()
    total = 0.0
    for i in range(100):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc: break
    print(f"  100-step episode: steps={i+1} total_reward={total:.3f}  OK")
    env.close()
    print("  A1 env: OK")

def test_terrain_switch():
    print("\n-- Terrain switching")
    from envs.a1_env import A1Env
    cfg = load_cfg()
    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])
    for t in ["flat","sand","ice","rock","regolith"]:
        obs, _ = env.reset(terrain_name=t, seed=42)
        fv = env.get_terrain_feature_vector()
        print(f"  {t:12s}  fv={fv.round(2)}  OK")
    env.close()
    print("  terrain switching: OK")

def test_obs_sanity():
    print("\n-- Observation sanity")
    from envs.a1_env import A1Env, DEFAULT_JOINT_ANGLES
    cfg = load_cfg()
    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])
    obs, _ = env.reset(seed=0)
    gravity = obs[30:33]
    probes  = obs[33:49]
    assert np.allclose(obs[0:12], DEFAULT_JOINT_ANGLES, atol=0.2)
    assert abs(np.linalg.norm(gravity) - 1.0) < 0.1
    assert np.all(np.abs(probes) < 0.5)
    print(f"  gravity={gravity.round(3)}  probes_max={np.abs(probes).max():.4f}  OK")
    env.close()
    print("  obs sanity: OK")

if __name__ == "__main__":
    print("="*60)
    print("  latent-terrain-locomotion — environment tests")
    print("="*60)
    try:
        test_terrain_generator()
        test_env_basic()
        test_terrain_switch()
        test_obs_sanity()
        print("\n" + "="*60)
        print("  ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
