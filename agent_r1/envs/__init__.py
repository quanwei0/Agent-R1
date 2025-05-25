def _default_env(name):
    if name == "nous":
        from agent_r1.envs.nous import NousToolEnv
        return NousToolEnv
    elif name == "retool":
        from agent_r1.envs.retool import ReToolEnv
        return ReToolEnv
    else:
        raise NotImplementedError(f"Tool environment {name} is not implemented")