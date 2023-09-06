import warnings
import torch as th
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import time
from concurrent.futures import ThreadPoolExecutor

#import multiprocessing as mp
#from multiprocessing import shared_memory    
#from multiprocessing import Process, Manager
import torch.multiprocessing as torch_mp
from torch.multiprocessing import Queue as TorchQueue
import multiprocessing as mp
from multiprocessing import shared_memory

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
debug = False 
def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: th.device) :
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def worker_collect_rollouts(policy, rollout_buffer, 
                            n_rollout_steps, device, action_space,
                            gamma, _last_episode_starts, env, _last_obs):
    # Switch to eval mode (this affects batch norm / dropout)
    policy.set_training_mode(False)
    _last_obs = _flatten_obs([_last_obs], env.observation_space)

    # Used to verify model is same with the main process
    # if env.rank == 0:
    #     for param in policy.parameters():
    #         print(param.data.sum())


    n_steps = 0
    #rollout_buffer.reset()

    sum_of_cal_time = 0
    sum_of_step_time = 0
    sum_of_left_time = 0
    sum_of_buffer_time = 0 

    while n_steps < n_rollout_steps:
        start_time = time.time()

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(_last_obs, device)
            actions, values, log_probs = policy(obs_tensor)
        actions = actions.cpu().numpy()
        predit_time = time.time()
        sum_of_cal_time += int((predit_time - start_time) * 1000)

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(action_space, spaces.Box):
            clipped_actions = np.clip(actions, action_space.low, action_space.high)
        

        start_time = time.time()
        #new_obs, rewards, dones, infos, _ = env.step(clipped_actions)
        new_obs, rewards, terminated, truncated, infos = env.step(clipped_actions)
        new_obs = _flatten_obs([new_obs], env.observation_space)

        dones = terminated or truncated
        infos["TimeLimit.truncated"] = truncated and not terminated
        if dones:
            # save final observation where user can get it, then reset
            infos["terminal_observation"] = new_obs 

        step_time = time.time()
        sum_of_step_time += int((step_time - start_time) * 1000)

        # Give access to local variables
        n_steps += 1


        if isinstance(action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        if (
            dones
            and infos.get("terminal_observation") is not None
            and infos.get("TimeLimit.truncated", False)
        ):
            terminal_obs = policy.obs_to_tensor(infos["terminal_observation"])[0]
            with th.no_grad():
                terminal_value = policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
            rewards += gamma * terminal_value

        update_buffer_time = time.time()
        sum_of_buffer_time += (update_buffer_time - step_time) * 1000

        rollout_buffer.add(
            _last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            _last_episode_starts,  # type: ignore[arg-type]
            values,
            log_probs,
            pos = n_steps - 1, 
            rank = env.rank,
        )
        _last_obs = new_obs  # type: ignore[assignment]
        _last_episode_starts = dones
        left_time = time.time()
        sum_of_left_time += (left_time - update_buffer_time) * 1000

    with th.no_grad():
        # Compute value for the last timestep
        values = policy.predict_values(obs_as_tensor(new_obs, device))  # type: ignore[arg-type]
    
    #print("Sum of time", sum_of_cal_time / 1000, sum_of_step_time/ 1000, sum_of_buffer_time / 1000, sum_of_left_time / 1000)
    #print("Sum of time", sum_of_cal_time / 1000  + sum_of_step_time/ 1000 + sum_of_buffer_time / 1000 + sum_of_left_time / 1000)

    #rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    if dones:
        env.reset()

    return values, dones

def _episode_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    torch_queue: TorchQueue,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    sum_of_step_time = 0
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                start_time = time.time()
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                send_time = time.time()
                sum_of_step_time += send_time - start_time
                remote.send((observation, reward, done, info, reset_info, send_time, sum_of_step_time))
            elif cmd == 'set_rollout_buffer':
                create_rf, shm_name = data
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                remote.send("")
            elif cmd == 'set_model':
                policy = torch_queue.get()

                if not debug:
                    rollout_buffer = create_rf.create(existing_shm.buf)
                else:
                    rollout_buffer = create_rf 
                    rollout_buffer.reset(existing_shm.buf)

                remote.send("Done")
            elif cmd == "rollbuf":
                start_time = time.time()
                n_rollout_steps,\
                device, action_space, gamma, _last_episode_starts = data
                last_values = worker_collect_rollouts(policy, rollout_buffer, 
                            n_rollout_steps, device, action_space,
                            gamma, _last_episode_starts, env,
                            _last_obs)
                work_time = time.time()
                print(rollout_buffer.rewards[:,env.rank].sum())
                #
                #data_bytes = _ForkingPickler.dumps(rollout_buffer)
                #shm = shared_memory.SharedMemory(create=True, size=data_bytes.nbytes)
                #print(data_bytes.nbytes)
                #shm.buf[:data_bytes.nbytes] = data_bytes
                print(f"Sum of time in rank {env.rank} : {time.time() - start_time}\n , wrok_time : {work_time - start_time}")
                if time.time() - start_time > 40:
                    with open(f"{env.rank}.timeout", "a") as fp:
                        fp.write(f"{env.current_time} : cost time {time.time() - start_time}\n")

                remote.send(last_values)
            elif cmd == "reset":
                _last_obs, _ = env.reset()
                remote.send("")
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    sum_of_step_time = 0
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                start_time = time.time()
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                send_time = time.time()
                sum_of_step_time += send_time - start_time
                remote.send((observation, reward, done, info, reset_info, send_time, sum_of_step_time))
            elif cmd == "reset":
                observation, reset_info = env.reset(seed=data)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self.thread_pool = ThreadPoolExecutor(max_workers=12)
        n_envs = len(env_fns)
        self.max_gap_list = []

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        th.multiprocessing.set_start_method(start_method)
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.torch_queues = [TorchQueue() for _ in range(n_envs)]
        self.processes = []
        for work_remote, remote, torch_queue, env_fn in zip(self.work_remotes, self.remotes, self.torch_queues, env_fns):
            args = (work_remote, remote, torch_queue, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            # pytype: disable=attribute-error
            #process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process = ctx.Process(target=_episode_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            # pytype: enable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        #task_list = [] 
        #start_time = time.time()
        results = [(remote.recv(), time.time()) for remote in self.remotes]
        finished_time = time.time()
        time_list = [item[0][5] for item in results]
        self.max_gap_list.append(max(time_list) - min(time_list))
        step_time = [item[0][6] for item in results]
        #print("max gap is", 1000 * (max(time_list) - min(time_list)))
        #for item in results:
        #    print(1000 * (item[1] - item[0][5]), 1000 * (finished_time - item[1]))
        results = [item[0][:5] for item in results]
        #result = self.remotes[0].recv()
        #results = [result] * len(self.remotes)
        #print("recv cost", time.time() - start_time)
        #for i in range(1, len(self.remotes)):
        #    self.thread_pool.submit(self.remotes[i].recv)
        #print("Finished")
        #results = []
        #for task in task_list:
        #    results.append(task.result())
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        if dones[0]:
            print("sum of gap list", sum(self.max_gap_list))
            print("step time list is ", step_time)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", self._seeds[env_idx]))
        results = [remote.recv() for remote in self.remotes]
        #obs, self.reset_infos = zip(*results)
        # Seeds are only used once
        #self._reset_seeds()
        return results #_flatten_obs(obs, self.observation_space)
    

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def rollout(self, rollout_buffer, data : Any, indices: VecEnvIndices = None) -> None:
        start_time = time.time()
        target_remotes = self._get_target_remotes(indices)
        rollout_buffer.reset(self.shm.buf)

        #print([item.rewards.sum() for item in self.sl])
        for idx, remote in enumerate(target_remotes):
            remote.send(("rollbuf",  data))
        results = [remote.recv() for remote in self.remotes]
        #print([item.rewards.sum() for item in self.sl])
        #results = []
        #for name in names:
        #    existing_shm = shared_memory.SharedMemory(name=name)
        #    data = _ForkingPickler.loads(existing_shm.buf)
        #    results.append(data)
        #results = self.sl
        #print("Main process time:", time.time() - start_time)
        #return results
        last_values = [item[0] for item in results]
        dones = [item[1] for item in results]
        last_values = th.stack(last_values)
        #dones = th.tensor(np.stack(dones))
        dones = np.stack(dones)
        return last_values, dones 
    
    def set_rollout_buffer(self, rollbuf, create_rf, indices = None):
        self.shm = shared_memory.SharedMemory(create=True, size=rollbuf.nbytes())
        #rollbuf.set_buffer(self.shm.buf)
        rollbuf.reset(self.shm.buf)

        target_remotes = self._get_target_remotes(indices)

        for idx, remote in enumerate(target_remotes):
            if not debug:
                remote.send(('set_rollout_buffer', (create_rf, self.shm.name)))
            else:
                remote.send(('set_rollout_buffer', (rollbuf, self.shm.name)))
                remote.recv()
        if not debug:
            result = [remote.recv() for remote in target_remotes]

        #_ = [remote.recv() for remote in self.remotes]

        # custom manager to support custom classes
        #class CustomManager(SharedMemoryManager):
        #    # nothing
        #    pass
        #self.smm = Manager()
        #buf_list = [buf_fn() for _ in range(len(self.remotes))]
        #self.sl = self.smm.list(buf_list)

        #for idx, remote in enumerate(target_remotes):
        #    remote.send(('set_rollout_buffer', self.sl))
        #for remote in target_remotes:
        #    remote.recv()

    def set_model(self, model, indices = None):
        target_remotes = self._get_target_remotes(indices)

        for remote in target_remotes:
            remote.send(('set_model', ""))

        for torch_queue in self.torch_queues:
            torch_queue.put(model)

        for remote in target_remotes:
            remote.recv()
        




    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]
