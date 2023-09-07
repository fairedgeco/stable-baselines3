import multiprocessing as mp
import time
import warnings
from collections import OrderedDict
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from stable_baselines3.common.preprocessing import get_obs_shape

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

def copy_obervation_to_share_mem(observation, shm):
    new_observation = {}
    current_bytes = 0
    for key, val in observation.items():
        new_observation[key] = np.ndarray(val.shape, dtype=val.dtype, buffer=shm.buf, offset = current_bytes)
        current_bytes += val.nbytes
        np.copyto(new_observation[key], val)
    return new_observation

def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    queue: mp.Queue,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    sum_of_step = 0
    sum_of_done = 0
    sum_of_copy = 0
    sum_of_all = 0
    sum_of_send = 0
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                start_time = time.time()
                observation, reward, terminated, truncated, info = env.step(data)
                step_time = time.time()
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                #if done:
                    # save final observation where user can get it, then reset
                    #info["terminal_observation"] = observation
                    #observation, reset_info = env.reset()
                done_time = time.time()
                new_obervation = copy_obervation_to_share_mem(observation, shm)
                copy_time = time.time()
                sum_of_step += step_time - start_time
                sum_of_done += done_time - step_time
                sum_of_copy += copy_time - done_time
                remote.send((reward, done, info, reset_info, time.time()))
                #queue.put((reward, done, info, reset_info, time.time()))
                send_time = time.time()
                sum_of_send += send_time - copy_time
                sum_of_all += send_time - start_time
                if done:
                    print(env.rank, sum_of_step, sum_of_done, sum_of_copy, sum_of_send, sum_of_all)
            elif cmd == "reset":
                observation, reset_info = env.reset(seed=data)
                new_obervation = copy_obervation_to_share_mem(observation, shm)
                remote.send((reset_info))
            elif cmd == "set_shared_mem":
                shm = shared_memory.SharedMemory(name=data, create=False)
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
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.queues = [mp.Queue() for _ in range(n_envs)]
        self.processes = []
        for work_remote, remote, queue, env_fn in zip(self.work_remotes, self.remotes, self.queues, env_fns):
            args = (work_remote, remote, queue, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            # pytype: disable=attribute-error
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            # pytype: enable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        self.sum_of_recv = 0
        self.sum_of_zip = 0
        self.sum_of_shared = 0
        self.sum_of_all = 0
        self.sum_of_send = 0
        self.sum_of_wait = 0
        self.sum_of_falt = 0
        from concurrent.futures import ThreadPoolExecutor
        self.thread_pool = ThreadPoolExecutor(max_workers=16)

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.step_start_time = time.time()
        self.task_list = []
        def send_and_recv(remote, queue, args):
            remote.send(args)
            #data = queue.get() 
            data = remote.recv()
            return data

        for remote, queue, action in zip(self.remotes, self.queues, actions):
            self.task_list.append(self.thread_pool.submit(send_and_recv, remote, queue, ('step', action)))
            #self.task_list.append(self.thread_pool.submit(remote.send, ('step', action)))
            #remote.send(("step", action))
        self.waiting = True
        self.sum_of_send += time.time() - self.step_start_time 
    
    def create_shared_mem(self, nbytes):
        self.shared_mem_list = [shared_memory.SharedMemory(size=nbytes, create=True) for _ in self.remotes]
        for shm, remote in zip(self.shared_mem_list, self.remotes):
            remote.send(("set_shared_mem", shm.name))

    def get_obs_from_shared_mem(self):
        obs_shape = get_obs_shape(self.observation_space)
        observations = [] 
        for shm, remote in zip(self.shared_mem_list, self.remotes):
            observation = {}
            current_ntypes = 0
            for key, obs_input_shape in obs_shape.items():
                observation[key] = np.ndarray(obs_input_shape, dtype=np.float32, buffer = shm.buf, offset = current_ntypes)
                current_ntypes += observation[key].nbytes
            observations.append(observation)
        return observations


    def step_wait(self) -> VecEnvStepReturn:
        start_time = time.time()
        results = [task.result() for task in self.task_list]
        #task_list = []
        #for idx, remote in enumerate(self.remotes):
        #    #self.task_list[idx].result()
        #    task_list.append(self.thread_pool.submit(remote.recv))
        #results = [task.result() for task in task_list]

        #results = [remote.recv() for remote in self.remotes]
        recv_time = time.time()
        self.waiting = False
        rews, dones, infos, self.reset_infos, time_stamp = zip(*results)
        #print(rews, dones, infos, self.reset_infos)

        self.sum_of_wait = max(time_stamp) - min(time_stamp)
        zip_time = time.time()
        obs = self.get_obs_from_shared_mem()
        shared_time = time.time()


        result = _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
        flatten_time = time.time()

        self.sum_of_recv += recv_time - start_time
        self.sum_of_zip +=zip_time - recv_time 
        self.sum_of_shared +=shared_time - zip_time 
        self.sum_of_falt += flatten_time - shared_time
        self.sum_of_all += flatten_time- self.step_start_time 

        if dones[0]:
            print("SubproceVecEnv", self.sum_of_send, self.sum_of_recv, self.sum_of_zip, self.sum_of_shared, self.sum_of_falt)
            print("SubproceVecEnv", self.sum_of_wait, self.sum_of_all)
        return result

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", self._seeds[env_idx]))
        results = [remote.recv() for remote in self.remotes]
        self.reset_infos = zip(*results)
        obs = self.get_obs_from_shared_mem()
        # Seeds are only used once
        self._reset_seeds()
        return _flatten_obs(obs, self.observation_space)

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
