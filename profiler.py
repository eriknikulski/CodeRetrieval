import torch
from torch.profiler import profile, ProfilerActivity

import const
import ddp


def trace_handler(p):
    if ddp.is_main_process():
        f_num = const.SLURM_JOB_ID + '_' + str(p.step_num)
        p.export_chrome_trace(const.PROFILER_TRACE_PATH + f_num + '.json')
        if ProfilerActivity.CUDA in p.activities:
            p.export_stacks(const.PROFILER_STACKS_PATH + f_num + '.txt', 'self_cuda_time_total')
    ddp.sync()


class Profiler:
    def __init__(self, profiler=None, active=True):
        if not profiler:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            profiler = profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1),
                on_trace_ready=trace_handler,
                with_stack=True,
            )
        self.profiler = profiler
        self.active = active

    def __enter__(self):
        if self.active:
            self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.active:
            self.profiler.stop()
        return self

    def step(self):
        if self.active:
            self.profiler.step()
