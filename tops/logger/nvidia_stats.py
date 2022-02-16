import pynvml
def get_inof():
    try:
        utilz = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        in_use_by_us = gpu_in_use_by_this_process(handle)

        stats["gpu.{}.{}".format(i, "gpu")] = utilz.gpu
        stats["gpu.{}.{}".format(i, "memory")] = utilz.memory
        stats["gpu.{}.{}".format(i, "memoryAllocated")] = (
            memory.used / float(memory.total)
        ) * 100
        stats["gpu.{}.{}".format(i, "temp")] = temp

        if in_use_by_us:
            stats["gpu.process.{}.{}".format(i, "gpu")] = utilz.gpu
            stats["gpu.process.{}.{}".format(i, "memory")] = utilz.memory
            stats["gpu.process.{}.{}".format(i, "memoryAllocated")] = (
                memory.used / float(memory.total)
            ) * 100
            stats["gpu.process.{}.{}".format(i, "temp")] = temp

            # Some GPUs don't provide information about power usage
        try:
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_capacity_watts = (
                pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            )
            power_usage = (power_watts / power_capacity_watts) * 100

            stats["gpu.{}.{}".format(i, "powerWatts")] = power_watts
            stats["gpu.{}.{}".format(i, "powerPercent")] = power_usage

            if in_use_by_us:
                stats["gpu.process.{}.{}".format(i, "powerWatts")] = power_watts
                stats[
                    "gpu.process.{}.{}".format(i, "powerPercent")
                ] = power_usage
