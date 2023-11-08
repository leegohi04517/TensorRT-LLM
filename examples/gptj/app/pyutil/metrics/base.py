import logging

from .os_info import get_gpu_info

device_name = 'None'
total_memory = 0
device_count = 0


def init_metrics_client():
    logging.info("metrics init begin")
    global device_name, total_memory, device_count
    device_name, total_memory, device_count = get_gpu_info()
    # MetricsManager.common_gpu_dim = [
    #     {
    #         'Name': 'gpuModel',
    #         'Value': gpu_model,
    #     },
    #     {
    #         'Name': 'gpuSize',
    #         'Value': str(gpu_size),
    #     },
    #     {
    #         'Name': 'gpuCount',
    #         'Value': str(gpu_count),
    #     },
    # ]
    logging.info(f"get gpu info gpu_model={device_name}, gpu_size={total_memory}")


def device_name():
    return device_name


def total_memory():
    return total_memory


def device_count():
    return device_count
