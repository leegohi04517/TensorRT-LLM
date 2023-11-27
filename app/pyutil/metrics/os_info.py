import torch
import logging


def get_gpu_info():
    device_count = torch.cuda.device_count()
    logging.info(f"get gpu count {device_count}")
    if device_count == 0:
        return 'None', 0, 0
    else:
        device = torch.cuda.get_device_properties(0)
        mem_info = device.total_memory / (1024 ** 3)
        mem_info = round(mem_info, 2)
        return device.name.replace(' ', '-'), mem_info, device_count
