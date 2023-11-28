import sys

sys.path.append("/code/tensorrt_llm")
import logging
import inspect
import time
import uuid
from http import HTTPStatus
from pathlib import Path
import numpy as np
import random
import torch
from transformers import LlamaTokenizer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

import tensorrt_llm
from app.api.protocol import (
    CompletionResponse, ErrorResponse, CompletionRequest, CompletionResponseChoice
)
from app.conf.config import config
from app.pyutil.log.log import _request_id_ctx_var
from app.pyutil.log.log import init as init_log
from build import get_engine_name  # isort:skip
from run import read_config, parse_input
from tensorrt_llm.runtime import SamplingConfig

EOS_TOKEN = 2
PAD_TOKEN = 2

decoder = None
tokenizer = None
model_config = None

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.post("/v1/completions")
async def completions(raw_request: Request):
    request = CompletionRequest(**await raw_request.json())
    logging.info(f"Received completion request: {request}")
    # if request.source != "GGCes6JvB6TM3x7KuirR":
    #     return create_error_response(HTTPStatus.BAD_REQUEST,
    #                                  "invalid source")
    output_texts = generate(request=request)
    choices = []
    for i in range(len(output_texts)):
        choices.append(CompletionResponseChoice(
            index=i,
            text=output_texts[i],
        ))
    return CompletionResponse(choices=choices)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    logging.info(
        f"{request.client.host} - "
        f"{request.method} "
        f"{request.url.path} "
        f"{response.status_code} "
        f"{int((time.time() - start_time) * 1000)}ms"
    )
    return response


@app.middleware("http")
async def dispatch_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", f"{uuid.uuid4()}{int(time.time() * 1000)}")
    token = _request_id_ctx_var.set(request_id)
    response = await call_next(request)
    _request_id_ctx_var.reset(token)
    return response


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def get_outputs(output_ids, input_lengths, max_output_len,
                tokenizer, output_csv, output_npy):
    output_texts = []
    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                output_texts.append(output_text)
                print(f'Output: \"{output_text}\"')
    return output_texts


def generate(
        request: CompletionRequest = None,
        input_file: str = None,
        output_csv: str = None,
        output_npy: str = None,
):
    random_seed_list = []
    for batch in range(request.n):
        random_seed_list.append([random.randint(0, 10000)])
    # random_seed = np.array(random_seed_list).astype(np.int32)
    tensor_from_list = torch.tensor(random_seed_list, dtype=torch.int64)
    global decoder, tokenizer, model_config
    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=request.beam_width,
                                     temperature=request.temperature,
                                     top_k=request.top_k,
                                     top_p=request.top_p,
                                     repetition_penalty=request.repetition_penalty,
                                     min_length=request.min_length)
    sampling_config.random_seed = tensor_from_list

    input_ids, input_lengths = parse_input(request.prompt, input_file, tokenizer,
                                           EOS_TOKEN,
                                           model_config.remove_input_padding, n=request.n)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0), max_input_length, request.max_tokens,
                  request.beam_width)

    output_gen_ids = decoder.decode(input_ids,
                                    input_lengths,
                                    sampling_config,
                                    streaming=request.streaming)
    torch.cuda.synchronize()

    if request.streaming:
        for output_ids in throttle_generator(output_gen_ids,
                                             request.streaming_interval):
            if runtime_rank == 0:
                return print_output(output_ids, input_lengths, request.max_tokens,
                                    tokenizer, output_csv, output_npy)
    else:
        output_ids = output_gen_ids
        if runtime_rank == 0:
            return get_outputs(output_ids, input_lengths, request.max_tokens, tokenizer,
                               output_csv, output_npy)


def create_session(log_level: str = 'error', engine_dir: str = 'gpt_outputs', tokenizer_dir: str = None, ):
    global model_config
    global tokenizer
    global decoder
    tensorrt_llm.logger.set_level(log_level)

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, tp_size, pp_size, dtype = read_config(config_path)
    world_size = tp_size * pp_size

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False,
                                                     debug_tensors_to_save=None)
    if runtime_rank == 0:
        print(f"Running the {dtype} engine ...")


@app.on_event("startup")
async def startup_event():
    init_log(config.log)
    create_session(engine_dir="./OpenHermes-2.5-Mistral-7B/trt_engines/fp16/1-gpu/",
                   tokenizer_dir="./OpenHermes-2.5-Mistral-7B/")


if __name__ == "__main__":
    uvicorn.run(app, port=8886, host="0.0.0.0")
