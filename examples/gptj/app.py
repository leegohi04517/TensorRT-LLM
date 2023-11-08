import asyncio
import logging
import time
import uuid
from http import HTTPStatus
from pathlib import Path

import torch
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
from examples.gptj.build import get_engine_name  # isort:skip
from examples.gptj.run import read_config, parse_input
from examples.gptj.utils import token_encoder
from tensorrt_llm.runtime import SamplingConfig

# 创建一个锁对象
lock = None

MERGES_FILE = "merges.txt"
VOCAB_FILE = "vocab.json"

PAD_ID = 50256
START_ID = 50256
END_ID = 50256

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
    output_texts = generate(input_text=request.prompt, max_output_len=request.max_tokens)
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


def get_outputs(output_ids, cum_log_probs, input_lengths, sequence_lengths,
                tokenizer, output_csv, output_npy):
    output_texts = []
    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input {b}: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = sequence_lengths[b][beam]
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                output_texts.append(output_text)
                if num_beams > 1:
                    cum_log_prob = cum_log_probs[b][beam]
                    print(
                        f'Output {b}, beam {beam}: \"{output_text}\" (cum_log_prob: {cum_log_prob})'
                    )
                else:
                    print(f'Output {b}: \"{output_text}\"')
    return output_texts


def generate(
        max_output_len: int,
        input_text: str = 'Born in north-east France, Soyer trained as a',
        input_file: str = None,
        output_csv: str = None,
        output_npy: str = None,
        num_beams: int = 1,
        min_length: int = 1,
):
    global decoder, tokenizer, model_config
    sampling_config = SamplingConfig(end_id=END_ID,
                                     pad_id=PAD_ID,
                                     num_beams=num_beams,
                                     min_length=min_length)
    session_time = time.time()
    input_ids, input_lengths = parse_input(input_text, input_file, tokenizer,
                                           PAD_ID,
                                           model_config.remove_input_padding)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0),
                  max_input_length,
                  max_output_len,
                  beam_width=num_beams)
    setup_time = time.time()
    print(f"setup_time cost:{setup_time - session_time}")

    outputs = decoder.decode(input_ids,
                             input_lengths,
                             sampling_config,
                             output_sequence_lengths=True,
                             return_dict=True)
    output_time = time.time()
    print(f"output_time cost:{output_time - setup_time}")
    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']
    torch.cuda.synchronize()

    cum_log_probs = decoder.cum_log_probs if num_beams > 1 else None

    return get_outputs(output_ids, cum_log_probs, input_lengths, sequence_lengths,
                       tokenizer, output_csv, output_npy)


def create_session(log_level: str = 'error', engine_dir: str = 'gpt_outputs', hf_model_location: str = 'gptj', ):
    tensorrt_llm.logger.set_level(log_level)

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    global model_config
    model_config, world_size, dtype, max_input_len = read_config(config_path)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    vocab_file = Path(hf_model_location) / VOCAB_FILE
    merges_file = Path(hf_model_location) / MERGES_FILE
    assert vocab_file.is_file(), f"{vocab_file} does not exist"
    assert merges_file.is_file(), f"{merges_file} does not exist"
    global tokenizer
    tokenizer = token_encoder.get_encoder(vocab_file, merges_file)
    engine_name = get_engine_name('gptj', dtype, world_size, runtime_rank)
    serialize_path = Path(engine_dir) / engine_name
    print(f"engine_name:{engine_name} path:{serialize_path}")
    start_time = time.time()
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    print(f"read engine buffer")
    load_time = time.time()
    print(f"load cost time:{load_time - start_time}")
    global decoder
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)
    session_time = time.time()
    print(f"session cost time:{session_time - load_time}")


@app.on_event("startup")
async def startup_event():
    init_log(config.log)
    create_session(engine_dir="examples/gptj/pygmalion-6b-engine", hf_model_location="examples/gptj/pygmalion-6b")


if __name__ == "__main__":
    uvicorn.run(app, port=13883, host="0.0.0.0")
