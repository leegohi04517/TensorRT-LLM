# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from math import ceil

import torch
from allowed_configs import get_build_config, get_model_family
from base_benchmark import BaseBenchmark, get_engine_name, serialize_engine

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode


class GPTBenchmark(BaseBenchmark):

    def __init__(self,
                 engine_dir,
                 model_name,
                 mode,
                 batch_sizes,
                 in_out_lens,
                 dtype,
                 refit,
                 num_beams,
                 top_k,
                 top_p,
                 output_dir,
                 n_positions=None,
                 max_input_len=None,
                 max_output_len=None,
                 max_batch_size=None,
                 enable_custom_all_reduce=None,
                 **kwargs):
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.refit = refit
        self.num_beams = num_beams
        self.build_time = 0
        self.mode = mode  # plugin or ootb or ootb-except-mha
        self.fuse_bias = True

        self.cuda_graph_mode = kwargs.get('enable_cuda_graph', False)
        self.strongly_typed = kwargs.get('strongly_typed', False)
        self.enable_custom_all_reduce = enable_custom_all_reduce

        if engine_dir is not None:
            # Get build configs from engine directory is done in base class
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(self.serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.world_size = tensorrt_llm.mpi_world_size()
            self.apply_query_key_layer_scaling = False

            self.use_weight_only = False
            self.per_group = False
            self.weight_only_precision = 'int8'
            self.per_token = False
            self.per_channel = False

            use_mha_plugin = mode == 'plugin' or mode == 'ootb-except-mha'
            mha_plg_dtype = dtype if use_mha_plugin else False
            use_non_mha_plugin = mode == 'plugin'
            non_mha_plg_dtype = dtype if use_non_mha_plugin else False

            self.use_gpt_attention_plugin = mha_plg_dtype
            self.use_gemm_plugin = non_mha_plg_dtype
            # Starting TRT9.1 OOTB norm layer sees improvement over plugin norm layer
            self.use_layernorm_plugin = False
            self.use_rmsnorm_plugin = False
            self.use_lookup_plugin = non_mha_plg_dtype
            self.enable_context_fmha = use_mha_plugin

            self.remove_input_padding = use_non_mha_plugin

            for key, value in get_build_config(model_name).items():
                setattr(self, key, value)

            if self.quantization is None:
                self.quantization = kwargs.get('quantization', None)

            self.set_quantization()

            # Override the n_position/max_input_len/max_output_len/max_batch_size to value from cmd line if that's specified.
            if n_positions is not None:
                assert isinstance(
                    n_positions, int
                ) and n_positions > 0, f"n_positions should be a valid int number, got {n_positions}"
                self.n_positions = n_positions
            if max_input_len is not None:
                assert isinstance(
                    max_input_len, int
                ) and max_input_len > 0, f"max_input_len should be a valid int number, got {max_input_len}"
                self.max_input_len = max_input_len
            if max_output_len is not None:
                assert isinstance(
                    max_output_len, int
                ) and max_output_len > 0, f"max_output_len should be a valid int number, got {max_output_len}"
                self.max_output_len = max_output_len
            if max_batch_size is not None:
                assert isinstance(
                    max_batch_size, int
                ) and max_batch_size > 0, f"max_batch_size should be a valid int number, got {max_batch_size}"
                self.max_batch_size = max_batch_size
            if self.num_kv_heads is None:
                self.num_kv_heads = self.num_heads
            if kwargs.get('force_num_layer_1', False):
                self.num_layers = 1
            engine_buffer = self.build()

        assert engine_buffer is not None

        model_config = tensorrt_llm.runtime.ModelConfig(
            num_heads=self.num_heads // self.world_size,
            num_kv_heads=ceil(self.num_kv_heads / self.world_size),
            hidden_size=self.hidden_size // self.world_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            gpt_attention_plugin=self.use_gpt_attention_plugin,
            remove_input_padding=self.remove_input_padding,
            quant_mode=self.quant_mode,
            use_custom_all_reduce=self.enable_custom_all_reduce,
        )
        if model_name == 'chatglm_6b':
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=130005,
                pad_id=3,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.ChatGLMGenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        elif model_name in ['chatglm2_6b', 'chatglm3_6b']:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=2,
                pad_id=0,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        else:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=50256,
                pad_id=50256,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config,
                engine_buffer,
                self.runtime_mapping,
                cuda_graph_mode=self.cuda_graph_mode)

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            if inlen > self.max_input_len or outlen > self.max_output_len:
                print(
                    f'[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) and '
                    f'outlen({outlen}) <= max_outlen({self.max_output_len}) failed, skipping.'
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f'[WARNING] check batch_size({batch_size}) '
                        f'<= max_batch_size({self.max_batch_size}) failed, skipping.'
                    )
                    continue
                yield (batch_size, inlen, outlen)

    def prepare_inputs(self, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = torch.tensor([inlen
                                      for _ in range(batch_size)]).int().cuda()

        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        return (input_ids, input_lengths)

    def set_quantization(self):
        self.quant_mode = QuantMode(0)

        if self.quantization == "fp8":
            self.strongly_typed = True
            self.quant_mode = self.quant_mode.set_fp8_qdq()
            self.quant_mode = self.quant_mode.set_fp8_kv_cache()

        elif self.quantization == "fp8_gemm":
            self.strongly_typed = True
            self.quant_mode = self.quant_mode.set_fp8_qdq()

        elif self.quantization == "fp8_kv_cache":
            self.strongly_typed = True
            self.quant_mode = self.quant_mode.set_fp8_kv_cache()

        elif self.quantization == "int8_sq_per_tensor":
            self.use_smooth_quant = True
            self.quant_mode = QuantMode.use_smooth_quant(
                self.per_token, self.per_channel)

        elif self.quantization == "int8_sq_per_token_channel":
            self.use_smooth_quant = True
            self.per_token = True
            self.per_channel = True
            self.quant_mode = QuantMode.use_smooth_quant(
                self.per_token, self.per_channel)

        elif self.quantization == "int8_weight_only":
            self.use_smooth_quant = False
            self.use_weight_only = True
            self.weight_only_precision = 'int8'
            self.quant_mode = QuantMode.use_weight_only(False)

        elif self.quantization == "int4_weight_only":
            self.use_weight_only = True
            self.weight_only_precision = 'int4'
            self.quant_mode = QuantMode.use_weight_only(True)

        elif self.quantization == "int4_weight_only_awq":
            self.use_weight_only = True
            self.per_group = True
            self.weight_only_precision = 'int4_awq'
            self.quant_mode = QuantMode.from_description(
                quantize_weights=True,
                quantize_activations=False,
                per_token=False,
                per_channel=False,
                per_group=True,
                use_int4_weights=True)

        elif self.quantization == "int4_weight_only_gptq":
            self.use_weight_only = True
            self.per_group = True
            self.weight_only_precision = 'int4_gptq'
            self.quant_mode = QuantMode.from_description(
                quantize_weights=True,
                quantize_activations=False,
                per_token=False,
                per_channel=False,
                per_group=True,
                use_int4_weights=True)

        elif self.quantization == None:
            pass

        else:
            raise Exception(f'{0} is invalid config: {self.quantization}')

    def build(self):
        builder = Builder()
        builder_config = builder.create_builder_config(
            name=self.model_name,
            precision=self.dtype,
            timing_cache=None,
            tensor_parallel=self.world_size,  # TP only
            parallel_build=True,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
            int8=self.quant_mode.has_act_and_weight_quant(),
            quant_mode=self.quant_mode,
            use_refit=self.refit,
            opt_level=self.builder_opt,
            strongly_typed=self.strongly_typed)
        engine_name = get_engine_name(self.model_name, self.dtype,
                                      self.world_size, self.runtime_rank)

        kv_dtype = str_dtype_to_trt(self.dtype)

        # Initialize Module
        family = get_model_family(self.model_name)
        if family == "gpt":
            tensorrt_llm_model = tensorrt_llm.models.GPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                position_embedding_type=PositionEmbeddingType.learned_absolute
                if self.position_embedding_type is None else
                self.position_embedding_type,
                rotary_embedding_percentage=self.rotary_pct,
                quant_mode=self.quant_mode,
                bias=self.bias)
        elif family == "opt":
            tensorrt_llm_model = tensorrt_llm.models.OPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                pre_norm=self.pre_norm,
                do_layer_norm_before=self.do_layer_norm_before)
        elif family == "llama":
            tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mlp_hidden_size=self.inter_size,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                quant_mode=self.quant_mode)
        elif family == "gptj":
            tensorrt_llm_model = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                quant_mode=self.quant_mode)
        elif family == "gptneox":
            tensorrt_llm_model = tensorrt_llm.models.GPTNeoXForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling)
        elif family == "chatglm":
            tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                quant_mode=self.quant_mode,
                model_name="chatglm_6b")
        elif family == "chatglm2":
            tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                quant_mode=self.quant_mode,
                model_name="chatglm2_6b")
        elif family == "chatglm3":
            tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                quant_mode=self.quant_mode,
                model_name="chatglm3_6b")
        elif family == "bloom":
            tensorrt_llm_model = tensorrt_llm.models.BloomForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                quant_mode=self.quant_mode,
                use_parallel_embedding=(self.model_name == 'bloom_176b'))
        elif family == "falcon":
            tensorrt_llm_model = tensorrt_llm.models.FalconForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                bias=self.bias,
                quant_mode=self.quant_mode,
                use_alibi=self.use_alibi,
                new_decoder_architecture=self.new_decoder_architecture,
                parallel_attention=self.parallel_attention,
                mapping=tensorrt_llm.Mapping(world_size=self.world_size,
                                             tp_size=self.world_size))
        else:
            raise Exception(f'Unexpected model: {self.model_name}')

        quant_kwargs = {}
        if family == "llama" and self.use_weight_only:
            if self.weight_only_precision == 'int4_awq':
                quant_kwargs = {
                    "group_size": 128,
                    "zero": False,
                    "pre_quant_scale": True,
                    "exclude_modules": [],
                }
            elif self.weight_only_precision == 'int4_gptq':
                quant_kwargs = {
                    "group_size": 128,
                    "zero": True,
                    "pre_quant_scale": False,
                }
        tensorrt_llm_model = quantize_model(tensorrt_llm_model, self.quant_mode,
                                            **quant_kwargs)

        # Module -> Network
        network = builder.create_network()
        network.trt_network.name = engine_name

        not_fp8_quantization = self.quantization is None or "fp8" not in self.quantization

        if self.use_gpt_attention_plugin:
            network.plugin_config.set_gpt_attention_plugin(
                dtype=self.use_gpt_attention_plugin)
        if self.use_gemm_plugin and not_fp8_quantization:
            network.plugin_config.set_gemm_plugin(dtype=self.use_gemm_plugin)
        if self.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                dtype=self.use_layernorm_plugin)
        if self.use_rmsnorm_plugin:
            network.plugin_config.set_rmsnorm_plugin(
                dtype=self.use_rmsnorm_plugin)
        if self.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self.remove_input_padding:
            network.plugin_config.enable_remove_input_padding()

        # Quantization plugins.
        if self.use_smooth_quant:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=self.dtype)
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=self.dtype)
            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif self.use_weight_only:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype=self.dtype)

        # RMS norm plugin for SmoothQuant
        if self.quant_mode.has_act_and_weight_quant(
        ) and 'llama' in self.model_name:
            network.plugin_config.set_rmsnorm_quantization_plugin()

        if self.world_size > 1:
            network.plugin_config.set_nccl_plugin(self.dtype,
                                                  self.enable_custom_all_reduce)

        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=self.use_lookup_plugin)

        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_model.named_parameters())

            # Forward
            inputs = tensorrt_llm_model.prepare_inputs(self.max_batch_size,
                                                       self.max_input_len,
                                                       self.max_output_len,
                                                       True, self.num_beams)
            tensorrt_llm_model(*inputs)

        if self.fuse_bias:
            tensorrt_llm.graph_rewriting.optimize(network)

        # Network -> Engine
        start = time.time()
        engine = builder.build_engine(network, builder_config)
        end = time.time()
        self.build_time = round(end - start, 2)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.serialize_path = os.path.join(self.output_dir,
                                               self.engine_name)
            serialize_engine(engine, self.serialize_path)
            if self.runtime_rank == 0:
                config_path = os.path.join(self.output_dir, 'config.json')
                builder_config.plugin_config = network.plugin_config
                builder.save_config(builder_config, config_path)
        return engine

    def run(self, inputs, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        if self.remove_input_padding:
            self.decoder.decode_batch(inputs[0], self.sampling_config)
        else:
            self.decoder.decode(inputs[0], inputs[1], self.sampling_config)
        torch.cuda.synchronize()

    def report(self, config, latency, percentile95, percentile99, peak_gpu_used,
               csv):
        report_dict = super().get_report_dict()
        batch_size, inlen, outlen = config[0], config[1], config[2]
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        report_dict["num_heads"] = self.num_heads
        report_dict["num_kv_heads"] = self.num_kv_heads
        report_dict["num_layers"] = self.num_layers
        report_dict["hidden_size"] = self.hidden_size
        report_dict["vocab_size"] = self.vocab_size
        report_dict["batch_size"] = batch_size
        report_dict["input_length"] = inlen
        report_dict["output_length"] = outlen
        report_dict["latency(ms)"] = latency
        report_dict["build_time(s)"] = self.build_time
        report_dict["tokens_per_sec"] = tokens_per_sec
        report_dict["percentile95(ms)"] = percentile95
        report_dict["percentile99(ms)"] = percentile99
        report_dict["gpu_peak_mem(gb)"] = peak_gpu_used
        if self.runtime_rank == 0:
            if csv:
                line = ",".join([str(v) for v in report_dict.values()])
                print(line)
                with open(self.get_csv_filename(), "a") as file:
                    file.write(line + "\n")
            else:
                kv_pairs = [f"{k} {v}" for k, v in report_dict.items()]
                line = '[BENCHMARK] ' + " ".join(kv_pairs)
                print(line)
