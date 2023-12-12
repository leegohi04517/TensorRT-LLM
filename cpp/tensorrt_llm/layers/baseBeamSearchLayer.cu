/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/beamSearchPenaltyKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"
#include "tensorrt_llm/layers/fillBuffers.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

__global__ void update_indir_cache_kernel(int* tgt_indir_cache, const int* src_indir_cache, const int** parent_ids,
    const bool* finished, const int* sequence_lengths, const int* input_lengths, int batch_dim, int local_batch_size,
    int beam_width, int max_kv_cache_length, int max_seq_len)
{
    int time_step = threadIdx.x + blockIdx.x * blockDim.x;
    int bb_id = threadIdx.y + blockIdx.y * blockDim.y;   // should be just blockIdx.y?
    const int current_step{sequence_lengths[bb_id] - 1}; // the sequence_lengths is updated, need to minus 1
    const int batch_id = bb_id / beam_width;
    const int beam_id = bb_id % beam_width;
    if (bb_id >= beam_width * local_batch_size || time_step < (max_seq_len - max_kv_cache_length) || finished[bb_id])
    {
        return;
    }
    int time_step_circ = time_step % max_kv_cache_length;

    // for the parent_ids, we will still keep it for all past tokens (i.e. max_seq_len)
    const int src_beam = parent_ids[batch_id][beam_id * max_seq_len + current_step];

    // for the indir tables, we have the cyclic kv cache.
    const uint32_t tgt_offset
        = batch_id * beam_width * max_kv_cache_length + beam_id * max_kv_cache_length + time_step_circ;
    const uint32_t src_offset
        = batch_id * beam_width * max_kv_cache_length + src_beam * max_kv_cache_length + time_step_circ;

    tgt_indir_cache[tgt_offset] = (time_step == current_step) ? beam_id : src_indir_cache[src_offset];
}

void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache, const int** parent_ids,
    const bool* finished, const int* sequence_lengths, const int* input_lengths, int batch_dim, int local_batch_size,
    int beam_width, int max_seq_len, int max_kv_cache_length, cudaStream_t stream)
{
    const dim3 block(32);
    // Update indirections steps [input_length[bb_id], sequence_lengths[bb_id]], included
    const dim3 grid((max_seq_len + block.x - 1) / block.x, local_batch_size * beam_width);
    update_indir_cache_kernel<<<grid, block, 0, stream>>>(tgt_indir_cache, src_indir_cache, parent_ids, finished,
        sequence_lengths, input_lengths, batch_dim, local_batch_size, beam_width, max_kv_cache_length, max_seq_len);
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    IAllocator* allocator, bool is_free_buffer_after_forward)
    : BaseLayer(stream, allocator, is_free_buffer_after_forward, nullptr)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer)
    : BaseLayer(beam_search_layer)
    , vocab_size_(beam_search_layer.vocab_size_)
    , vocab_size_padded_(beam_search_layer.vocab_size_padded_)
    , topk_softmax_workspace_size_(beam_search_layer.topk_softmax_workspace_size_)
{
}

template <typename T>
BaseBeamSearchLayer<T>::~BaseBeamSearchLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
void BaseBeamSearchLayer<T>::freeBuffer()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&temperature_buf_));
        allocator_->free((void**) (&min_lengths_buf_));
        allocator_->free((void**) (&repetition_penalty_buf_));
        is_allocate_buffer_ = false;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::allocateBuffer(size_t batch_size)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    temperature_buf_ = allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false);
    min_lengths_buf_ = allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false);
    repetition_penalty_buf_ = allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false);

    is_allocate_buffer_ = true;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::setupBase(size_t batch_size, SetupParams const& setupParams)
{
    allocateBuffer(batch_size);
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // Setup penalties.
    FillBuffers const fillBuffers{batch_size, stream_};

    fillBuffers(setupParams.temperature, 1.0f, mTemperature, temperature_buf_);
    fillBuffers(setupParams.min_length, 1, mMinLength, min_lengths_buf_);

    mRepetitionPenaltyType = RepetitionPenaltyType::None;
    if (setupParams.repetition_penalty || setupParams.presence_penalty)
    {
        TLLM_CHECK_WITH_INFO(!(setupParams.repetition_penalty && setupParams.presence_penalty),
            "Found ambiguous parameters repetition_penalty and presence_penalty "
            "which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        mRepetitionPenaltyType
            = setupParams.repetition_penalty ? RepetitionPenaltyType::Multiplicative : RepetitionPenaltyType::Additive;
        if (mRepetitionPenaltyType == RepetitionPenaltyType::Multiplicative)
        {
            fillBuffers(setupParams.repetition_penalty, 1.0f, mRepetitionPenalty, repetition_penalty_buf_);
        }
        else
        {
            fillBuffers(setupParams.presence_penalty, 1.0f, mRepetitionPenalty, repetition_penalty_buf_);
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::forward(BeamSearchOutputParams& outputs, ForwardParams const& params)
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    Tensor& output_ids_ptr = outputs.output_ids_ptr;

    const auto batch_size = static_cast<std::int32_t>(output_ids_ptr.shape[0]);
    const auto beam_width = static_cast<std::int32_t>(output_ids_ptr.shape[1]);
    const auto max_seq_len = static_cast<std::int32_t>(output_ids_ptr.shape[2]);

    TLLM_CHECK_WITH_INFO(params.ite == 0, "Pipeline Parallelism is not supported yet !");

    const int ite{params.ite};
    Tensor const& logits = params.logits;
    const auto local_batch_size = logits.shape[0];

    const T* embedding_bias = params.embedding_bias ? params.embedding_bias->template getPtr<const T>() : nullptr;

    auto* end_ids = params.end_ids.template getPtr<const int>();
    auto* const input_lengths = params.input_lengths ? params.input_lengths->template getPtr<const int>() : nullptr;
    int* sequence_length = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    invokeAddBiasApplyPenalties(logits.getPtr<T>(), output_ids_ptr.template getPtr<const int*>(),
        outputs.parent_ids_ptr.template getPtr<const int*>(), input_lengths, sequence_length, embedding_bias, ite,
        local_batch_size, batch_size, beam_width, vocab_size_, vocab_size_padded_, end_ids, temperature_buf_,
        mTemperature, repetition_penalty_buf_, mRepetitionPenalty, mRepetitionPenaltyType, min_lengths_buf_,
        max_seq_len, stream_);
    sync_check_cuda_error();

    invokeSoftMax(outputs, params);

    if (beam_width > 1)
    {
        update_indir_cache_kernelLauncher(outputs.tgt_cache_indirection.template getPtr<int>(),
            params.src_cache_indirection.template getPtr<const int>(),
            outputs.parent_ids_ptr.template getPtr<const int*>(), outputs.finished->template getPtr<const bool>(),
            sequence_length, input_lengths, batch_size, local_batch_size, beam_width, max_seq_len,
            params.max_kv_cache_length, stream_);
        sync_check_cuda_error();
    }
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_)
    {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseBeamSearchLayer<float>;
template class BaseBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
