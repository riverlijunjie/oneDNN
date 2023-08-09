/*******************************************************************************
* Copyright 2022 Intel Corporation
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
*******************************************************************************/
#include <float.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_weights_decompression_kernel.hpp"

#define GET_OFF(field) offsetof(weights_decompression_runtime_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::generate() {
    preamble();

    mov(reg_weights, ptr[param1 + GET_OFF(weights_ptr)]);
    mov(reg_decomp_buffer, ptr[param1 + GET_OFF(decomp_buffer_ptr)]);
    if (jcp_.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    }
    if (jcp_.with_zero_points) {
        mov(reg_zero_points, ptr[param1 + GET_OFF(zero_points_ptr)]);
    }
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);
    for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
        if (jcp_.with_scales)
            uni_vmovups(vmm_scales(ocb), ptr[reg_scales + ocb * vec_size * sizeof(float)]);
        if (jcp_.with_zero_points)
            uni_vmovups(vmm_zero_points(ocb), ptr[reg_zero_points + ocb * vec_size * sizeof(float)]);
    }

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t decomp_buf_dt_size = types::data_type_size(jcp_.decomp_buffer_dt);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, 1);
        jl(ic_end_label, T_NEAR);

        for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
            uni_vpmovzxbd(vmm_weights(ocb), ptr[reg_weights + ocb * vec_size * sizeof(uint8_t)]);
            uni_vcvtdq2ps(vmm_weights(ocb), vmm_weights(ocb));
            if (jcp_.with_zero_points)
                uni_vsubps(vmm_weights(ocb), vmm_weights(ocb), vmm_zero_points(ocb));
            if (jcp_.with_scales)
                uni_vmulps(vmm_weights(ocb), vmm_weights(ocb), vmm_scales(ocb));

            switch (jcp_.decomp_buffer_dt) {
                case data_type::f32: {
                    uni_vmovups(ptr[reg_decomp_buffer + ocb * vec_size * decomp_buf_dt_size], vmm_weights(ocb));
                    break;
                }
                case data_type::bf16: {
                    Ymm ymm_weights = Ymm(vmm_weights(ocb).getIdx());
                    vcvtneps2bf16(ymm_weights, vmm_weights(ocb));
                    vmovdqu16(ptr[reg_decomp_buffer + ocb * vec_size * decomp_buf_dt_size], ymm_weights);
                    break;
                }
                default: assert(!"unsupported data type");
            }
        }

        dec(reg_ic_size);
        add(reg_weights, sizeof(uint8_t) * jcp_.oc_size);
        add(reg_decomp_buffer, decomp_buf_dt_size * jcp_.oc_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}

template struct jit_brgemm_weights_decompression_kernel_t<avx512_core>;
template struct jit_brgemm_weights_decompression_kernel_t<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
