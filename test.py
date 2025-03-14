# Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
from typing import Sequence

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import gpu as plgpu
import jax.numpy as jnp


from jax._src import core as jax_core
from jax._src.pallas.triton import lowering
from jax._src.lib.triton import dialect as tt_dialect


from jax._src.pallas.triton import pallas_call_registration
from pallas_call_registration import pallas_call_lowering
pallas_call_registration.pallas_call_lowering = pallas_call_lowering

LEN = 33280
ST = LEN - 256
LEN_BLOCK_SIZE = LEN // 128

def join(
    args: Sequence[jax.Array],
) -> Sequence[jax.Array]:
  return join_p.bind(
      *args,
  )

join_p = jax_core.Primitive("join_p")

@join_p.def_abstract_eval
def _join_abstract_eval(
    *avals: jax_core.ShapedArray, **kwargs
) -> Sequence[jax_core.ShapedArray]:
    del kwargs  # Unused.
    if not len(avals) == 2:
        raise ValueError(
            "join should have two arguments"
        )

    if avals[0].shape != avals[1].shape:
        raise ValueError(
            "Join arguments should have the same shape"
        )
    return jax_core.ShapedArray(avals[0].shape + (2,), avals[0].dtype)


@lowering.register_lowering(join_p)
def _join_lowering(
    ctx: lowering.LoweringRuleContext,
    *args,
):
    return tt_dialect.JoinOp(
          *args,
    ).result


def kernel_1_impl(
    x_ref,
    ips_ref,
    y_ref_out,
    z_ref_out,
    ys_ref_out,
    zs_ref_out,
    o_ref,
    block_k
):
  block_h, dim = x_ref.shape
  start_x = pl.program_id(0)
  split_block_idx = pl.program_id(1)

  ips = ips_ref[...]

  def read_and_unpack(src):
    assert src.dtype == jnp.int8
    sdtype = jax.ShapeDtypeStruct(src.shape, jnp.bfloat16)
    target_cast_lower, target_cast_upper = plgpu.elementwise_inline_asm(
        asm="""
        {
            .reg .s32 src_shifted;
            .reg .b32 bias;

            mov.b32 bias, 0x43084308;

            shr.s32 src_shifted, $4, 4;

            // normal ordering:
            // prmt.b32 $0, $4, src_shifted, 0xF4F0;
            // prmt.b32 $1, $4, src_shifted, 0xF5F1;
            // prmt.b32 $2, $4, src_shifted, 0xF6F2;
            // prmt.b32 $3, $4, src_shifted, 0xF7F3;

            // interleaved ordering:
            prmt.b32 $0, $4, src_shifted, 0xF1F0;
            prmt.b32 $1, $4, src_shifted, 0xF3F2;
            prmt.b32 $2, $4, src_shifted, 0xF5F4;
            prmt.b32 $3, $4, src_shifted, 0xF7F6;

            lop3.b32 $0, $0, 0x000F000F, bias, 0x6a;
            lop3.b32 $1, $1, 0x000F000F, bias, 0x6a;
            lop3.b32 $2, $2, 0x000F000F, bias, 0x6a;
            lop3.b32 $3, $3, 0x000F000F, bias, 0x6a;

            sub.bf16x2 $0, $0, bias;
            sub.bf16x2 $1, $1, bias;
            sub.bf16x2 $2, $2, bias;
            sub.bf16x2 $3, $3, bias;
        }
        """,
        args=[src],
        constraints=(
        "=r,=r,=r,=r,"
        "r"),
        result_shape_dtypes=(sdtype, sdtype),
        pack=4,
    )
    target_joined = join([target_cast_lower, target_cast_upper])
    target = target_joined.reshape(*src.shape[:-1], src.shape[-1] * 2)
    return target

  block_h_padding = max(block_h, 16)

  o = jnp.zeros((block_h_padding, dim), dtype=jnp.float32)

  curr_x_slice = pl.dslice(start_x * block_h, block_h_padding)
  x_mask = None
  if block_h < block_h_padding:
    x_mask = jnp.arange(block_h_padding) < block_h
  x = pl.load(x_ref, (curr_x_slice, pl.dslice(None)), mask=x_mask[:, None], other=0)

  def _dot(a, b):
    return pl.dot(a, b.astype(a.dtype))

  def body(start_i, _):
    curr_yz_slice = pl.dslice(start_i * block_k, block_k)

    mask = None
    span_k = start_i * block_k + jnp.arange(block_k)
    mask = (span_k < ips)

    y = pl.load(y_ref_out, (curr_yz_slice, slice(None)), mask=mask[:, None], other=0)
    y = read_and_unpack(y)

    c = _dot(x, y.T) 
    y_scale = pl.load(ys_ref_out, (curr_yz_slice, slice(None)), mask=mask[:, None], other=0)
    y_scale = jnp.broadcast_to(jnp.squeeze(y_scale), c.shape)
    c = c * y_scale

    z = pl.load(z_ref_out, (curr_yz_slice, slice(None)), mask=mask[:, None], other=0)
    z = read_and_unpack(z)
    z_scale = pl.load(zs_ref_out, (curr_yz_slice, slice(None)), mask=mask[:, None], other=0)
    z = z.astype(jnp.bfloat16) * z_scale

    o_curr = _dot(c.astype(x.dtype), z)

    return o_curr

  upper_bound = pl.cdiv(lax.min(LEN_BLOCK_SIZE * (split_block_idx+1), ips), block_k)
  lower_bound = pl.cdiv(LEN_BLOCK_SIZE * split_block_idx, block_k)

  o = lax.fori_loop(lower_bound, upper_bound, body, o)

  curr_st_slice_padding = pl.dslice(start_x * block_h_padding, block_h_padding)

  o = o.astype(o_ref.dtype)
  pl.store(o_ref, (curr_st_slice_padding, pl.dslice(None)), o, mask=x_mask[:, None])


def kernel_1(
    x,
    ips,
    block_h: int,
    block_k: int,
):
  fist_dim, dim = x.shape

  block_h = min(block_h, fist_dim)
  splits = pl.cdiv(fist_dim, block_h)
  grid_ = (splits, 128)

  kernel = functools.partial(
      kernel_1_impl,
      block_k=block_k
  )

  half_dim = dim // 2
  kernel_in_specs = [
      pl.BlockSpec((block_h, dim), lambda i, j: (i, 0)),
      pl.BlockSpec((), lambda i, j: ()),
  ]
  kernel_out_specs=[
      pl.BlockSpec((LEN, half_dim), lambda i, j: (0, 0)),
      pl.BlockSpec((LEN, half_dim), lambda i, j: (0, 0)),
      pl.BlockSpec((LEN, 1), lambda i, j: (0, 0)),
      pl.BlockSpec((LEN, 1), lambda i, j: (0, 0)),
      pl.BlockSpec((None, block_h, dim), lambda i, j: (j, i, 0)), 
  ]
  kernel_out_shape=[
      jax.ShapeDtypeStruct(shape=(LEN, half_dim), dtype=jnp.int8), 
      jax.ShapeDtypeStruct(shape=(LEN, half_dim), dtype=jnp.int8), 
      jax.ShapeDtypeStruct(shape=(LEN, 1), dtype=jnp.float32), 
      jax.ShapeDtypeStruct(shape=(LEN, 1), dtype=jnp.float32), 
      jax.ShapeDtypeStruct(shape=(128, *x.shape), dtype=x.dtype), 
  ]

  outputs = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=kernel_in_specs,
      out_specs=kernel_out_specs,
      compiler_params=plgpu.TritonCompilerParams(
          num_warps=8, num_stages=1,
      ),
      out_shape=kernel_out_shape,
      name="kernel_1",
  )(x, ips)
  
  o = outputs[-1]
  return o[0, :, :]


@jax.jit
def entry_kernel(
    x,
    ips,
):
  first_dim, second_heads, last_dim = x.shape
  x_reshaped = x.reshape(first_dim, 2, 2, last_dim)


  inner = functools.partial(
      kernel_1,
      block_h=32,
      block_k=128,
  )

  with_kv_heads = jax.vmap(inner, in_axes=(0, None))
  o = jax.vmap(with_kv_heads,  in_axes=(0, 0))(
    x_reshaped, ips)

  return o.reshape(first_dim, second_heads, last_dim)

k1 = jax.random.key(0)
x =  jax.random.normal(k1, (4, 4, 128), dtype=jnp.bfloat16)
ips = jnp.ones(4, dtype=jnp.int32) * ST

impl = functools.partial(
    entry_kernel,
    ips=ips
)

for _ in range(20):
    out = entry_kernel(x, ips)
out = jax.block_until_ready(out)

