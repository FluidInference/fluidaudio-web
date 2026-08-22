import {
  planAceAttention,
  type AceAttentionShape,
} from "./attention.js";

export type AceOpt0030AttentionQueriesPerWorkgroup = 8 | 16 | 32;

export interface AceOpt0030AttentionPlan {
  readonly shape: ReturnType<typeof planAceAttention>;
  readonly queriesPerWorkgroup: AceOpt0030AttentionQueriesPerWorkgroup;
  readonly queryTokensPerTile: 4 | 8 | 16;
  readonly workgroupSize: 256 | 512 | 1024;
  readonly queryTokenTiles: number;
  readonly workgroupCount: number;
  readonly keyValueScalarLoads: number;
  readonly query8KeyValueScalarLoads: number;
  readonly loadReductionVersusQuery8: 1 | 2 | 4;
}

export function planAceOpt0030Attention(
  shape: AceAttentionShape,
  queriesPerWorkgroup: AceOpt0030AttentionQueriesPerWorkgroup,
): AceOpt0030AttentionPlan {
  const base = planAceAttention(shape);
  if (
    base.mode !== "full" ||
    base.keyValidity !== "none" ||
    base.queryHeadsPerKeyValueHead !== 2 ||
    base.headDimension !== 128 ||
    base.queryTokens < 2_250 ||
    !([8, 16, 32] as const).includes(queriesPerWorkgroup)
  ) {
    throw new RangeError("OPT-0030 requires long unmasked full GQA2 D128");
  }
  const queryTokensPerTile = (queriesPerWorkgroup / 2) as 4 | 8 | 16;
  const workgroupSize = (queriesPerWorkgroup * 32) as 256 | 512 | 1024;
  const queryTokenTiles = Math.ceil(base.queryTokens / queryTokensPerTile);
  const workgroupCount = base.batch * base.keyValueHeads * queryTokenTiles;
  const keyValueScalarLoads =
    base.batch * base.keyValueHeads * queryTokenTiles * base.keyValueTokens *
    base.headDimension * 2;
  const query8KeyValueScalarLoads =
    base.batch * base.keyValueHeads * Math.ceil(base.queryTokens / 4) *
    base.keyValueTokens * base.headDimension * 2;
  return Object.freeze({
    shape: base,
    queriesPerWorkgroup,
    queryTokensPerTile,
    workgroupSize,
    queryTokenTiles,
    workgroupCount,
    keyValueScalarLoads,
    query8KeyValueScalarLoads,
    loadReductionVersusQuery8: (queriesPerWorkgroup / 8) as 1 | 2 | 4,
  });
}

export function aceOpt0030AttentionWgsl(
  shape: AceAttentionShape,
  queriesPerWorkgroup: AceOpt0030AttentionQueriesPerWorkgroup,
): string {
  const plan = planAceOpt0030Attention(shape, queriesPerWorkgroup);
  const base = plan.shape;
  const scale = 1 / Math.sqrt(base.headDimension);
  return /* wgsl */ `
// OPT-0030 stock-WebGPU query${queriesPerWorkgroup}
enable subgroups;

const QUERY_HEADS: u32 = ${base.queryHeads}u;
const KV_HEADS: u32 = ${base.keyValueHeads}u;
const QUERY_TOKENS: u32 = ${base.queryTokens}u;
const KV_TOKENS: u32 = ${base.keyValueTokens}u;
const HEAD_DIMENSION: u32 = 128u;
const QUERY_TOKENS_PER_TILE: u32 = ${plan.queryTokensPerTile}u;
const QUERY_TOKEN_TILES: u32 = ${plan.queryTokenTiles}u;
const TOTAL_WORKGROUPS: u32 = ${plan.workgroupCount}u;
const ATTENTION_SCALE: f32 = ${scale};

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> valid_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> key_tile: array<f32, 128>;
var<workgroup> value_tile: array<f32, 128>;

@compute @workgroup_size(${plan.workgroupSize}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u || group.x >= TOTAL_WORKGROUPS) { return; }
  let groups_per_batch = KV_HEADS * QUERY_TOKEN_TILES;
  let batch = group.x / groups_per_batch;
  let within_batch = group.x % groups_per_batch;
  let kv_head = within_batch / QUERY_TOKEN_TILES;
  let query_tile = within_batch % QUERY_TOKEN_TILES;
  let query_head = kv_head * 2u + subgroup / QUERY_TOKENS_PER_TILE;
  let query_token =
    query_tile * QUERY_TOKENS_PER_TILE + subgroup % QUERY_TOKENS_PER_TILE;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
  let query_is_in_tensor = query_token < QUERY_TOKENS;
  let query_is_valid = query_token < valid_query_tokens;
  let query_base =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token) * 128u;

  var query_values: array<f32, 4>;
  var weighted_values: array<f32, 4>;
  for (var chunk = 0u; chunk < 4u; chunk += 1u) {
    let dimension = subgroup_lane + chunk * 32u;
    query_values[chunk] = 0.0;
    weighted_values[chunk] = 0.0;
    if (query_is_valid) {
      query_values[chunk] = query[query_base + dimension];
    }
  }
  if (query_tile * QUERY_TOKENS_PER_TILE >= valid_query_tokens ||
      valid_key_tokens == 0u) {
    if (query_is_in_tensor) {
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        output[query_base + subgroup_lane + chunk * 32u] = 0.0;
      }
    }
    return;
  }

  let kv_batch_base =
    (batch * KV_HEADS + kv_head) * KV_TOKENS * HEAD_DIMENSION;
  var online_max = -3.4028234663852886e38;
  var online_denominator = 0.0;
  for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u) {
    if (local_lane < 128u) {
      let kv_index = kv_batch_base + key_token * 128u + local_lane;
      key_tile[local_lane] = key[kv_index];
      value_tile[local_lane] = value[kv_index];
    }
    workgroupBarrier();
    var dot_partial = 0.0;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      dot_partial = dot_partial + query_values[chunk] * key_tile[dimension];
    }
    let score = subgroupAdd(dot_partial) * ATTENTION_SCALE;
    let new_max = max(online_max, score);
    let online_alpha = exp(online_max - new_max);
    let online_beta = exp(score - new_max);
    online_denominator = online_denominator * online_alpha + online_beta;
    online_max = new_max;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      weighted_values[chunk] = weighted_values[chunk] * online_alpha +
        online_beta * value_tile[dimension];
    }
    workgroupBarrier();
  }
  if (query_is_in_tensor) {
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      output[query_base + dimension] = select(
        0.0,
        weighted_values[chunk] / online_denominator,
        query_is_valid && online_denominator > 0.0,
      );
    }
  }
}
`;
}
