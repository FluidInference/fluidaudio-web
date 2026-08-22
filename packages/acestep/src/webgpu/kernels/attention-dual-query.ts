import {
  planAceFixed32TiledFullAttention,
  type AceAttentionShape,
} from "./attention.js";

export const ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE = 256;
export const ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE = 32;
export const ACE_OPT_0039_ATTENTION_HEAD_DIMENSION = 128;
export const ACE_OPT_0039_ATTENTION_QUERIES_PER_WORKGROUP = 16;
export const ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE = 8;
export const ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_SUBGROUP = 2;
export const ACE_OPT_0039_ATTENTION_WORKGROUP_STORAGE_BYTES = 1_024;
export const ACE_OPT_0039_ATTENTION_BARRIERS_PER_KEY = 2;

export interface AceOpt0039AttentionPlan {
  readonly shape: ReturnType<typeof planAceFixed32TiledFullAttention>;
  readonly workgroupSize: typeof ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE;
  readonly queriesPerWorkgroup:
    typeof ACE_OPT_0039_ATTENTION_QUERIES_PER_WORKGROUP;
  readonly queryTokensPerTile:
    typeof ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE;
  readonly queryTokensPerSubgroup:
    typeof ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_SUBGROUP;
  readonly queryTokenTiles: number;
  readonly workgroupCount: number;
  readonly query8WorkgroupCount: number;
  readonly keyValueScalarLoads: number;
  readonly query8KeyValueScalarLoads: number;
  readonly keyValueLoadReductionVersusQuery8: number;
  readonly barriersPerWorkgroup: number;
  readonly barrierEvents: number;
  readonly query8BarrierEvents: number;
  readonly barrierEventReductionVersusQuery8: number;
  readonly workgroupStorageBytes:
    typeof ACE_OPT_0039_ATTENTION_WORKGROUP_STORAGE_BYTES;
  readonly outputElements: number;
}

/**
 * Isolated OPT-0039 exact-shape screen. Production query8 remains unchanged.
 */
export function planAceOpt0039Attention(
  shape: AceAttentionShape,
): AceOpt0039AttentionPlan {
  if (
    shape.batch !== 1 ||
    shape.queryHeads !== 16 ||
    shape.keyValueHeads !== 8 ||
    shape.queryTokens !== 2_250 ||
    shape.keyValueTokens !== 2_250 ||
    shape.headDimension !== ACE_OPT_0039_ATTENTION_HEAD_DIMENSION ||
    shape.mode !== "full" ||
    (shape.keyValidity ?? "none") !== "none"
  ) {
    throw new RangeError(
      "OPT-0039 requires exact B1/Hq16/Hkv8/M2250/D128 unmasked full attention",
    );
  }
  const base = planAceFixed32TiledFullAttention(shape);
  if (
    base.batch !== 1 ||
    base.queryHeads !== 16 ||
    base.keyValueHeads !== 8 ||
    base.queryTokens !== 2_250 ||
    base.keyValueTokens !== 2_250 ||
    base.headDimension !== ACE_OPT_0039_ATTENTION_HEAD_DIMENSION ||
    base.mode !== "full" ||
    base.keyValidity !== "none" ||
    base.queryHeadsPerKeyValueHead !== 2
  ) {
    throw new RangeError(
      "OPT-0039 requires exact B1/Hq16/Hkv8/M2250/D128 unmasked full attention",
    );
  }
  const queryTokenTiles = Math.ceil(
    base.queryTokens / ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE,
  );
  const workgroupCount = checkedProduct(
    checkedProduct(base.batch, base.keyValueHeads, "batch/KV-head groups"),
    queryTokenTiles,
    "dual-query workgroups",
  );
  const keyValueScalarsPerWorkgroup = checkedProduct(
    base.keyValueTokens,
    base.headDimension * 2,
    "K/V scalars per workgroup",
  );
  const keyValueScalarLoads = checkedProduct(
    workgroupCount,
    keyValueScalarsPerWorkgroup,
    "dual-query K/V scalar loads",
  );
  const barriersPerWorkgroup = checkedProduct(
    base.keyValueTokens,
    ACE_OPT_0039_ATTENTION_BARRIERS_PER_KEY,
    "barriers per workgroup",
  );
  const barrierEvents = checkedProduct(
    workgroupCount,
    barriersPerWorkgroup,
    "dual-query barrier events",
  );
  const query8BarrierEvents = checkedProduct(
    base.workgroupCount,
    barriersPerWorkgroup,
    "query8 barrier events",
  );
  return Object.freeze({
    shape: base,
    workgroupSize: ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE,
    queriesPerWorkgroup: ACE_OPT_0039_ATTENTION_QUERIES_PER_WORKGROUP,
    queryTokensPerTile: ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE,
    queryTokensPerSubgroup:
      ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_SUBGROUP,
    queryTokenTiles,
    workgroupCount,
    query8WorkgroupCount: base.workgroupCount,
    keyValueScalarLoads,
    query8KeyValueScalarLoads: base.tiledKeyValueScalarLoads,
    keyValueLoadReductionVersusQuery8:
      base.tiledKeyValueScalarLoads / keyValueScalarLoads,
    barriersPerWorkgroup,
    barrierEvents,
    query8BarrierEvents,
    barrierEventReductionVersusQuery8: query8BarrierEvents / barrierEvents,
    workgroupStorageBytes: ACE_OPT_0039_ATTENTION_WORKGROUP_STORAGE_BYTES,
    outputElements: base.outputElements,
  });
}

/**
 * Fixed-WG256 dual-query attention.
 *
 * Each subgroup owns two query tokens for one query head. Both streams use
 * the query8 dot expression, subgroup reduction, and ascending-key FP32 online
 * softmax update independently. The only shared state is the K/V row loaded
 * once by the workgroup.
 */
export function aceOpt0039AttentionWgsl(shape: AceAttentionShape): string {
  const plan = planAceOpt0039Attention(shape);
  const base = plan.shape;
  const scale = 1 / Math.sqrt(base.headDimension);
  return /* wgsl */ `
// OPT-0039 fixed-WG256 dual-query16 full attention
// reduction-semantics: two independent query8 streams in ascending key order
enable subgroups;

const QUERY_HEADS: u32 = ${base.queryHeads}u;
const KV_HEADS: u32 = ${base.keyValueHeads}u;
const QUERY_TOKENS: u32 = ${base.queryTokens}u;
const KV_TOKENS: u32 = ${base.keyValueTokens}u;
const HEAD_DIMENSION: u32 = ${base.headDimension}u;
const HEADS_PER_KV: u32 = ${base.queryHeadsPerKeyValueHead}u;
const QUERY_TOKENS_PER_TILE: u32 = ${plan.queryTokensPerTile}u;
const QUERY_TOKEN_TILES: u32 = ${plan.queryTokenTiles}u;
const TOTAL_WORKGROUPS: u32 = ${plan.workgroupCount}u;
const ATTENTION_SCALE: f32 = ${scale};

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> valid_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

struct AttentionRangeParameters {
  first_workgroup: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}
@group(0) @binding(7) var<uniform>
  attention_range: AttentionRangeParameters;

var<workgroup> key_tile: array<f32, ${base.headDimension}>;
var<workgroup> value_tile: array<f32, ${base.headDimension}>;

@compute @workgroup_size(${ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE}u) {
    return;
  }
  let linear_group = attention_range.first_workgroup + group.x;
  if (linear_group >= TOTAL_WORKGROUPS) {
    return;
  }
  let groups_per_batch = KV_HEADS * QUERY_TOKEN_TILES;
  let batch = linear_group / groups_per_batch;
  let within_batch = linear_group % groups_per_batch;
  let kv_head = within_batch / QUERY_TOKEN_TILES;
  let query_tile = within_batch % QUERY_TOKEN_TILES;
  let query_head = kv_head * HEADS_PER_KV + subgroup / 4u;
  let query_token_0 = query_tile * QUERY_TOKENS_PER_TILE + subgroup % 4u;
  let query_token_1 = query_token_0 + 4u;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
  let query_0_is_in_tensor = query_token_0 < QUERY_TOKENS;
  let query_1_is_in_tensor = query_token_1 < QUERY_TOKENS;
  let query_0_is_valid = query_token_0 < valid_query_tokens;
  let query_1_is_valid = query_token_1 < valid_query_tokens;
  let query_base_0 =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token_0) *
    HEAD_DIMENSION;
  let query_base_1 =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token_1) *
    HEAD_DIMENSION;

  var query_values_0: array<f32, 4>;
  var query_values_1: array<f32, 4>;
  var weighted_values_0: array<f32, 4>;
  var weighted_values_1: array<f32, 4>;
  for (var chunk = 0u; chunk < 4u; chunk += 1u) {
    let dimension = subgroup_lane + chunk * 32u;
    query_values_0[chunk] = 0.0;
    query_values_1[chunk] = 0.0;
    weighted_values_0[chunk] = 0.0;
    weighted_values_1[chunk] = 0.0;
    if (query_0_is_valid) {
      query_values_0[chunk] = query[query_base_0 + dimension];
    }
    if (query_1_is_valid) {
      query_values_1[chunk] = query[query_base_1 + dimension];
    }
  }

  // This condition is workgroup-uniform. Per-stream validity remains a final
  // select so a partial tile cannot place a barrier in divergent control flow.
  if (query_tile * QUERY_TOKENS_PER_TILE >= valid_query_tokens ||
      valid_key_tokens == 0u) {
    if (query_0_is_in_tensor) {
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        output[query_base_0 + subgroup_lane + chunk * 32u] = 0.0;
      }
    }
    if (query_1_is_in_tensor) {
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        output[query_base_1 + subgroup_lane + chunk * 32u] = 0.0;
      }
    }
    return;
  }

  let kv_batch_base =
    (batch * KV_HEADS + kv_head) * KV_TOKENS * HEAD_DIMENSION;
  var online_max_0 = -3.4028234663852886e38;
  var online_max_1 = -3.4028234663852886e38;
  var online_denominator_0 = 0.0;
  var online_denominator_1 = 0.0;

  for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u) {
    if (local_lane < HEAD_DIMENSION) {
      let kv_index = kv_batch_base + key_token * HEAD_DIMENSION + local_lane;
      key_tile[local_lane] = key[kv_index];
      value_tile[local_lane] = value[kv_index];
    }
    workgroupBarrier();

    var dot_partial_0 = 0.0;
    var dot_partial_1 = 0.0;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      dot_partial_0 =
        dot_partial_0 + query_values_0[chunk] * key_tile[dimension];
      dot_partial_1 =
        dot_partial_1 + query_values_1[chunk] * key_tile[dimension];
    }

    let score_0 = subgroupAdd(dot_partial_0) * ATTENTION_SCALE;
    let new_max_0 = max(online_max_0, score_0);
    let online_alpha_0 = exp(online_max_0 - new_max_0);
    let online_beta_0 = exp(score_0 - new_max_0);
    online_denominator_0 =
      online_denominator_0 * online_alpha_0 + online_beta_0;
    online_max_0 = new_max_0;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      weighted_values_0[chunk] =
        weighted_values_0[chunk] * online_alpha_0 +
        online_beta_0 * value_tile[dimension];
    }

    let score_1 = subgroupAdd(dot_partial_1) * ATTENTION_SCALE;
    let new_max_1 = max(online_max_1, score_1);
    let online_alpha_1 = exp(online_max_1 - new_max_1);
    let online_beta_1 = exp(score_1 - new_max_1);
    online_denominator_1 =
      online_denominator_1 * online_alpha_1 + online_beta_1;
    online_max_1 = new_max_1;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      weighted_values_1[chunk] =
        weighted_values_1[chunk] * online_alpha_1 +
        online_beta_1 * value_tile[dimension];
    }
    workgroupBarrier();
  }

  if (query_0_is_in_tensor) {
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      output[query_base_0 + dimension] = select(
        0.0,
        weighted_values_0[chunk] / online_denominator_0,
        query_0_is_valid && online_denominator_0 > 0.0,
      );
    }
  }
  if (query_1_is_in_tensor) {
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      output[query_base_1 + dimension] = select(
        0.0,
        weighted_values_1[chunk] / online_denominator_1,
        query_1_is_valid && online_denominator_1 > 0.0,
      );
    }
  }
}
`;
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0039 ${label} is not a safe integer`);
  }
  return product;
}
