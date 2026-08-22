import {
  planAceFixed32TiledFullAttention,
  type AceAttentionShape,
} from "./attention.js";
import { planAceOpt0039Attention } from "./attention-dual-query.js";

export const ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE = 256;
export const ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE = 32;
export const ACE_OPT_0061_ATTENTION_HEAD_DIMENSION = 128;
export const ACE_OPT_0061_ATTENTION_SUBGROUPS_PER_WORKGROUP = 8;
export const ACE_OPT_0061_ATTENTION_QUERY_HEADS_PER_KV_HEAD = 2;
export const ACE_OPT_0061_ATTENTION_TOKEN_STRIDE = 4;
export const ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES = 1_024;
export const ACE_OPT_0061_ATTENTION_BARRIERS_PER_KEY = 2;

export type AceOpt0061StreamCount = 3 | 4;

export interface AceOpt0061AttentionPlan {
  readonly shape: ReturnType<typeof planAceFixed32TiledFullAttention>;
  readonly streamCount: AceOpt0061StreamCount;
  readonly workgroupSize: typeof ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE;
  readonly queryTokensPerSubgroup: AceOpt0061StreamCount;
  readonly queriesPerWorkgroup: number;
  readonly queryTokensPerTile: number;
  readonly queryTokenTiles: number;
  readonly workgroupCount: number;
  readonly query8WorkgroupCount: number;
  readonly dualQueryWorkgroupCount: number;
  readonly keyValueScalarLoads: number;
  readonly query8KeyValueScalarLoads: number;
  readonly dualQueryKeyValueScalarLoads: number;
  readonly keyValueLoadReductionVersusQuery8: number;
  readonly keyValueLoadReductionVersusDualQuery: number;
  readonly barriersPerWorkgroup: number;
  readonly barrierEvents: number;
  readonly query8BarrierEvents: number;
  readonly dualQueryBarrierEvents: number;
  readonly barrierEventReductionVersusQuery8: number;
  readonly barrierEventReductionVersusDualQuery: number;
  readonly retainedPrivateFp32ValuesPerLane: number;
  readonly workgroupStorageBytes:
    typeof ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES;
  readonly outputElements: number;
}

export function planAceOpt0061Attention(
  shape: AceAttentionShape,
  streamCount: AceOpt0061StreamCount,
): AceOpt0061AttentionPlan {
  if (streamCount !== 3 && streamCount !== 4) {
    throw new RangeError("OPT-0061 supports exactly three or four streams per subgroup");
  }
  if (
    shape.batch !== 1 ||
    shape.queryHeads !== 16 ||
    shape.keyValueHeads !== 8 ||
    shape.queryTokens !== 2_250 ||
    shape.keyValueTokens !== 2_250 ||
    shape.headDimension !== ACE_OPT_0061_ATTENTION_HEAD_DIMENSION ||
    shape.mode !== "full" ||
    (shape.keyValidity ?? "none") !== "none"
  ) {
    throw new RangeError(
      "OPT-0061 requires exact B1/Hq16/Hkv8/M2250/D128 unmasked full attention",
    );
  }
  const base = planAceFixed32TiledFullAttention(shape);
  if (
    base.queryHeadsPerKeyValueHead !==
      ACE_OPT_0061_ATTENTION_QUERY_HEADS_PER_KV_HEAD
  ) {
    throw new RangeError("OPT-0061 requires GQA2 ownership");
  }
  const dual = planAceOpt0039Attention(shape);
  const queryTokensPerTile = ACE_OPT_0061_ATTENTION_TOKEN_STRIDE * streamCount;
  const queriesPerWorkgroup =
    ACE_OPT_0061_ATTENTION_SUBGROUPS_PER_WORKGROUP * streamCount;
  const queryTokenTiles = Math.ceil(base.queryTokens / queryTokensPerTile);
  const workgroupCount = checkedProduct(
    checkedProduct(base.batch, base.keyValueHeads, "batch/KV-head groups"),
    queryTokenTiles,
    "multi-query workgroups",
  );
  const keyValueScalarsPerWorkgroup = checkedProduct(
    base.keyValueTokens,
    base.headDimension * 2,
    "K/V scalars per workgroup",
  );
  const keyValueScalarLoads = checkedProduct(
    workgroupCount,
    keyValueScalarsPerWorkgroup,
    "multi-query K/V scalar loads",
  );
  const barriersPerWorkgroup = checkedProduct(
    base.keyValueTokens,
    ACE_OPT_0061_ATTENTION_BARRIERS_PER_KEY,
    "barriers per workgroup",
  );
  const barrierEvents = checkedProduct(
    workgroupCount,
    barriersPerWorkgroup,
    "multi-query barrier events",
  );
  return Object.freeze({
    shape: base,
    streamCount,
    workgroupSize: ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE,
    queryTokensPerSubgroup: streamCount,
    queriesPerWorkgroup,
    queryTokensPerTile,
    queryTokenTiles,
    workgroupCount,
    query8WorkgroupCount: base.workgroupCount,
    dualQueryWorkgroupCount: dual.workgroupCount,
    keyValueScalarLoads,
    query8KeyValueScalarLoads: base.tiledKeyValueScalarLoads,
    dualQueryKeyValueScalarLoads: dual.keyValueScalarLoads,
    keyValueLoadReductionVersusQuery8:
      base.tiledKeyValueScalarLoads / keyValueScalarLoads,
    keyValueLoadReductionVersusDualQuery:
      dual.keyValueScalarLoads / keyValueScalarLoads,
    barriersPerWorkgroup,
    barrierEvents,
    query8BarrierEvents: checkedProduct(
      base.workgroupCount,
      barriersPerWorkgroup,
      "query8 barrier events",
    ),
    dualQueryBarrierEvents: dual.barrierEvents,
    barrierEventReductionVersusQuery8:
      checkedProduct(
        base.workgroupCount,
        barriersPerWorkgroup,
        "query8 barrier events",
      ) / barrierEvents,
    barrierEventReductionVersusDualQuery: dual.barrierEvents / barrierEvents,
    retainedPrivateFp32ValuesPerLane: streamCount * 10,
    workgroupStorageBytes: ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES,
    outputElements: base.outputElements,
  });
}

export function aceOpt0061AttentionWgsl(
  shape: AceAttentionShape,
  streamCount: AceOpt0061StreamCount,
): string {
  const plan = planAceOpt0061Attention(shape, streamCount);
  const base = plan.shape;
  const scale = 1 / Math.sqrt(base.headDimension);
  const streamIds = Array.from({ length: streamCount }, (_, index) => index);
  const streamDeclarations = streamIds.map((stream) => `
  let query_token_${stream} = query_token_base + ${stream * 4}u;
  let query_${stream}_is_in_tensor = query_token_${stream} < QUERY_TOKENS;
  let query_${stream}_is_valid = query_token_${stream} < valid_query_tokens;
  let query_base_${stream} =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token_${stream}) *
    HEAD_DIMENSION;
  var query_values_${stream}: array<f32, 4>;
  var weighted_values_${stream}: array<f32, 4>;
  for (var chunk = 0u; chunk < 4u; chunk += 1u) {
    let dimension = subgroup_lane + chunk * 32u;
    query_values_${stream}[chunk] = 0.0;
    weighted_values_${stream}[chunk] = 0.0;
    if (query_${stream}_is_valid) {
      query_values_${stream}[chunk] = query[query_base_${stream} + dimension];
    }
  }`).join("\n");
  const zeroStores = streamIds.map((stream) => `
    if (query_${stream}_is_in_tensor) {
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        output[query_base_${stream} + subgroup_lane + chunk * 32u] = 0.0;
      }
    }`).join("\n");
  const onlineDeclarations = streamIds.map((stream) => `
  var online_max_${stream} = -3.4028234663852886e38;
  var online_denominator_${stream} = 0.0;`).join("\n");
  const streamUpdates = streamIds.map((stream) => `
    var dot_partial_${stream} = 0.0;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      dot_partial_${stream} =
        dot_partial_${stream} + query_values_${stream}[chunk] * key_tile[dimension];
    }
    let score_${stream} = subgroupAdd(dot_partial_${stream}) * ATTENTION_SCALE;
    let new_max_${stream} = max(online_max_${stream}, score_${stream});
    let online_alpha_${stream} = exp(online_max_${stream} - new_max_${stream});
    let online_beta_${stream} = exp(score_${stream} - new_max_${stream});
    online_denominator_${stream} =
      online_denominator_${stream} * online_alpha_${stream} + online_beta_${stream};
    online_max_${stream} = new_max_${stream};
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      weighted_values_${stream}[chunk] =
        weighted_values_${stream}[chunk] * online_alpha_${stream} +
        online_beta_${stream} * value_tile[dimension];
    }`).join("\n");
  const finalStores = streamIds.map((stream) => `
  if (query_${stream}_is_in_tensor) {
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * 32u;
      output[query_base_${stream} + dimension] = select(
        0.0,
        weighted_values_${stream}[chunk] / online_denominator_${stream},
        query_${stream}_is_valid && online_denominator_${stream} > 0.0,
      );
    }
  }`).join("\n");

  return /* wgsl */ `
// OPT-0061 fixed-WG256 ${streamCount}-query-stream full attention
// reduction-semantics: independent query8 streams in ascending key order
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

@compute @workgroup_size(${ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE}u) {
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
  let query_token_base =
    query_tile * QUERY_TOKENS_PER_TILE + subgroup % 4u;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
${streamDeclarations}

  if (query_tile * QUERY_TOKENS_PER_TILE >= valid_query_tokens ||
      valid_key_tokens == 0u) {${zeroStores}
    return;
  }

  let kv_batch_base =
    (batch * KV_HEADS + kv_head) * KV_TOKENS * HEAD_DIMENSION;
${onlineDeclarations}

  for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u) {
    if (local_lane < HEAD_DIMENSION) {
      let kv_index = kv_batch_base + key_token * HEAD_DIMENSION + local_lane;
      key_tile[local_lane] = key[kv_index];
      value_tile[local_lane] = value[kv_index];
    }
    workgroupBarrier();
${streamUpdates}
    workgroupBarrier();
  }
${finalStores}
}
`;
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0061 ${label} is not a safe integer`);
  }
  return product;
}
