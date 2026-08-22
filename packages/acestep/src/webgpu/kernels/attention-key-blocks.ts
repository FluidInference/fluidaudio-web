import {
  planAceFixed32TiledFullAttention,
  type AceAttentionShape,
} from "./attention.js";

export type AceOpt0033AttentionKeyBlock = 8 | 16;

export const ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE = 256;
export const ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE = 32;
export const ACE_OPT_0033_ATTENTION_QUERIES_PER_WORKGROUP = 8;
export const ACE_OPT_0033_ATTENTION_QUERY_TOKENS_PER_TILE = 4;
export const ACE_OPT_0033_ATTENTION_HEAD_DIMENSION = 128;
export const ACE_OPT_0033_ATTENTION_BARRIERS_PER_KEY_BLOCK = 2;

export interface AceOpt0033AttentionPlan {
  readonly shape: ReturnType<typeof planAceFixed32TiledFullAttention>;
  readonly keyBlock: AceOpt0033AttentionKeyBlock;
  readonly workgroupSize: typeof ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE;
  readonly queriesPerWorkgroup:
    typeof ACE_OPT_0033_ATTENTION_QUERIES_PER_WORKGROUP;
  readonly queryTokensPerTile:
    typeof ACE_OPT_0033_ATTENTION_QUERY_TOKENS_PER_TILE;
  readonly keyBlocksPerWorkgroup: number;
  readonly barriersPerWorkgroup: number;
  readonly rowStagedBarriersPerWorkgroup: number;
  readonly barrierReductionVersusRowStaged: number;
  readonly workgroupCount: number;
  readonly barrierEvents: number;
  readonly keyValueScalarLoads: number;
  readonly outputElements: number;
  readonly keyPanelElements: number;
  readonly valuePanelElements: number;
  readonly workgroupStorageBytes: 8_192 | 16_384;
}

export function planAceOpt0033Attention(
  shape: AceAttentionShape,
  keyBlock: AceOpt0033AttentionKeyBlock,
): AceOpt0033AttentionPlan {
  const base = planAceFixed32TiledFullAttention(shape);
  if (
    base.batch !== 1 ||
    base.queryHeads !== 16 ||
    base.keyValueHeads !== 8 ||
    base.queryTokens !== 2_250 ||
    base.keyValueTokens !== 2_250 ||
    base.headDimension !== ACE_OPT_0033_ATTENTION_HEAD_DIMENSION ||
    base.mode !== "full" ||
    base.keyValidity !== "none" ||
    base.queryHeadsPerKeyValueHead !== 2 ||
    (keyBlock !== 8 && keyBlock !== 16)
  ) {
    throw new RangeError(
      "OPT-0033 requires exact B1/Hq16/Hkv8/M2250/D128 unmasked full attention and key-block 8 or 16",
    );
  }
  const keyBlocksPerWorkgroup = Math.ceil(base.keyValueTokens / keyBlock);
  const barriersPerWorkgroup = checkedProduct(
    keyBlocksPerWorkgroup,
    ACE_OPT_0033_ATTENTION_BARRIERS_PER_KEY_BLOCK,
    "barriers per workgroup",
  );
  const rowStagedBarriersPerWorkgroup = checkedProduct(
    base.keyValueTokens,
    2,
    "row-staged barriers per workgroup",
  );
  const keyPanelElements = checkedProduct(
    keyBlock,
    base.headDimension,
    "key-panel elements",
  );
  const valuePanelElements = keyPanelElements;
  const workgroupStorageBytes = checkedProduct(
    keyPanelElements + valuePanelElements,
    Float32Array.BYTES_PER_ELEMENT,
    "workgroup-storage bytes",
  ) as 8_192 | 16_384;
  const keyValueScalarLoads = checkedProduct(
    checkedProduct(base.workgroupCount, base.keyValueTokens, "key visits"),
    base.headDimension * 2,
    "K/V scalar loads",
  );
  return Object.freeze({
    shape: base,
    keyBlock,
    workgroupSize: ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE,
    queriesPerWorkgroup: ACE_OPT_0033_ATTENTION_QUERIES_PER_WORKGROUP,
    queryTokensPerTile: ACE_OPT_0033_ATTENTION_QUERY_TOKENS_PER_TILE,
    keyBlocksPerWorkgroup,
    barriersPerWorkgroup,
    rowStagedBarriersPerWorkgroup,
    barrierReductionVersusRowStaged:
      rowStagedBarriersPerWorkgroup / barriersPerWorkgroup,
    workgroupCount: base.workgroupCount,
    barrierEvents: checkedProduct(
      base.workgroupCount,
      barriersPerWorkgroup,
      "barrier events",
    ),
    keyValueScalarLoads,
    outputElements: base.outputElements,
    keyPanelElements,
    valuePanelElements,
    workgroupStorageBytes,
  });
}

export function aceOpt0033AttentionWgsl(
  shape: AceAttentionShape,
  keyBlock: AceOpt0033AttentionKeyBlock,
): string {
  const plan = planAceOpt0033Attention(shape, keyBlock);
  const base = plan.shape;
  const scale = 1 / Math.sqrt(base.headDimension);
  return /* wgsl */ `
// OPT-0033 query8 key-block${keyBlock}
// reduction-semantics: original ascending-key query8 online softmax
enable subgroups;

const QUERY_HEADS: u32 = ${base.queryHeads}u;
const KV_HEADS: u32 = ${base.keyValueHeads}u;
const QUERY_TOKENS: u32 = ${base.queryTokens}u;
const KV_TOKENS: u32 = ${base.keyValueTokens}u;
const HEAD_DIMENSION: u32 = ${base.headDimension}u;
const HEADS_PER_KV: u32 = ${base.queryHeadsPerKeyValueHead}u;
const QUERY_TOKENS_PER_TILE: u32 = ${base.queryTokensPerTile}u;
const QUERY_TOKEN_TILES: u32 = ${base.queryTokenTiles}u;
const TOTAL_WORKGROUPS: u32 = ${base.workgroupCount}u;
const KEY_BLOCK: u32 = ${keyBlock}u;
const KEY_PANEL_ELEMENTS: u32 = ${plan.keyPanelElements}u;
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

var<workgroup> key_tile: array<f32, ${plan.keyPanelElements}>;
var<workgroup> value_tile: array<f32, ${plan.valuePanelElements}>;

@compute @workgroup_size(${ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE}u) {
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
  let query_head =
    kv_head * HEADS_PER_KV + subgroup / QUERY_TOKENS_PER_TILE;
  let query_token =
    query_tile * QUERY_TOKENS_PER_TILE + subgroup % QUERY_TOKENS_PER_TILE;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
  let query_is_in_tensor = query_token < QUERY_TOKENS;
  let query_is_valid = query_token < valid_query_tokens;
  let query_base =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token) *
    HEAD_DIMENSION;

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

  for (
    var key_block_start = 0u;
    key_block_start < valid_key_tokens;
    key_block_start += KEY_BLOCK
  ) {
    // All lanes stage the complete ascending K/V block. Invalid tail rows are
    // initialized but never enter the arithmetic loop below.
    for (
      var panel_index = local_lane;
      panel_index < KEY_PANEL_ELEMENTS;
      panel_index += ${ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE}u
    ) {
      let block_key = panel_index / HEAD_DIMENSION;
      let dimension = panel_index % HEAD_DIMENSION;
      let key_token = key_block_start + block_key;
      var staged_key = 0.0;
      var staged_value = 0.0;
      if (key_token < valid_key_tokens) {
        let kv_index = kv_batch_base + key_token * HEAD_DIMENSION + dimension;
        staged_key = key[kv_index];
        staged_value = value[kv_index];
      }
      key_tile[panel_index] = staged_key;
      value_tile[panel_index] = staged_value;
    }
    workgroupBarrier();

    let valid_keys_in_block = min(KEY_BLOCK, valid_key_tokens - key_block_start);
    for (
      var key_in_block = 0u;
      key_in_block < valid_keys_in_block;
      key_in_block += 1u
    ) {
      let panel_base = key_in_block * HEAD_DIMENSION;
      var dot_partial = 0.0;
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        let dimension = subgroup_lane + chunk * 32u;
        dot_partial = dot_partial +
          query_values[chunk] * key_tile[panel_base + dimension];
      }
      let score = subgroupAdd(dot_partial) * ATTENTION_SCALE;
      let new_max = max(online_max, score);
      let online_alpha = exp(online_max - new_max);
      let online_beta = exp(score - new_max);
      online_denominator =
        online_denominator * online_alpha + online_beta;
      online_max = new_max;
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        let dimension = subgroup_lane + chunk * 32u;
        weighted_values[chunk] = weighted_values[chunk] * online_alpha +
          online_beta * value_tile[panel_base + dimension];
      }
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

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0033 ${label} is not a safe integer`);
  }
  return product;
}
