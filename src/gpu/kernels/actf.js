// Shared activation table for all GEMM/conv kernels (interpolated into the
// WGSL template strings). ONE copy: adding an act here reaches every kernel;
// there are no per-route act allowlists to keep in sync.
// 1=gelu(tanh) 2=tanh 3=relu 4=silu 5=gelu_erf; unknown -> identity.
export const WGSL_ACTF = `
fn actf_gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn actf_erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x);
  return select(-y, y, x >= 0.0);
}
fn actf(x: f32, a: u32) -> f32 {
  if (a == 1u) { return actf_gelu(x); }
  if (a == 2u) { return tanh(clamp(x, -20.0, 20.0)); }
  if (a == 3u) { return max(x, 0.0); }
  if (a == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  if (a == 5u) { return 0.5 * x * (1.0 + actf_erf(x * 0.70710678118654752)); }
  return x;
}`;

export const ACT = { none: 0, gelu: 1, tanh: 2, relu: 3, silu: 4, gelu_erf: 5 };
