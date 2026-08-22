# OPT-0060 — DiT dual-query full-attention integration

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: exact ascending-key FP32 online-softmax owner/routing change

## First-principles basis

OPT-0039 demonstrated a distinct exact full-attention owner on the target
browser. Keeping WG256 while assigning each fixed-32 subgroup two queries for
the same head halved workgroups (`4,504 -> 2,256`) and reduced K/V loads and
barrier events by `1.99645x`. All `13,824,000` compared FP32 words were
bit-identical, deterministic, finite, complete, and canary-safe. Balanced
median primitive wall improved `132.2 -> 97.4 ms` (`1.3572895x`), clearing the
frozen `1.25x` gate.

This exact mechanism is independent of dense precision, timestep count, VAE
layouts, and window geometry. It does not require another model package: query,
key, value, and output tensors retain their existing layouts and FP32 online
softmax order. The remaining uncertainty is production graph attribution and
resource ownership, not primitive arithmetic.

## Frozen integration direction

- Add a distinct fixed32 diagnostic DiT profile that changes only full
  self-attention from current query8 to OPT-0039 dual-query16. Sliding-local
  attention, cross attention, every dense/norm/residual/plumbing owner, graph
  order, sampler/DCW, package identity, activation storage, command batching,
  and production default remain unchanged.
- Route only exact M2250 full-self-attention labels/shapes. Keep WG256, eight
  fixed-32 subgroups, 1,024 bytes of workgroup storage, ascending-key online
  max/denominator/value updates, and FP32 output. Fail closed on feature,
  subgroup, label, shape, head geometry, or storage mismatch.
- Cache bounded pipelines/bind groups, expose the actual owner in graph
  descriptors and diagnostics, retain cancellation/progress bounds, and prove
  exactly-once resource destruction. Add no fallback that silently changes
  arithmetic or ownership.

## Gates

1. At the production-layer boundary, require raw-U32 identity between query8
   and dual-query16 for complete full-attention outputs across boundary/tail
   queries, deterministic repeats, finite values, guards, and clean cleanup.
   Reconcile exact operation, workgroup, K/V-load, barrier, command-buffer, and
   drain counts.
2. Run a short balanced repeated-layer/evaluation-slice screen after one
   nominal thermal check. Require both directional comparisons to improve the
   homogeneous full-attention family, median family speedup at least `1.20x`,
   and no regression in the enclosing layer/evaluation wall. Report GPU
   timestamps and fenced wall separately.
3. Before production selection, run the authenticated M2250 graph to the
   detached final latent. Require raw-U32 identity, unchanged final hash,
   identical graph math/topology outside the selected owner, cancellation,
   no device loss, and zero live resources. A combined final integration gate
   may include separately approved dense and sampler changes only if each
   owner's attribution remains independently recoverable.

Because the arithmetic and final latent must be bit-exact, no listening gate
is required for this mechanism alone. OPT-0039's primitive ratio is not an
end-to-end projection, and this experiment makes no product-speed or
under-one-minute claim until the actual integrated walls are measured.

## Closeout — duplicate registration caught before implementation

The ledger audit after registration found that OPT-0045 had already frozen the
same production integration direction, routing boundary, primitive authority,
and graph/final-latent gates. No OPT-0060 performance code, profile, package,
browser run, GPU work, result, or production change was started. Repeating the
unchanged experiment would violate the ledger's revisit rule, so this later
registration is retained for provenance but superseded by OPT-0045. All future
dual-query integration work and evidence remain under OPT-0045.
