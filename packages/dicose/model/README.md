# Preparing the browser model

Downloaded checkpoints, caches, and packed weights are not committed. The
small audited manifest remains tracked. From the repository root, run the
preparation explicitly:

```sh
pnpm model:prepare
```

The command downloads only these files from `karchkha/DiCoSe` at revision
`b3e44147b96e55b08eea2dd0b6b4e017748a87a9`:

- `Deterministic_model_MSST_bs_roformer/model.ckpt`
- `CD_MSST_bs_roformer/model.ckpt`

Both source files are checked by byte length and SHA-256 before PyTorch opens
them. Conversion is staged transactionally, and the final manifest and 623 MB
f16 blob must match the canonical production hashes before replacing
`public/model/`. Downloads are resumable and retained in ignored
`model/cache/`.

For an already-downloaded pair, the lower-level converter accepts explicit
`--deterministic` and `--cd` paths. Run `pnpm model:test` for the no-download
preparation tests and `pnpm verify:package` to recheck an installed package.
