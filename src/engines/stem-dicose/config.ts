// Production model package for DiCoSe stem separation: the converter's
// manifest.json (1.25 MB) + single weights.f16.bin (623 MB), mirrored on the
// FluidInference HF repo. The loader resolves the weight file relative to the
// manifest URL, so a base directory is all an origin needs to provide.
export const DICOSE_DEFAULT_BASE_URL = "https://huggingface.co/FluidInference/fluidaudio-web/resolve/main/dicose";
