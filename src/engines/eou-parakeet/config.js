// EOU FastConformer encoder config — shared between the engine and the CI smoke
// so the smoke always exercises the shipped configuration.
export const EOU_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 2, attLeft: 70 };
