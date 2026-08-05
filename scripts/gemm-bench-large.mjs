import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
const dev = await getDevice(); const ctx = new GpuContext(dev);
const rand=(n)=>Float32Array.from({length:n},()=>(Math.random()*2-1)*0.05);
const done=()=>dev.queue.onSubmittedWorkDone();
function cpuMM(a,b,M,K,N){const o=new Float32Array(M*N);for(let i=0;i<M;i++)for(let k=0;k<K;k++){const av=a[i*K+k];for(let j=0;j<N;j++)o[i*N+j]+=av*b[k*N+j];}return o;}
{const M=136,K=64,N=132;const A=rand(M*K),B=rand(K*N);const ga=ctx.upload(A,M,K),gb=ctx.upload(B,K,N);
 const g4=await ctx.download(ctx.matmulV4(ga,gb));const cpu=cpuMM(A,B,M,K,N);
 let e4=0;for(let i=0;i<M*N;i++)e4=Math.max(e4,Math.abs(g4[i]-cpu[i]));
 console.log(`v4 parity vs CPU (${M}x${K}x${N}): maxΔ ${e4.toExponential(2)}`);}
async function t(fn,name,n,iters=30){
  const a=ctx.upload(rand(n*n),n,n),b=ctx.upload(rand(n*n),n,n);
  for(let w=0;w<5;w++)fn(a,b); await done();
  const t0=performance.now(); for(let i=0;i<iters;i++)fn(a,b); await done();
  const ms=(performance.now()-t0)/iters;
  console.log(`  ${name} ${n}^3: ${ms.toFixed(3)} ms  ${(2*n**3/(ms/1e3)/1e12).toFixed(2)} TFLOP/s  (${(2*n**3/(ms/1e3)/1e12/5.87*100).toFixed(0)}% of MLX)`);
}
console.log("v1 vs v3 vs v4 (vs MLX 5.87):");
for(const n of [2048,4096]){await t((a,b)=>ctx.matmul(a,b),"v1",n);await t((a,b)=>ctx.matmulV3(a,b),"v3",n);await t((a,b)=>ctx.matmulV4(a,b),"v4",n);}
process.exit(0);
