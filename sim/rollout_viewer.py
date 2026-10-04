#!/usr/bin/env python3
"""Play back a Newton rollout as an animated WebGL page.

    .venv-sam2/bin/python sim/rollout_viewer.py out/rollout.npz -o preview/rollout.html

Only the surface of each organ is drawn. tetgen keeps the input surface vertices
as the first nodes of the tet mesh, so the surface triangles index straight into
the particle array -- the mapping is asserted, not assumed.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
COLOURS = {}  # empty -> use the photographed vertex colour


def organ_surface(name: str):
    """Surface faces plus how many leading tet nodes they index into."""
    tet = np.load(ROOT / "tet" / f"{name}.npz")
    faces = tet["surface_faces"]
    surf = trimesh.load(ROOT / "mesh" / f"organ_{name}.ply", process=False)
    n_surf = len(surf.vertices)
    verts_m = np.asarray(surf.vertices) * 0.001

    # tetgen preserves input vertices in order; verify before trusting it
    lead = tet["vertices"][:n_surf]
    err = np.abs(lead - verts_m).max()
    if err > 1e-6:
        raise SystemExit(
            f"{name}: tet nodes do not start with the surface vertices (max err {err:.2e} m). "
            "Re-export with a tetgen build that preserves input point order."
        )
    return faces.astype(np.uint32), n_surf


HTML = """<title>%(title)s</title>
<style>
  :root{color-scheme:dark}
  html,body{margin:0;height:100%%;background:#0d0f12;color:#d8dee6;
    font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  #c{display:block;width:100%%;height:100%%;cursor:grab;touch-action:none}
  #c:active{cursor:grabbing}
  #hud{position:fixed;top:14px;left:14px;background:rgba(16,19,24,.86);padding:12px 14px;
    border-radius:9px;border:1px solid #262c36;max-width:290px;backdrop-filter:blur(8px)}
  h1{margin:0 0 7px;font-size:14px;font-weight:600}
  .m{color:#8a94a3;font-size:12px;margin:2px 0}.k{color:#e6b980}
  .warn{margin-top:9px;padding:7px 9px;background:#2a1f14;border:1px solid #4a3520;
    border-radius:6px;color:#d3a86a;font-size:11.5px;line-height:1.45}
  #bar{position:fixed;bottom:0;left:0;right:0;background:rgba(13,15,18,.94);
    border-top:1px solid #262c36;padding:10px 14px;display:flex;gap:11px;align-items:center}
  button{background:#1d222b;color:#d8dee6;border:1px solid #303845;border-radius:6px;
    padding:6px 13px;font-size:12px;cursor:pointer}
  button:hover{background:#262d38}
  input[type=range]{flex:1;accent-color:#e6b980}
  #t{color:#8a94a3;font:12px ui-monospace,monospace;min-width:120px;text-align:right}
</style>
<canvas id=c></canvas>
<div id=hud>
  <h1>%(title)s</h1>
  <div class=m>frames <span class=k>%(nf)s</span> @ <span class=k>%(fps)s</span> fps</div>
  <div class=m>particles <span class=k>%(np)s</span></div>
  <div class=m>%(organs)s</div>
  <div class=warn>%(warning)s</div>
</div>
<div id=bar>
  <button id=play>pause</button>
  <input type=range id=scrub min=0 max="%(last)d" value=0 step=1>
  <span id=t></span>
</div>
<script>
const POS="%(pos)s", IDX="%(idx)s", COL="%(col)s";
const NF=%(nf)d, NV=%(nv)d, NIDX=%(nidx)d, FPS=%(fps)s;
const CENTER=%(center)s, RADIUS=%(radius)s, QS=%(qscale)s, QO=%(qorigin)s;

function unb64(s){const r=atob(s),a=new Uint8Array(r.length);
  for(let i=0;i<r.length;i++)a[i]=r.charCodeAt(i);return a;}
const qpos=new Int16Array(unb64(POS).buffer);
const idx=new Uint32Array(unb64(IDX).buffer);
const col=unb64(COL);

const cv=document.getElementById('c');
const gl=cv.getContext('webgl2',{antialias:true,alpha:false});
const vs=`#version 300 es
in vec3 p; in vec3 c; uniform mat4 mvp; out vec3 vp; out vec3 vc;
void main(){ gl_Position=mvp*vec4(p,1.0); vp=p; vc=c; }`;
const fs=`#version 300 es
precision highp float; in vec3 vp; in vec3 vc; out vec4 o;
void main(){
  vec3 n=normalize(cross(dFdx(vp),dFdy(vp)));
  vec3 key=normalize(vec3(0.4,-0.6,0.7));
  float d=abs(dot(n,key))*0.8+0.25;
  o=vec4(vc*d,1.0);
}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const pr=gl.createProgram();
gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(pr);gl.useProgram(pr);

const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);
gl.bufferData(gl.ARRAY_BUFFER,NV*12,gl.DYNAMIC_DRAW);
let l=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,0,0);
const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);
gl.bufferData(gl.ARRAY_BUFFER,col,gl.STATIC_DRAW);
l=gl.getAttribLocation(pr,'c');gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.UNSIGNED_BYTE,true,0,0);
const eb=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,eb);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);
gl.enable(gl.DEPTH_TEST);gl.clearColor(0.051,0.059,0.071,1);
const uMvp=gl.getUniformLocation(pr,'mvp');

const scratch=new Float32Array(NV*3);
function upload(f){
  const off=f*NV*3;
  for(let i=0;i<NV*3;i++) scratch[i]=qpos[off+i]*QS+QO[i%%3];
  gl.bindBuffer(gl.ARRAY_BUFFER,pb);
  gl.bufferSubData(gl.ARRAY_BUFFER,0,scratch);
}

let yaw=0.5,pitch=-0.3,dist=RADIUS*3.2,frame=0,playing=true;
function mul(a,b){const o=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
    for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function draw(){
  const w=cv.width,h=cv.height;
  gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const f=1/Math.tan(0.45),a=w/h,near=RADIUS*0.05,far=RADIUS*40;
  const P=new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1,
                            0,0,2*far*near/(near-far),0]);
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const R=new Float32Array([cy,sp*sy,-cp*sy,0, 0,cp,sp,0, sy,-sp*cy,cp*cy,0, 0,0,0,1]);
  const T=new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, -CENTER[0],-CENTER[1],-CENTER[2],1]);
  const V=new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-dist,1]);
  gl.uniformMatrix4fv(uMvp,false,mul(P,mul(V,mul(R,T))));
  gl.drawElements(gl.TRIANGLES,NIDX,gl.UNSIGNED_INT,0);
}
const scrub=document.getElementById('scrub'),tl=document.getElementById('t');
function show(f){frame=f;upload(f);scrub.value=f;
  tl.textContent=`${String(f).padStart(4)} / ${NF-1}   ${(f/FPS).toFixed(2)}s`;draw();}
function resize(){const d=Math.min(devicePixelRatio||1,2);
  cv.width=innerWidth*d;cv.height=(innerHeight-46)*d;show(frame);}
addEventListener('resize',resize);

let drag=null;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('pointermove',e=>{if(!drag)return;
  yaw+=(e.clientX-drag.x)*0.008;
  pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag.y)*0.008));
  drag.x=e.clientX;drag.y=e.clientY;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();
  dist=Math.max(RADIUS*0.4,Math.min(RADIUS*12,dist*Math.exp(e.deltaY*0.0012)));draw();},{passive:false});
scrub.addEventListener('input',e=>{playing=false;
  document.getElementById('play').textContent='play';show(+e.target.value);});
document.getElementById('play').onclick=e=>{playing=!playing;
  e.target.textContent=playing?'pause':'play';};

let acc=0,last=performance.now();
function loop(now){
  acc+=(now-last)/1000;last=now;
  if(playing&&acc>1/FPS){acc=0;show((frame+1)%%NF);}
  requestAnimationFrame(loop);
}
resize();requestAnimationFrame(loop);
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout")
    ap.add_argument("-o", "--out", default="preview/rollout.html")
    ap.add_argument("--title", default="Newton rollout — rabbit organs")
    args = ap.parse_args()

    data = np.load(args.rollout, allow_pickle=True)
    traj = data["trajectory"]  # (frames, particles, 3), metres
    fps = float(data["fps"])
    organs = [(str(n), int(a), int(b)) for n, a, b in data["organs"]]

    verts, faces, cols = [], [], []
    base = 0
    labels = []
    for name, a, _b in organs:
        f, n_surf = organ_surface(name)
        faces.append(f + base)
        if name in COLOURS:
            cols.append(np.tile(np.array(COLOURS[name], np.uint8), (n_surf, 1)))
        else:
            sm = trimesh.load(ROOT / "mesh" / f"organ_{name}.ply", process=False)
            cols.append(np.asarray(sm.visual.vertex_colors)[:, :3].astype(np.uint8))
        verts.append((a, n_surf))
        labels.append(f"{name} {n_surf} verts / {len(f)} tris")
        base += n_surf

    sel = np.concatenate([np.arange(a, a + n) for a, n in verts])
    surf_traj = traj[:, sel, :]  # (frames, nv, 3)
    F = np.concatenate(faces).astype(np.uint32)
    C = np.concatenate(cols)

    lo = surf_traj.reshape(-1, 3).min(0)
    hi = surf_traj.reshape(-1, 3).max(0)
    qscale = float((hi - lo).max() / 32000.0) or 1e-6
    q = np.rint((surf_traj - lo) / qscale).astype(np.int16)

    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)

    html = HTML % {
        "title": args.title,
        "nf": len(traj),
        "last": len(traj) - 1,
        "np": f"{traj.shape[1]:,}",
        "nv": surf_traj.shape[1],
        "nidx": F.size,
        "fps": repr(fps),
        "organs": " · ".join(labels),
        "warning": "Material is literature soft tissue (E=20 kPa), not identified from this "
                   "recording. Organ back sides are inflated from the silhouette, not observed.",
        "pos": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode(),
        "idx": base64.b64encode(np.ascontiguousarray(F).tobytes()).decode(),
        "col": base64.b64encode(np.ascontiguousarray(C).tobytes()).decode(),
        "center": json.dumps([round(float(x), 5) for x in centre]),
        "radius": repr(round(radius, 5)),
        "qscale": repr(qscale),
        "qorigin": json.dumps([round(float(x), 6) for x in lo]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{len(traj)} frames, {surf_traj.shape[1]} surface verts, {F.size//3} tris")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
