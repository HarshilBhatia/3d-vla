#!/usr/bin/env python3
"""
Interactive 3D visualization of camera positions from camera_render/.

Usage (from 3d-vla/):
    python scripts/rlbench/visualize_camera_positions.py \
        --zarr     Peract_zarr/val.zarr \
        --configs  camera_render/configs.json \
        --cam_dir  camera_render/ \
        --out      camera_render/viz.html
"""

import argparse, base64, io, json, math, os, re, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

# ── Point cloud helpers ──────────────────────────────────────────────────────

def load_point_cloud(zarr_path, n_frames=20, seed=0, max_pts_per_cam=1200):
    import zarr
    z = zarr.open_group(zarr_path, mode="r")
    total = z["rgb"].shape[0]
    rng = np.random.default_rng(seed)
    frames = sorted(rng.choice(total, size=min(n_frames, total), replace=False).tolist())
    has_pcd = "pcd" in z
    has_ext = "extrinsics" in z and "intrinsics" in z and "depth" in z
    if not has_pcd and not has_ext:
        sys.exit("[ERROR] Zarr missing pcd or extrinsics+depth.")
    all_pts, all_cols = [], []
    for fi in frames:
        rgb_e = z["rgb"][fi]
        if rgb_e.ndim == 4 and rgb_e.shape[0] > 6:
            rgb_e = rgb_e[0]
        ncam, _, H, W = rgb_e.shape
        if has_pcd:
            pcd_e = z["pcd"][fi]
            if pcd_e.ndim == 4 and pcd_e.shape[0] > 6:
                pcd_e = pcd_e[0]
            for c in range(ncam):
                pts_c = pcd_e[c].astype(np.float32).reshape(3,-1).T
                col_c = rgb_e[c].reshape(3,-1).T
                valid = np.isfinite(pts_c).all(1) & (pts_c[:,2]>0.005)
                idx = np.where(valid)[0]
                if len(idx)==0: continue
                if len(idx)>max_pts_per_cam:
                    idx = rng.choice(idx, max_pts_per_cam, replace=False)
                all_pts.append(pts_c[idx]); all_cols.append(col_c[idx])
        else:
            ext_e = z["extrinsics"][fi]; intr_e = z["intrinsics"][fi]; dep_e = z["depth"][fi]
            if ext_e.ndim==3 and ext_e.shape[0]>6:
                ext_e=ext_e[0]; intr_e=intr_e[0]; dep_e=dep_e[0]
            for c in range(ncam):
                K_c=intr_e[c].astype(np.float64); E_c=ext_e[c].astype(np.float64)
                d_c=dep_e[c].astype(np.float32); valid_d=d_c>0.01
                if valid_d.sum()<10: continue
                vi,ui=np.where(valid_d); Zv=d_c[vi,ui]
                Xv=(ui-K_c[0,2])/K_c[0,0]*Zv; Yv=(vi-K_c[1,2])/K_c[1,1]*Zv
                pts_cam=np.stack([Xv,Yv,Zv],1)
                pts_w=pts_cam@E_c[:3,:3].T+E_c[:3,3]
                col_c=rgb_e[c,:,vi,ui].T
                idx=np.arange(len(pts_w))
                if len(idx)>max_pts_per_cam:
                    idx=rng.choice(idx,max_pts_per_cam,replace=False)
                all_pts.append(pts_w[idx].astype(np.float32)); all_cols.append(col_c[idx])
    if not all_pts: sys.exit("[ERROR] No valid 3-D points.")
    pts=np.concatenate(all_pts,0); cols=np.concatenate(all_cols,0)
    print("[INFO] Loaded {:,} world points from {} frames".format(len(pts),len(frames)))
    return pts, cols

def estimate_workspace(pts):
    c=np.median(pts,0); r=float(np.percentile(np.linalg.norm(pts-c,axis=1),80))
    zt=float(np.percentile(pts[:,2],15))
    print("[INFO] Centre={}, radius={:.3f} m, table_z~{:.3f} m".format(c.round(3),r,zt))
    return c, r, zt

def make_K(fov=65.0, sz=256):
    f=(sz/2)/math.tan(math.radians(fov/2)); c=sz/2
    return np.array([[f,0,c],[0,f,c],[0,0,1]])

def render_view(pts, colors, pos, R, K, H=256, W=256):
    Xc=(pts-pos)@R; ok=Xc[:,2]>0.01; Xc=Xc[ok]; col=colors[ok]
    if len(Xc)==0: return np.zeros((H,W,3),dtype=np.uint8)
    u=np.round(K[0,0]*Xc[:,0]/Xc[:,2]+K[0,2]).astype(np.int32)
    v=np.round(K[1,1]*Xc[:,1]/Xc[:,2]+K[1,2]).astype(np.int32)
    d=Xc[:,2]; mask=(u>=1)&(u<W-1)&(v>=1)&(v<H-1)
    u,v,d,col=u[mask],v[mask],d[mask],col[mask]
    o=np.argsort(-d); u,v,col=u[o],v[o],col[o]
    img=np.zeros((H,W,3),dtype=np.uint8)
    for dv,du in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
        img[v+dv,u+du]=col
    return img

def parse_camera_info(path):
    txt=open(path).read(); res={}
    for key,pat in [("pos",r"pos_xyz\s*:\s*\[([^\]]+)\]"),("euler",r"euler_xyz\s*:\s*\[([^\]]+)\]"),("quat",r"quaternion\s*:\s*\[([^\]]+)\]")]:
        m=re.search(pat,txt)
        if m: res[key]=[float(v) for v in m.group(1).split(",")]
    for key,pat in [("roll_deg",r"roll_deg\s*:\s*([\-\d.]+)"),("dist_m",r"dist_m\s*:\s*([\d.]+)")]:
        m=re.search(pat,txt)
        if m: res[key]=float(m.group(1))
    rms=txt.find("rot_matrix")
    if rms>=0:
        rows=re.findall(r"\[([^\]]+)\]",txt[rms:])
        R=[[float(v) for v in row.split(",")] for row in rows[:3]]
        if len(R)==3: res["R"]=np.array(R)
    return res

def find_candidate_dirs(d):
    out={}
    for e in os.listdir(d):
        m=re.match(r"candidate_(\d{4})_(.+)",e)
        if m: out[m.group(2)]=os.path.join(d,e)
    return out

def img_to_b64(img):
    try:
        from PIL import Image as PI
        buf=io.BytesIO(); PI.fromarray(img).save(buf,"PNG",optimize=True)
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    except ImportError: pass
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1,figsize=(2,2)); ax.imshow(img); ax.axis("off")
    fig.tight_layout(pad=0); buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=64); plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

# ── HTML template (base64-encoded to avoid escaping issues) ───────────────────
import base64 as _b64
_HTML_B64 = """PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ij4KPHRpdGxlPjNEIENhbWVyYSBQb3NpdGlvbiBFeHBsb3JlcjwvdGl0bGU+CjxzdHlsZT4KKiwqOjpiZWZvcmUsKjo6YWZ0ZXJ7Ym94LXNpemluZzpib3JkZXItYm94O21hcmdpbjowO3BhZGRpbmc6MH0KYm9keXtiYWNrZ3JvdW5kOiMwYTBhMTQ7Y29sb3I6I2UwZTBmMDtmb250LWZhbWlseTonU2Vnb2UgVUknLHN5c3RlbS11aSxzYW5zLXNlcmlmO292ZXJmbG93OmhpZGRlbjtoZWlnaHQ6MTAwdmg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbn0KI3RvcGJhcntiYWNrZ3JvdW5kOiMxMTExMjg7Ym9yZGVyLWJvdHRvbToxcHggc29saWQgIzFlMWU0MDtwYWRkaW5nOjhweCAxNnB4O2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEycHg7ZmxleC1zaHJpbms6MDtmbGV4LXdyYXA6d3JhcH0KI3RvcGJhciBoMXtmb250LXNpemU6MWVtO2ZvbnQtd2VpZ2h0OjcwMDtjb2xvcjojYTBjNGZmO3doaXRlLXNwYWNlOm5vd3JhcH0KLnN0YXR7Zm9udC1zaXplOi43MmVtO2NvbG9yOiM1MDYwYTA7d2hpdGUtc3BhY2U6bm93cmFwfQouc3RhdCBie2NvbG9yOiNhMGM0ZmZ9CiNzZWFyY2h7YmFja2dyb3VuZDojMGUwZTIyO2JvcmRlcjoxcHggc29saWQgIzI1MjU1MDtjb2xvcjojYzBjOGZmO2JvcmRlci1yYWRpdXM6NHB4O3BhZGRpbmc6NHB4IDhweDtmb250LXNpemU6Ljc4ZW07d2lkdGg6MTYwcHg7b3V0bGluZTpub25lfQojc2VhcmNoOmZvY3Vze2JvcmRlci1jb2xvcjojNDRhYWZmfQojZmlsdGVyLWJ0bnN7ZGlzcGxheTpmbGV4O2dhcDo2cHh9Ci5mYnRue2JvcmRlcjpub25lO2JvcmRlci1yYWRpdXM6NHB4O3BhZGRpbmc6M3B4IDEwcHg7Zm9udC1zaXplOi43MmVtO2N1cnNvcjpwb2ludGVyO2ZvbnQtd2VpZ2h0OjYwMDt0cmFuc2l0aW9uOm9wYWNpdHkgLjFzfQouZmJ0bi5vbntvcGFjaXR5OjF9LmZidG4ub2Zme29wYWNpdHk6MC4zNX0KI2ZiLWFsbHtiYWNrZ3JvdW5kOiMzMzQ7Y29sb3I6I2FhY30KI2ZiLXN0cnVjdHtiYWNrZ3JvdW5kOiMwZTIwNDA7Y29sb3I6IzQ0YWFmZn0KI2ZiLXJhbmR7YmFja2dyb3VuZDojMjgxMjAwO2NvbG9yOiNmZjg4NDR9Ci5zZXB7d2lkdGg6MXB4O2JhY2tncm91bmQ6IzFlMWU0MDtoZWlnaHQ6MjBweDthbGlnbi1zZWxmOmNlbnRlcn0KI21haW57ZGlzcGxheTpmbGV4O2ZsZXg6MTttaW4taGVpZ2h0OjB9CiN2aWV3cG9ydHtmbGV4OjE7cG9zaXRpb246cmVsYXRpdmU7bWluLXdpZHRoOjB9CmNhbnZhc3tkaXNwbGF5OmJsb2NrO3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCV9CiNvdmVybGF5LWhpbnR7cG9zaXRpb246YWJzb2x1dGU7Ym90dG9tOjEwcHg7bGVmdDo1MCU7dHJhbnNmb3JtOnRyYW5zbGF0ZVgoLTUwJSk7Zm9udC1zaXplOi42OGVtO2NvbG9yOiM0MDQwNjA7cG9pbnRlci1ldmVudHM6bm9uZTt3aGl0ZS1zcGFjZTpub3dyYXB9CiNzaWRlYmFye3dpZHRoOjM0MHB4O2ZsZXgtc2hyaW5rOjA7YmFja2dyb3VuZDojMGQwZDFmO2JvcmRlci1sZWZ0OjFweCBzb2xpZCAjMWExYTM4O2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47b3ZlcmZsb3c6aGlkZGVufQojY2FtLWxpc3QtaGRye3BhZGRpbmc6OHB4IDEwcHggNnB4O2JvcmRlci1ib3R0b206MXB4IHNvbGlkICMxYTFhMzg7Zm9udC1zaXplOi43MmVtO2NvbG9yOiM2MDcwYTA7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpzcGFjZS1iZXR3ZWVufQojY2FtLWxpc3R7ZmxleDowIDAgMTcwcHg7b3ZlcmZsb3cteTphdXRvO2Rpc3BsYXk6ZmxleDtmbGV4LXdyYXA6d3JhcDtnYXA6M3B4O3BhZGRpbmc6NXB4O2FsaWduLWNvbnRlbnQ6ZmxleC1zdGFydDtib3JkZXItYm90dG9tOjFweCBzb2xpZCAjMWExYTM4fQojY2FtLWxpc3Q6Oi13ZWJraXQtc2Nyb2xsYmFye3dpZHRoOjVweH0KI2NhbS1saXN0Ojotd2Via2l0LXNjcm9sbGJhci10cmFja3tiYWNrZ3JvdW5kOiMwYTBhMTR9CiNjYW0tbGlzdDo6LXdlYmtpdC1zY3JvbGxiYXItdGh1bWJ7YmFja2dyb3VuZDojMjUyNTQwO2JvcmRlci1yYWRpdXM6M3B4fQoudGh1bWJ7d2lkdGg6NTBweDtoZWlnaHQ6NTBweDtib3JkZXItcmFkaXVzOjNweDtjdXJzb3I6cG9pbnRlcjtwb3NpdGlvbjpyZWxhdGl2ZTtib3JkZXI6MnB4IHNvbGlkIHRyYW5zcGFyZW50O292ZXJmbG93OmhpZGRlbjtmbGV4LXNocmluazowfQoudGh1bWIgaW1ne3dpZHRoOjEwMCU7aGVpZ2h0OjEwMCU7ZGlzcGxheTpibG9jaztvYmplY3QtZml0OmNvdmVyfQoudGh1bWIuc3RydWN0e2JvcmRlci1jb2xvcjojMWEzYTZhfQoudGh1bWIucmFuZHtib3JkZXItY29sb3I6IzFlMTIwOH0KLnRodW1iLnNlbHtib3JkZXItY29sb3I6IzQ0YWFmZiFpbXBvcnRhbnQ7Ym94LXNoYWRvdzowIDAgNnB4ICM0NGFhZmY4OH0KLnRodW1iLWlkeHtwb3NpdGlvbjphYnNvbHV0ZTtib3R0b206MXB4O3JpZ2h0OjJweDtmb250LXNpemU6LjVlbTtjb2xvcjojZmZmODt0ZXh0LXNoYWRvdzowIDAgM3B4ICMwMDB9Ci50aHVtYi5oaWRkZW57ZGlzcGxheTpub25lfQojZGV0YWlse2ZsZXg6MTttaW4taGVpZ2h0OjA7b3ZlcmZsb3cteTphdXRvO3BhZGRpbmc6MTBweH0KI2RldGFpbDo6LXdlYmtpdC1zY3JvbGxiYXJ7d2lkdGg6NXB4fQojZGV0YWlsOjotd2Via2l0LXNjcm9sbGJhci10cmFja3tiYWNrZ3JvdW5kOiMwYTBhMTR9CiNkZXRhaWw6Oi13ZWJraXQtc2Nyb2xsYmFyLXRodW1ie2JhY2tncm91bmQ6IzI1MjU0MDtib3JkZXItcmFkaXVzOjNweH0KI2RldGFpbC1ub25le2NvbG9yOiM0MDQwNjA7Zm9udC1zaXplOi44ZW07dGV4dC1hbGlnbjpjZW50ZXI7cGFkZGluZy10b3A6MzBweDtsaW5lLWhlaWdodDoxLjh9CiNkZXRhaWwtY29udGVudHtkaXNwbGF5Om5vbmV9CiNkZXQtdGl0bGV7Zm9udC1zaXplOi45NWVtO2ZvbnQtd2VpZ2h0OjcwMDtjb2xvcjojYTBjNGZmO21hcmdpbi1ib3R0b206NnB4fQouZGV0LWJhZGdle2Rpc3BsYXk6aW5saW5lLWJsb2NrO2JvcmRlci1yYWRpdXM6M3B4O3BhZGRpbmc6MXB4IDZweDtmb250LXNpemU6LjY4ZW07Zm9udC13ZWlnaHQ6NzAwO21hcmdpbi1ib3R0b206OHB4fQouYmFkZ2Utc3tiYWNrZ3JvdW5kOiMwZTIwNDA7Y29sb3I6IzQ0YWFmZn0KLmJhZGdlLXJ7YmFja2dyb3VuZDojMjgxMjAwO2NvbG9yOiNmZjg4NDR9CiNkZXQtaW1nLXdyYXB7dGV4dC1hbGlnbjpjZW50ZXI7bWFyZ2luLWJvdHRvbTo4cHh9CiNkZXQtaW1ne3dpZHRoOjI1NnB4O2hlaWdodDoyNTZweDtib3JkZXItcmFkaXVzOjVweDtpbWFnZS1yZW5kZXJpbmc6cGl4ZWxhdGVkO2JvcmRlcjoxcHggc29saWQgIzI1MjU0MDtkaXNwbGF5OmJsb2NrO21hcmdpbjowIGF1dG87YmFja2dyb3VuZDojMDUwNTEwfQojZGV0LWltZy1uYXZ7ZGlzcGxheTpmbGV4O2p1c3RpZnktY29udGVudDpjZW50ZXI7Z2FwOjZweDttYXJnaW4tdG9wOjVweDthbGlnbi1pdGVtczpjZW50ZXJ9Ci5uYXYtYnRue2JhY2tncm91bmQ6IzE1MTUzMDtjb2xvcjojYTBjNGZmO2JvcmRlcjoxcHggc29saWQgIzI1MjU1MDtib3JkZXItcmFkaXVzOjRweDtwYWRkaW5nOjNweCAxMHB4O2ZvbnQtc2l6ZTouNzJlbTtjdXJzb3I6cG9pbnRlcn0KLm5hdi1idG46aG92ZXJ7YmFja2dyb3VuZDojMWUxZTQ0fQojZGV0LW5hdi1pZHh7Zm9udC1zaXplOi43MmVtO2NvbG9yOiM1MDU4ODB9CiNkZXQtaW5mb3tmb250LXNpemU6LjdlbTtmb250LWZhbWlseTptb25vc3BhY2U7Y29sb3I6IzcwODBhMDtsaW5lLWhlaWdodDoxLjc7YmFja2dyb3VuZDojMDgwODE0O2JvcmRlci1yYWRpdXM6NHB4O3BhZGRpbmc6OHB4O2JvcmRlcjoxcHggc29saWQgIzE1MTUyODttYXJnaW4tdG9wOjhweH0KI2RldC1pbmZvIGJ7Y29sb3I6I2EwYzRmZn0KI2RldC1pbmZvIC5sYmx7Y29sb3I6IzUwNTg4MH0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KCjxkaXYgaWQ9InRvcGJhciI+CiAgPGgxPiYjMTI3NzYwOyAzRCBDYW1lcmEgRXhwbG9yZXI8L2gxPgogIDxkaXYgY2xhc3M9InNlcCI+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdCI+WmFycjogPGIgaWQ9InphcnItbGJsIj4tPC9iPjwvZGl2PgogIDxkaXYgY2xhc3M9InN0YXQiPkNhbWVyYXM6IDxiIGlkPSJjYW0tY291bnQiPi08L2I+PC9kaXY+CiAgPGRpdiBjbGFzcz0ic3RhdCI+UG9pbnQgY2xvdWQ6IDxiIGlkPSJwdC1jb3VudCI+LTwvYj4gcHRzPC9kaXY+CiAgPGRpdiBjbGFzcz0ic2VwIj48L2Rpdj4KICA8aW5wdXQgaWQ9InNlYXJjaCIgdHlwZT0idGV4dCIgcGxhY2Vob2xkZXI9IlNlYXJjaCBuYW1lIC8gI2lkeC4uLiI+CiAgPGRpdiBpZD0iZmlsdGVyLWJ0bnMiPgogICAgPGJ1dHRvbiBjbGFzcz0iZmJ0biBvbiIgaWQ9ImZiLWFsbCIgICAgb25jbGljaz0ic2V0RmlsdGVyKCdhbGwnKSI+QWxsPC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJmYnRuIG9uIiBpZD0iZmItc3RydWN0IiBvbmNsaWNrPSJzZXRGaWx0ZXIoJ3N0cnVjdCcpIj5TdHJ1Y3R1cmVkPC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJmYnRuIG9uIiBpZD0iZmItcmFuZCIgICBvbmNsaWNrPSJzZXRGaWx0ZXIoJ3JhbmQnKSI+UmFuZG9tPC9idXR0b24+CiAgPC9kaXY+CjwvZGl2PgoKPGRpdiBpZD0ibWFpbiI+CiAgPGRpdiBpZD0idmlld3BvcnQiPgogICAgPGNhbnZhcyBpZD0iY2FudmFzIj48L2NhbnZhcz4KICAgIDxkaXYgaWQ9Im92ZXJsYXktaGludCI+RHJhZyB0byBvcmJpdCAmbmJzcDsmbWlkZG90OyZuYnNwOyBTY3JvbGwgdG8gem9vbSAmbmJzcDsmbWlkZG90OyZuYnNwOyBDbGljayBhIGNhbWVyYSB0byBpbnNwZWN0PC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBpZD0ic2lkZWJhciI+CiAgICA8ZGl2IGlkPSJjYW0tbGlzdC1oZHIiPgogICAgICA8c3Bhbj5DYW1lcmEgdGh1bWJuYWlsczwvc3Bhbj4KICAgICAgPHNwYW4gaWQ9InZpcy1jb3VudCI+PC9zcGFuPgogICAgPC9kaXY+CiAgICA8ZGl2IGlkPSJjYW0tbGlzdCI+PC9kaXY+CiAgICA8ZGl2IGlkPSJkZXRhaWwiPgogICAgICA8ZGl2IGlkPSJkZXRhaWwtbm9uZSI+Q2xpY2sgYSBjYW1lcmEgaW4gdGhlIDNEIHZpZXc8YnI+b3IgYSB0aHVtYm5haWwgYWJvdmUgdG8gaW5zcGVjdCBpdC48L2Rpdj4KICAgICAgPGRpdiBpZD0iZGV0YWlsLWNvbnRlbnQiPgogICAgICAgIDxkaXYgaWQ9ImRldC10aXRsZSI+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZGV0LWJhZGdlIiBpZD0iZGV0LWJhZGdlIj48L2Rpdj4KICAgICAgICA8ZGl2IGlkPSJkZXQtaW1nLXdyYXAiPgogICAgICAgICAgPGltZyBpZD0iZGV0LWltZyIgc3JjPSIiIGFsdD0icmVuZGVyZWQgdmlldyI+CiAgICAgICAgICA8ZGl2IGlkPSJkZXQtaW1nLW5hdiI+CiAgICAgICAgICAgIDxidXR0b24gY2xhc3M9Im5hdi1idG4iIG9uY2xpY2s9Im5hdmlnYXRlQ2FtZXJhKC0xKSI+JiM4NTkyOyBQcmV2PC9idXR0b24+CiAgICAgICAgICAgIDxzcGFuIGlkPSJkZXQtbmF2LWlkeCI+PC9zcGFuPgogICAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJuYXYtYnRuIiBvbmNsaWNrPSJuYXZpZ2F0ZUNhbWVyYSgrMSkiPk5leHQgJiM4NTk0OzwvYnV0dG9uPgogICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8ZGl2IHN0eWxlPSJ0ZXh0LWFsaWduOmNlbnRlcjttYXJnaW4tdG9wOjRweCI+CiAgICAgICAgICAgIDxidXR0b24gY2xhc3M9Im5hdi1idG4iIGlkPSJkZXQtcmVzZXQtYnRuIiBvbmNsaWNrPSJjeWNsZVJlc2V0KCkiIHN0eWxlPSJkaXNwbGF5Om5vbmU7YmFja2dyb3VuZDojMWExYTJlO2ZvbnQtc2l6ZTouNjhlbSI+JiM5NjU0OyBSZXNldDwvYnV0dG9uPgogICAgICAgICAgICA8c3BhbiBpZD0iZGV0LXJlc2V0LWxibCIgc3R5bGU9ImZvbnQtc2l6ZTouNjhlbTtjb2xvcjojNTA1ODgwO21hcmdpbi1sZWZ0OjZweCI+PC9zcGFuPgogICAgICAgICAgPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBpZD0iZGV0LWluZm8iPjwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L2Rpdj4KCjxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL3RocmVlQDAuMTI4LjAvYnVpbGQvdGhyZWUubWluLmpzIj48L3NjcmlwdD4KPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vdGhyZWVAMC4xMjguMC9leGFtcGxlcy9qcy9jb250cm9scy9PcmJpdENvbnRyb2xzLmpzIj48L3NjcmlwdD4KPHNjcmlwdD4KY29uc3QgQ0FNRVJBUyAgPSBfX0NBTUVSQVNfSlNPTl9fOwpjb25zdCBQVFNfWFlaICA9IF9fUFRTX0pTT05fXzsKY29uc3QgUFRTX1JHQiAgPSBfX1JHQl9KU09OX187CmNvbnN0IENFTlRFUiAgID0gX19DRU5URVJfSlNPTl9fOwpjb25zdCBaX1RBQkxFICA9IF9fWlRBQkxFX0pTT05fXzsKY29uc3QgWkFSUl9MQkwgPSBfX1pBUlJfSlNPTl9fOwoKLy8gQ29vcmQ6IFJMQmVuY2ggKFggZndkLCBZIGxlZnQsIFogdXApIC0+IFRocmVlLmpzIChZIHVwKQovLyBUaHJlZVg9UkwuWCwgVGhyZWVZPVJMLlosIFRocmVlWj0tUkwuWQpmdW5jdGlvbiBybDJ0KHgseSx6KXtyZXR1cm4gbmV3IFRIUkVFLlZlY3RvcjMoeCx6LC15KTt9CmZ1bmN0aW9uIHJsMnRBcnIoYSl7cmV0dXJuIHJsMnQoYVswXSxhWzFdLGFbMl0pO30KCmxldCBzZWxlY3RlZElkeD0tMSwgYWN0aXZlRmlsdGVyPSdhbGwnLCB2aXNpYmxlSW5kaWNlcz1bXTsKY29uc3QgJD1pZD0+ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpOwokKCd6YXJyLWxibCcpLnRleHRDb250ZW50PVpBUlJfTEJMOwokKCdjYW0tY291bnQnKS50ZXh0Q29udGVudD1DQU1FUkFTLmxlbmd0aDsKCi8vIFRodW1ibmFpbCBzdHJpcApjb25zdCBjYW1MaXN0PSQoJ2NhbS1saXN0Jyk7CkNBTUVSQVMuZm9yRWFjaCgoY2FtLGkpPT57CiAgY29uc3QgZWw9ZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7CiAgZWwuY2xhc3NOYW1lPSd0aHVtYiAnKyhjYW0uc3RydWN0dXJlZD8nc3RydWN0JzoncmFuZCcpOwogIGVsLmRhdGFzZXQuaWR4PWk7CiAgZWwudGl0bGU9JyMnK2krJyAnK2NhbS5uYW1lOwogIGNvbnN0IGltZz1kb2N1bWVudC5jcmVhdGVFbGVtZW50KCdpbWcnKTsKICBpbWcuc3JjPWNhbS5pbWdzWzBdOyBpbWcubG9hZGluZz0nbGF6eSc7CiAgY29uc3QgbGJsPWRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ3NwYW4nKTsKICBsYmwuY2xhc3NOYW1lPSd0aHVtYi1pZHgnOyBsYmwudGV4dENvbnRlbnQ9JyMnK2k7CiAgZWwuYXBwZW5kQ2hpbGQoaW1nKTsgZWwuYXBwZW5kQ2hpbGQobGJsKTsKICBlbC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsKCk9PnNlbGVjdENhbWVyYShpKSk7CiAgY2FtTGlzdC5hcHBlbmRDaGlsZChlbCk7Cn0pOwoKZnVuY3Rpb24gdXBkYXRlVGh1bWJWaXNpYmlsaXR5KCl7CiAgY29uc3QgcT0kKCdzZWFyY2gnKS52YWx1ZS50cmltKCkudG9Mb3dlckNhc2UoKTsKICBsZXQgY291bnQ9MDsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcudGh1bWInKS5mb3JFYWNoKGVsPT57CiAgICBjb25zdCBpZHg9cGFyc2VJbnQoZWwuZGF0YXNldC5pZHgpOwogICAgY29uc3QgY2FtPUNBTUVSQVNbaWR4XTsKICAgIGNvbnN0IG1mPWFjdGl2ZUZpbHRlcj09PSdhbGwnfHwoYWN0aXZlRmlsdGVyPT09J3N0cnVjdCcmJmNhbS5zdHJ1Y3R1cmVkKXx8KGFjdGl2ZUZpbHRlcj09PSdyYW5kJyYmIWNhbS5zdHJ1Y3R1cmVkKTsKICAgIGNvbnN0IG1zPSFxfHxjYW0ubmFtZS50b0xvd2VyQ2FzZSgpLmluY2x1ZGVzKHEpfHxTdHJpbmcoaWR4KS5pbmNsdWRlcyhxKTsKICAgIGNvbnN0IHZpcz1tZiYmbXM7CiAgICBlbC5jbGFzc0xpc3QudG9nZ2xlKCdoaWRkZW4nLCF2aXMpOwogICAgaWYodmlzKWNvdW50Kys7CiAgfSk7CiAgJCgndmlzLWNvdW50JykudGV4dENvbnRlbnQ9Y291bnQrJyBzaG93bic7CiAgdmlzaWJsZUluZGljZXM9W107CiAgQ0FNRVJBUy5mb3JFYWNoKChfLGkpPT57aWYoIWRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy50aHVtYicpW2ldLmNsYXNzTGlzdC5jb250YWlucygnaGlkZGVuJykpdmlzaWJsZUluZGljZXMucHVzaChpKTt9KTsKfQpmdW5jdGlvbiBzZXRGaWx0ZXIoZil7CiAgYWN0aXZlRmlsdGVyPWY7CiAgWydhbGwnLCdzdHJ1Y3QnLCdyYW5kJ10uZm9yRWFjaChrPT57CiAgICAkKCdmYi0nK2spLmNsYXNzTGlzdC50b2dnbGUoJ29uJyxrPT09Zik7CiAgICAkKCdmYi0nK2spLmNsYXNzTGlzdC50b2dnbGUoJ29mZicsayE9PWYpOwogIH0pOwogIHVwZGF0ZVRodW1iVmlzaWJpbGl0eSgpOyB1cGRhdGVNZXNoVmlzaWJpbGl0eSgpOwp9CiQoJ3NlYXJjaCcpLmFkZEV2ZW50TGlzdGVuZXIoJ2lucHV0JywoKT0+dXBkYXRlVGh1bWJWaXNpYmlsaXR5KCkpOwp1cGRhdGVUaHVtYlZpc2liaWxpdHkoKTsKCi8vIFRocmVlLmpzIHNjZW5lCmNvbnN0IGNhbnZhcz0kKCdjYW52YXMnKTsKY29uc3QgcmVuZGVyZXI9bmV3IFRIUkVFLldlYkdMUmVuZGVyZXIoe2NhbnZhcyxhbnRpYWxpYXM6dHJ1ZX0pOwpyZW5kZXJlci5zZXRQaXhlbFJhdGlvKHdpbmRvdy5kZXZpY2VQaXhlbFJhdGlvKTsKY29uc3Qgc2NlbmU9bmV3IFRIUkVFLlNjZW5lKCk7CnNjZW5lLmJhY2tncm91bmQ9bmV3IFRIUkVFLkNvbG9yKDB4MGEwYTE0KTsKc2NlbmUuZm9nPW5ldyBUSFJFRS5Gb2dFeHAyKDB4MGEwYTE0LDAuMDUpOwpjb25zdCBjYW0zPW5ldyBUSFJFRS5QZXJzcGVjdGl2ZUNhbWVyYSg1NSwxLDAuMDEsNTApOwpjYW0zLnBvc2l0aW9uLmNvcHkocmwydChDRU5URVJbMF0tMy44LENFTlRFUlsxXSsyLjIsQ0VOVEVSWzJdKzEuOCkpOwpjb25zdCBjb250cm9scz1uZXcgVEhSRUUuT3JiaXRDb250cm9scyhjYW0zLHJlbmRlcmVyLmRvbUVsZW1lbnQpOwpjb250cm9scy50YXJnZXQuY29weShybDJ0KENFTlRFUlswXSxDRU5URVJbMV0sQ0VOVEVSWzJdKSk7CmNvbnRyb2xzLmVuYWJsZURhbXBpbmc9dHJ1ZTsgY29udHJvbHMuZGFtcGluZ0ZhY3Rvcj0wLjA4Owpjb250cm9scy5taW5EaXN0YW5jZT0wLjM7IGNvbnRyb2xzLm1heERpc3RhbmNlPTE1Owpjb250cm9scy51cGRhdGUoKTsKc2NlbmUuYWRkKG5ldyBUSFJFRS5BbWJpZW50TGlnaHQoMHhmZmZmZmYsMC43KSk7CmNvbnN0IGRMPW5ldyBUSFJFRS5EaXJlY3Rpb25hbExpZ2h0KDB4ZmZmZmZmLDAuNTUpOwpkTC5wb3NpdGlvbi5zZXQoMiw1LDMpOyBzY2VuZS5hZGQoZEwpOwoKLy8gVGFibGUgcGxhbmUKY29uc3QgdGFibGVZPVpfVEFCTEU7CmNvbnN0IHRNZXNoPW5ldyBUSFJFRS5NZXNoKAogIG5ldyBUSFJFRS5QbGFuZUdlb21ldHJ5KDgsOCksCiAgbmV3IFRIUkVFLk1lc2hTdGFuZGFyZE1hdGVyaWFsKHtjb2xvcjoweDE0MTQyMixzaWRlOlRIUkVFLkRvdWJsZVNpZGUsdHJhbnNwYXJlbnQ6dHJ1ZSxvcGFjaXR5OjAuNjV9KQopOwp0TWVzaC5yb3RhdGlvbi54PS1NYXRoLlBJLzI7IHRNZXNoLnBvc2l0aW9uLnk9dGFibGVZOyBzY2VuZS5hZGQodE1lc2gpOwpjb25zdCBncmlkPW5ldyBUSFJFRS5HcmlkSGVscGVyKDgsNDAsMHgxZTFlMzgsMHgxNDE0MjgpOwpncmlkLnBvc2l0aW9uLnk9dGFibGVZLTAuMDAxOyBzY2VuZS5hZGQoZ3JpZCk7CgovLyBXb3Jrc3BhY2UgY2VudHJlIG1hcmtlcgpjb25zdCBjTWVzaD1uZXcgVEhSRUUuTWVzaCgKICBuZXcgVEhSRUUuU3BoZXJlR2VvbWV0cnkoMC4wNDUsMTYsMTIpLAogIG5ldyBUSFJFRS5NZXNoU3RhbmRhcmRNYXRlcmlhbCh7Y29sb3I6MHhmZmRkNDQsZW1pc3NpdmU6MHg5OTc3MDAsZW1pc3NpdmVJbnRlbnNpdHk6MC41fSkKKTsKY01lc2gucG9zaXRpb24uY29weShybDJ0KENFTlRFUlswXSxDRU5URVJbMV0sQ0VOVEVSWzJdKSk7CnNjZW5lLmFkZChjTWVzaCk7CmNvbnN0IGF4SD1uZXcgVEhSRUUuQXhlc0hlbHBlcigwLjI1KTsKYXhILnBvc2l0aW9uLmNvcHkoY01lc2gucG9zaXRpb24pOyBzY2VuZS5hZGQoYXhIKTsKCi8vIFBvaW50IGNsb3VkCihmdW5jdGlvbigpewogIGlmKCFQVFNfWFlafHxQVFNfWFlaLmxlbmd0aD09PTApcmV0dXJuOwogIGNvbnN0IE49UFRTX1hZWi5sZW5ndGgvMzsKICAkKCdwdC1jb3VudCcpLnRleHRDb250ZW50PU4udG9Mb2NhbGVTdHJpbmcoKTsKICBjb25zdCBwb3M9bmV3IEZsb2F0MzJBcnJheShOKjMpLCBjb2w9bmV3IEZsb2F0MzJBcnJheShOKjMpOwogIGZvcihsZXQgaT0wO2k8TjtpKyspewogICAgcG9zW2kqM109UFRTX1hZWltpKjNdOyBwb3NbaSozKzFdPVBUU19YWVpbaSozKzJdOyBwb3NbaSozKzJdPS1QVFNfWFlaW2kqMysxXTsKICAgIGNvbFtpKjNdPVBUU19SR0JbaSozXS8yNTU7IGNvbFtpKjMrMV09UFRTX1JHQltpKjMrMV0vMjU1OyBjb2xbaSozKzJdPVBUU19SR0JbaSozKzJdLzI1NTsKICB9CiAgY29uc3QgZ2VvPW5ldyBUSFJFRS5CdWZmZXJHZW9tZXRyeSgpOwogIGdlby5zZXRBdHRyaWJ1dGUoJ3Bvc2l0aW9uJyxuZXcgVEhSRUUuQnVmZmVyQXR0cmlidXRlKHBvcywzKSk7CiAgZ2VvLnNldEF0dHJpYnV0ZSgnY29sb3InLG5ldyBUSFJFRS5CdWZmZXJBdHRyaWJ1dGUoY29sLDMpKTsKICBzY2VuZS5hZGQobmV3IFRIUkVFLlBvaW50cyhnZW8sbmV3IFRIUkVFLlBvaW50c01hdGVyaWFsKHtzaXplOjAuMDE0LHZlcnRleENvbG9yczp0cnVlLHNpemVBdHRlbnVhdGlvbjp0cnVlLHRyYW5zcGFyZW50OnRydWUsb3BhY2l0eTowLjl9KSkpOwp9KSgpOwoKLy8gQ2FtZXJhIG1hcmtlcnMKY29uc3QgQ0FNX1NSPTAuMDMwLCBDQU1fUlI9MC4wMTc7CmNvbnN0IENPTF9TPW5ldyBUSFJFRS5Db2xvcigweDQ0YWFmZiksIENPTF9SPW5ldyBUSFJFRS5Db2xvcigweGZmODg0NCksIENPTF9TRUw9bmV3IFRIUkVFLkNvbG9yKDB4ZmZmZmZmKTsKY29uc3QgY2FtTWVzaGVzPVtdLCBjYW1BcnJvd3M9W107CmNvbnN0IHNHZW89bmV3IFRIUkVFLlNwaGVyZUdlb21ldHJ5KDEsMTIsOCk7CkNBTUVSQVMuZm9yRWFjaCgoY2FtLGkpPT57CiAgY29uc3Qgcj1jYW0uc3RydWN0dXJlZD9DQU1fU1I6Q0FNX1JSOwogIGNvbnN0IGNvbD1jYW0uc3RydWN0dXJlZD9DT0xfUzpDT0xfUjsKICBjb25zdCBtYXQ9bmV3IFRIUkVFLk1lc2hTdGFuZGFyZE1hdGVyaWFsKHtjb2xvcjpjb2wuY2xvbmUoKSxlbWlzc2l2ZTpjb2wuY2xvbmUoKSxlbWlzc2l2ZUludGVuc2l0eTowLjQscm91Z2huZXNzOjAuNSxtZXRhbG5lc3M6MC4yfSk7CiAgY29uc3QgbWVzaD1uZXcgVEhSRUUuTWVzaChzR2VvLG1hdCk7CiAgbWVzaC5zY2FsZS5zZXRTY2FsYXIocik7CiAgY29uc3QgcD1ybDJ0QXJyKGNhbS5wb3MpOwogIG1lc2gucG9zaXRpb24uY29weShwKTsKICBtZXNoLnVzZXJEYXRhLmNhbUlkeD1pOwogIHNjZW5lLmFkZChtZXNoKTsgY2FtTWVzaGVzLnB1c2gobWVzaCk7CiAgY29uc3QgZlQ9cmwydChjYW0uZndkWzBdLGNhbS5md2RbMV0sY2FtLmZ3ZFsyXSkubm9ybWFsaXplKCk7CiAgY29uc3QgYUxlbj1jYW0uc3RydWN0dXJlZD8wLjIwOjAuMTM7CiAgY29uc3QgYXJyPW5ldyBUSFJFRS5BcnJvd0hlbHBlcihmVCxwLGFMZW4sY2FtLnN0cnVjdHVyZWQ/MHg0NGFhZmY6MHhmZjg4NDQsYUxlbiowLjM1LGFMZW4qMC4xOCk7CiAgYXJyLnVzZXJEYXRhLmNhbUlkeD1pOyBzY2VuZS5hZGQoYXJyKTsgY2FtQXJyb3dzLnB1c2goYXJyKTsKfSk7CgovLyBGcnVzdHVtIGZvciBzZWxlY3RlZCBjYW1lcmEKbGV0IGZydXN0dW1HcnA9bnVsbDsKZnVuY3Rpb24gc2hvd0ZydXN0dW0oaWR4KXsKICBpZihmcnVzdHVtR3JwKXtzY2VuZS5yZW1vdmUoZnJ1c3R1bUdycCk7ZnJ1c3R1bUdycD1udWxsO30KICBpZihpZHg8MClyZXR1cm47CiAgY29uc3QgY2FtPUNBTUVSQVNbaWR4XTsKICBjb25zdCBwPXJsMnRBcnIoY2FtLnBvcyk7CiAgY29uc3QgZm92PU1hdGguUEkvMywgZmFyPTAuNjA7CiAgY29uc3QgaEY9TWF0aC50YW4oZm92LzIpKmZhcjsKICBjb25zdCBSPWNhbS5SOwogIGZ1bmN0aW9uIGN2KGN4LGN5LGN6KXsKICAgIHJldHVybiBybDJ0KFJbMF1bMF0qY3grUlswXVsxXSpjeStSWzBdWzJdKmN6LAogICAgICAgICAgICAgICAgUlsxXVswXSpjeCtSWzFdWzFdKmN5K1JbMV1bMl0qY3osCiAgICAgICAgICAgICAgICBSWzJdWzBdKmN4K1JbMl1bMV0qY3krUlsyXVsyXSpjeik7CiAgfQogIGZ1bmN0aW9uIGNvcm5lcihzeCxzeSl7cmV0dXJuIG5ldyBUSFJFRS5WZWN0b3IzKCkuY29weShwKS5hZGQoY3Yoc3gqaEYsc3kqaEYsZmFyKSk7fQogIGNvbnN0IHRsPWNvcm5lcigtMSwtMSksdHI9Y29ybmVyKDEsLTEpLGJsPWNvcm5lcigtMSwxKSxicj1jb3JuZXIoMSwxKTsKICBjb25zdCBsbT1uZXcgVEhSRUUuTGluZUJhc2ljTWF0ZXJpYWwoe2NvbG9yOjB4NDRmZmNjLHRyYW5zcGFyZW50OnRydWUsb3BhY2l0eTowLjc1fSk7CiAgZnJ1c3R1bUdycD1uZXcgVEhSRUUuR3JvdXAoKTsKICBbW3AsdGxdLFtwLHRyXSxbcCxibF0sW3AsYnJdLFt0bCx0cl0sW3RyLGJyXSxbYnIsYmxdLFtibCx0bF1dLmZvckVhY2goKFthLGJdKT0+ewogICAgZnJ1c3R1bUdycC5hZGQobmV3IFRIUkVFLkxpbmUobmV3IFRIUkVFLkJ1ZmZlckdlb21ldHJ5KCkuc2V0RnJvbVBvaW50cyhbYSxiXSksbG0pKTsKICB9KTsKICBjb25zdCBmZz1uZXcgVEhSRUUuQnVmZmVyR2VvbWV0cnkoKTsKICBmZy5zZXRBdHRyaWJ1dGUoJ3Bvc2l0aW9uJyxuZXcgVEhSRUUuRmxvYXQzMkJ1ZmZlckF0dHJpYnV0ZShbCiAgICB0bC54LHRsLnksdGwueix0ci54LHRyLnksdHIueixici54LGJyLnksYnIueiwKICAgIHRsLngsdGwueSx0bC56LGJyLngsYnIueSxici56LGJsLngsYmwueSxibC56CiAgXSwzKSk7CiAgZnJ1c3R1bUdycC5hZGQobmV3IFRIUkVFLk1lc2goZmcsbmV3IFRIUkVFLk1lc2hCYXNpY01hdGVyaWFsKHtjb2xvcjoweDQ0ZmZjYyx0cmFuc3BhcmVudDp0cnVlLG9wYWNpdHk6MC4wNyxzaWRlOlRIUkVFLkRvdWJsZVNpZGV9KSkpOwogIHNjZW5lLmFkZChmcnVzdHVtR3JwKTsKfQoKZnVuY3Rpb24gdXBkYXRlTWVzaFZpc2liaWxpdHkoKXsKICBjYW1NZXNoZXMuZm9yRWFjaChtPT57CiAgICBjb25zdCBjPUNBTUVSQVNbbS51c2VyRGF0YS5jYW1JZHhdOwogICAgbS52aXNpYmxlPWFjdGl2ZUZpbHRlcj09PSdhbGwnfHwoYWN0aXZlRmlsdGVyPT09J3N0cnVjdCcmJmMuc3RydWN0dXJlZCl8fChhY3RpdmVGaWx0ZXI9PT0ncmFuZCcmJiFjLnN0cnVjdHVyZWQpOwogIH0pOwogIGNhbUFycm93cy5mb3JFYWNoKGE9PnsKICAgIGNvbnN0IGM9Q0FNRVJBU1thLnVzZXJEYXRhLmNhbUlkeF07CiAgICBhLnZpc2libGU9YWN0aXZlRmlsdGVyPT09J2FsbCd8fChhY3RpdmVGaWx0ZXI9PT0nc3RydWN0JyYmYy5zdHJ1Y3R1cmVkKXx8KGFjdGl2ZUZpbHRlcj09PSdyYW5kJyYmIWMuc3RydWN0dXJlZCk7CiAgfSk7Cn0KCi8vIFJheWNhc3RpbmcKY29uc3QgcmM9bmV3IFRIUkVFLlJheWNhc3RlcigpOwpjb25zdCBtb3VzZT1uZXcgVEhSRUUuVmVjdG9yMigpOwpsZXQgaXNEcmFnPWZhbHNlLCBtZHA9e3g6MCx5OjB9OwpjYW52YXMuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vkb3duJyxlPT57bWRwPXt4OmUuY2xpZW50WCx5OmUuY2xpZW50WX07aXNEcmFnPWZhbHNlO30pOwpjYW52YXMuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJyxlPT57aWYoTWF0aC5hYnMoZS5jbGllbnRYLW1kcC54KT40fHxNYXRoLmFicyhlLmNsaWVudFktbWRwLnkpPjQpaXNEcmFnPXRydWU7fSk7CmNhbnZhcy5hZGRFdmVudExpc3RlbmVyKCdtb3VzZXVwJyxlPT57CiAgaWYoaXNEcmFnKXJldHVybjsKICBjb25zdCByPWNhbnZhcy5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTsKICBtb3VzZS54PSgoZS5jbGllbnRYLXIubGVmdCkvci53aWR0aCkqMi0xOwogIG1vdXNlLnk9KChlLmNsaWVudFktci50b3ApL3IuaGVpZ2h0KSotMisxOwogIHJjLnNldEZyb21DYW1lcmEobW91c2UsY2FtMyk7CiAgY29uc3QgaGl0cz1yYy5pbnRlcnNlY3RPYmplY3RzKGNhbU1lc2hlcyxmYWxzZSk7CiAgaWYoaGl0cy5sZW5ndGg+MClzZWxlY3RDYW1lcmEoaGl0c1swXS5vYmplY3QudXNlckRhdGEuY2FtSWR4KTsKfSk7CgovLyBTZWxlY3QgY2FtZXJhCmZ1bmN0aW9uIHNlbGVjdENhbWVyYShpZHgpewogIGlmKHNlbGVjdGVkSWR4Pj0wKXsKICAgIGNvbnN0IG9sZD1jYW1NZXNoZXMuZmluZChtPT5tLnVzZXJEYXRhLmNhbUlkeD09PXNlbGVjdGVkSWR4KTsKICAgIGlmKG9sZCl7Y29uc3QgYz1DQU1FUkFTW3NlbGVjdGVkSWR4XTtjb25zdCBjb2w9Yy5zdHJ1Y3R1cmVkP0NPTF9TOkNPTF9SOwogICAgICBvbGQubWF0ZXJpYWwuY29sb3IuY29weShjb2wpO29sZC5tYXRlcmlhbC5lbWlzc2l2ZS5jb3B5KGNvbCk7CiAgICAgIG9sZC5tYXRlcmlhbC5lbWlzc2l2ZUludGVuc2l0eT0wLjQ7CiAgICAgIG9sZC5zY2FsZS5zZXRTY2FsYXIoYy5zdHJ1Y3R1cmVkP0NBTV9TUjpDQU1fUlIpO30KICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy50aHVtYi5zZWwnKS5mb3JFYWNoKGVsPT5lbC5jbGFzc0xpc3QucmVtb3ZlKCdzZWwnKSk7CiAgfQogIHNlbGVjdGVkSWR4PWlkeDsKICBjb25zdCBtZXNoPWNhbU1lc2hlcy5maW5kKG09Pm0udXNlckRhdGEuY2FtSWR4PT09aWR4KTsKICBpZihtZXNoKXttZXNoLm1hdGVyaWFsLmNvbG9yLmNvcHkoQ09MX1NFTCk7bWVzaC5tYXRlcmlhbC5lbWlzc2l2ZS5jb3B5KENPTF9TRUwpOwogICAgbWVzaC5tYXRlcmlhbC5lbWlzc2l2ZUludGVuc2l0eT0wLjk7bWVzaC5zY2FsZS5zZXRTY2FsYXIoMC4wNDgpO30KICBjb25zdCB0aHVtYj1kb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcudGh1bWInKVtpZHhdOwogIGlmKHRodW1iKXt0aHVtYi5jbGFzc0xpc3QuYWRkKCdzZWwnKTt0aHVtYi5zY3JvbGxJbnRvVmlldyh7YmxvY2s6J25lYXJlc3QnLGJlaGF2aW9yOidzbW9vdGgnfSk7fQogIHNob3dGcnVzdHVtKGlkeCk7IHVwZGF0ZUluZm9QYW5lbChpZHgpOwp9CmZ1bmN0aW9uIG5hdmlnYXRlQ2FtZXJhKGQpewogIGlmKCF2aXNpYmxlSW5kaWNlcy5sZW5ndGgpcmV0dXJuOwogIGNvbnN0IGN1cj12aXNpYmxlSW5kaWNlcy5pbmRleE9mKHNlbGVjdGVkSWR4KTsKICBzZWxlY3RDYW1lcmEodmlzaWJsZUluZGljZXNbKGN1citkK3Zpc2libGVJbmRpY2VzLmxlbmd0aCkldmlzaWJsZUluZGljZXMubGVuZ3RoXSk7Cn0KCi8vIEluZm8gcGFuZWwKZnVuY3Rpb24gdXBkYXRlSW5mb1BhbmVsKGlkeCl7CiAgY29uc3QgY2FtPUNBTUVSQVNbaWR4XTsKICAkKCdkZXRhaWwtbm9uZScpLnN0eWxlLmRpc3BsYXk9J25vbmUnOwogICQoJ2RldGFpbC1jb250ZW50Jykuc3R5bGUuZGlzcGxheT0nYmxvY2snOwogICQoJ2RldC10aXRsZScpLnRleHRDb250ZW50PScjJytTdHJpbmcoaWR4KS5wYWRTdGFydCgzLCcwJykrJyAgJytjYW0ubmFtZTsKICBjb25zdCBiPSQoJ2RldC1iYWRnZScpOwogIGIudGV4dENvbnRlbnQ9Y2FtLnN0cnVjdHVyZWQ/J1tTXSBTdHJ1Y3R1cmVkJzonW1JdIFJhbmRvbSc7CiAgYi5jbGFzc05hbWU9J2RldC1iYWRnZSAnKyhjYW0uc3RydWN0dXJlZD8nYmFkZ2Utcyc6J2JhZGdlLXInKTsKICAvLyBTaG93IHJlc2V0IGltYWdlcyB3aXRoIGN5Y2xpbmcKICBjb25zdCByZXNldEltZ3M9Y2FtLmltZ3M7CiAgbGV0IHJlc2V0SWR4PTA7CiAgJCgnZGV0LWltZycpLnNyYz1yZXNldEltZ3NbMF07CiAgJCgnZGV0LXJlc2V0LWxibCcpLnRleHRDb250ZW50PXJlc2V0SW1ncy5sZW5ndGg+MT8nUmVzZXQgMCAvICcrKHJlc2V0SW1ncy5sZW5ndGgtMSk6Jyc7CiAgJCgnZGV0LXJlc2V0LWJ0bicpLnN0eWxlLmRpc3BsYXk9cmVzZXRJbWdzLmxlbmd0aD4xPydpbmxpbmUtYmxvY2snOidub25lJzsKICB3aW5kb3cuX2N1ckNhbVJlc2V0cz1yZXNldEltZ3M7IHdpbmRvdy5fY3VyUmVzZXRJZHg9MDsKICBjb25zdCBjdXI9dmlzaWJsZUluZGljZXMuaW5kZXhPZihpZHgpOwogICQoJ2RldC1uYXYtaWR4JykudGV4dENvbnRlbnQ9KGN1cisxKSsnIC8gJyt2aXNpYmxlSW5kaWNlcy5sZW5ndGg7CiAgY29uc3QgcD1jYW0ucG9zOwogIGNvbnN0IGU9Y2FtLmV1bGVyfHxbJy0nLCctJywnLSddOwogIGNvbnN0IHE9Y2FtLnF1YXR8fFsnLScsJy0nLCctJywnLSddOwogICQoJ2RldC1pbmZvJykuaW5uZXJIVE1MPQogICAgJzxiPlBvc2l0aW9uIChtKTwvYj48YnI+JysKICAgICc8c3BhbiBjbGFzcz0ibGJsIj54PTwvc3Bhbj4nK3BbMF0udG9GaXhlZCg0KSsnJm5ic3A7Jm5ic3A7JysKICAgICc8c3BhbiBjbGFzcz0ibGJsIj55PTwvc3Bhbj4nK3BbMV0udG9GaXhlZCg0KSsnJm5ic3A7Jm5ic3A7JysKICAgICc8c3BhbiBjbGFzcz0ibGJsIj56PTwvc3Bhbj4nK3BbMl0udG9GaXhlZCg0KSsnPGJyPjxicj4nKwogICAgJzxiPk9yaWVudGF0aW9uPC9iPjxicj4nKwogICAgJzxzcGFuIGNsYXNzPSJsYmwiPkV1bGVyIFhZWjo8L3NwYW4+IFsnK2UubWFwKHY9PnR5cGVvZiB2PT09J251bWJlcic/di50b0ZpeGVkKDIpOnYpLmpvaW4oJywgJykrJ10mZGVnOzxicj4nKwogICAgJzxzcGFuIGNsYXNzPSJsYmwiPlF1YXRlcm5pb24gKHh5encpOjwvc3Bhbj4gWycrcS5tYXAodj0+dHlwZW9mIHY9PT0nbnVtYmVyJz92LnRvRml4ZWQoNCk6dikuam9pbignLCAnKSsnXTxicj4nKwogICAgJzxzcGFuIGNsYXNzPSJsYmwiPlJvbGw6PC9zcGFuPiAnK2NhbS5yb2xsX2RlZy50b0ZpeGVkKDIpKycmZGVnOzxicj48YnI+JysKICAgICc8Yj5TY2VuZTwvYj48YnI+JysKICAgICc8c3BhbiBjbGFzcz0ibGJsIj5EaXN0IGZyb20gY2VudHJlOjwvc3Bhbj4gJytjYW0uZGlzdF9mcm9tX2NlbnRlci50b0ZpeGVkKDQpKycgbTxicj4nKwogICAgJzxzcGFuIGNsYXNzPSJsYmwiPlR5cGU6PC9zcGFuPiAnKyhjYW0uc3RydWN0dXJlZD8nU3RydWN0dXJlZCAobmFtZWQpJzonUmFuZG9tIHNhbXBsZScpOwp9CgovLyBSZXNpemUKCmZ1bmN0aW9uIGN5Y2xlUmVzZXQoKXsKICBpZighd2luZG93Ll9jdXJDYW1SZXNldHMpcmV0dXJuOwogIHdpbmRvdy5fY3VyUmVzZXRJZHg9KHdpbmRvdy5fY3VyUmVzZXRJZHgrMSkld2luZG93Ll9jdXJDYW1SZXNldHMubGVuZ3RoOwogICQoJ2RldC1pbWcnKS5zcmM9d2luZG93Ll9jdXJDYW1SZXNldHNbd2luZG93Ll9jdXJSZXNldElkeF07CiAgJCgnZGV0LXJlc2V0LWxibCcpLnRleHRDb250ZW50PSdSZXNldCAnK3dpbmRvdy5fY3VyUmVzZXRJZHgrJyAvICcrKHdpbmRvdy5fY3VyQ2FtUmVzZXRzLmxlbmd0aC0xKTsKfQoKZnVuY3Rpb24gb25SZXNpemUoKXsKICBjb25zdCB2cD0kKCd2aWV3cG9ydCcpOwogIHJlbmRlcmVyLnNldFNpemUodnAuY2xpZW50V2lkdGgsdnAuY2xpZW50SGVpZ2h0LGZhbHNlKTsKICBjYW0zLmFzcGVjdD12cC5jbGllbnRXaWR0aC92cC5jbGllbnRIZWlnaHQ7IGNhbTMudXBkYXRlUHJvamVjdGlvbk1hdHJpeCgpOwp9CndpbmRvdy5hZGRFdmVudExpc3RlbmVyKCdyZXNpemUnLG9uUmVzaXplKTsgb25SZXNpemUoKTsKCi8vIFJlbmRlciBsb29wCihmdW5jdGlvbiBhbmltYXRlKCl7cmVxdWVzdEFuaW1hdGlvbkZyYW1lKGFuaW1hdGUpO2NvbnRyb2xzLnVwZGF0ZSgpO3JlbmRlcmVyLnJlbmRlcihzY2VuZSxjYW0zKTt9KSgpOwo8L3NjcmlwdD4KPC9ib2R5Pgo8L2h0bWw+Cg=="""

def get_html_template():
    return _b64.b64decode(_HTML_B64).decode('utf-8')


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(candidates_data, pts, cols, center, z_table, zarr_label,
               max_cloud_pts=8000):
    rng = np.random.default_rng(42)
    n = len(pts)
    if n > max_cloud_pts:
        idx = rng.choice(n, max_cloud_pts, replace=False)
        pts_sub, cols_sub = pts[idx], cols[idx]
    else:
        pts_sub, cols_sub = pts, cols

    pts_flat  = pts_sub.astype(np.float32).flatten().tolist()
    cols_flat = cols_sub.astype(np.uint8).flatten().tolist()

    html = get_html_template()
    html = html.replace("__CAMERAS_JSON__", json.dumps(candidates_data))
    html = html.replace("__PTS_JSON__",     json.dumps(pts_flat))
    html = html.replace("__RGB_JSON__",     json.dumps(cols_flat))
    html = html.replace("__CENTER_JSON__",  json.dumps([round(float(v),4) for v in center]))
    html = html.replace("__ZTABLE_JSON__",  json.dumps(round(float(z_table),4)))
    html = html.replace("__ZARR_JSON__",    json.dumps(zarr_label))
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="3D camera-position visualisation.")
    p.add_argument("--zarr",      default="Peract_zarr/val.zarr")
    p.add_argument("--configs",   default="camera_render/configs.json")
    p.add_argument("--cam_dir",   default="camera_render/")
    p.add_argument("--out",       default="camera_render/viz.html")
    p.add_argument("--n_frames",  type=int,   default=20)
    p.add_argument("--img_size",  type=int,   default=256)
    p.add_argument("--fov_deg",   type=float, default=65.0)
    p.add_argument("--cloud_pts", type=int,   default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = args.zarr
    if not zarr_path.endswith(".zarr"): zarr_path += ".zarr"
    if not os.path.isdir(zarr_path):
        sys.exit("[ERROR] Zarr not found: " + zarr_path)
    if not os.path.isfile(args.configs):
        sys.exit("[ERROR] configs.json not found: " + args.configs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[1/4] Loading point cloud from {}...".format(zarr_path))
    pts, cols = load_point_cloud(zarr_path, n_frames=args.n_frames)
    center, radius, z_table = estimate_workspace(pts)

    print("[2/4] Loading camera configs...")
    with open(args.configs) as f:
        configs = json.load(f)
    cand_dirs = find_candidate_dirs(args.cam_dir)
    K = make_K(args.fov_deg, args.img_size)

    print("[3/4] Rendering {} camera views...".format(len(configs)))
    candidates_data = []
    for i, cfg in enumerate(configs):
        name = cfg["name"]
        folder = cand_dirs.get(name)
        info_path = os.path.join(folder, "camera_info.txt") if folder else None
        if info_path and os.path.isfile(info_path):
            info = parse_camera_info(info_path)
        else:
            pos = np.array(cfg["pos"], dtype=float)
            def _norm(v): n=np.linalg.norm(v); return v/n if n>1e-9 else v
            z_ax = _norm(np.array(center, dtype=float) - pos)
            wu = np.array([0.,0.,1.])
            if abs(float(np.dot(z_ax,wu)))>0.99: wu=np.array([0.,1.,0.])
            x_ax=_norm(np.cross(wu,z_ax)); y_ax=np.cross(z_ax,x_ax)
            R=np.column_stack([x_ax,y_ax,z_ax])
            info = {"pos":list(cfg["pos"]),"R":R,"euler":None,"quat":None,
                    "roll_deg":cfg.get("roll_deg",0.0)}

        cam_pos = np.array(info["pos"], dtype=np.float32)
        cam_R   = info.get("R")

        # Prefer real CoppeliaSim renders; fall back to point-cloud projection
        imgs_b64 = []
        if folder:
            for r in range(5):  # try reset_00 … reset_04
                png_path = os.path.join(folder, "reset_{:02d}.png".format(r))
                if os.path.isfile(png_path):
                    with open(png_path, "rb") as fh:
                        imgs_b64.append("data:image/png;base64," +
                                        base64.b64encode(fh.read()).decode())
        if not imgs_b64:
            if cam_R is not None:
                img = render_view(pts, cols, cam_pos, np.array(cam_R), K,
                                  args.img_size, args.img_size)
            else:
                img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            imgs_b64 = [img_to_b64(img)]

        fwd   = cam_R[:,2].tolist()   if cam_R is not None else [0.,0.,1.]
        R_lst = cam_R.tolist()        if cam_R is not None else [[1,0,0],[0,1,0],[0,0,1]]
        candidates_data.append({
            "idx":            i,
            "name":           name,
            "structured":     bool(cfg.get("structured", False)),
            "pos":            [round(float(v),4) for v in info["pos"]],
            "euler":          [round(float(v),3) for v in info["euler"]] if info.get("euler") else None,
            "quat":           [round(float(v),5) for v in info["quat"]]  if info.get("quat")  else None,
            "roll_deg":       round(float(cfg.get("roll_deg", info.get("roll_deg",0.0))),2),
            "dist_from_center": round(float(cfg.get("dist_from_center",0.0)),4),
            "fwd":            [round(float(v),4) for v in fwd],
            "R":              [[round(float(v),5) for v in row] for row in R_lst],
            "imgs":           imgs_b64,   # list: reset_00, reset_01, reset_02 …
        })
        print("  [{:>3}/{}] {}        ".format(i+1,len(configs),name), end="\r", flush=True)
    print()

    print("[4/4] Generating HTML...")
    zarr_label = os.path.basename(os.path.abspath(zarr_path))
    html = build_html(candidates_data, pts, cols, center, z_table, zarr_label,
                      max_cloud_pts=args.cloud_pts)
    with open(args.out, "w") as fh:
        fh.write(html)
    size_mb = os.path.getsize(args.out)/1e6
    print("[DONE]  {:.1f} MB  ->  {}".format(size_mb, args.out))
    print("  Open: file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
