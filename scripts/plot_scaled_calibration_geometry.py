#!/usr/bin/env python3
"""Plot exact scaled calibration geometry: camera frusta and point-cloud overlays."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

def frame(cam):
    T = np.eye(4); T[:3,:3] = np.asarray(cam["R"]); T[:3,3] = cam["pos"]
    return T

def load_cameras(group="G2"):
    rows = json.loads((ROOT/"instructions/orbital_cameras_grouped.json").read_text())
    row = next(x for x in rows if x["group"] == group)
    return [frame(row["left"]), frame(row["right"])]

def load_calibration(group, level):
    path = ROOT/"instructions/eval_calibrations_idlt_scaled_extreme_v01.json"
    if level == 20:
        path = ROOT/"instructions/eval_calibrations_idlt_extreme_v01.json"
    d=json.loads(path.read_text())
    key=f"seen-{group.lower()}-medium-unknown-e{level}deg-t{level}cm-v01"
    return [np.asarray(d["realizations"][key]["transforms"][c]) for c in ["orbital_left","orbital_right"]]

def frustum(ax, T, color, label, scale=.22):
    o=T[:3,3]; R=T[:3,:3]
    # Camera looks along local +Z for this schematic; show a square image plane.
    corners=np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)*scale
    pts=(R@corners.T).T+o
    for p in pts: ax.plot([o[0],p[0]],[o[1],p[1]],[o[2],p[2]],color=color,lw=.8)
    for i in range(4):
        a,b=pts[i],pts[(i+1)%4]; ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],color=color,lw=1.2)
    ax.scatter(*o,color=color,s=22,label=label)

cams=load_cameras("G2")
levels=[0,20,40,60]
fig=plt.figure(figsize=(13,10));
for j,level in enumerate(levels,1):
    ax=fig.add_subplot(2,2,j,projection="3d")
    for i,T in enumerate(cams):
        frustum(ax,T,"#555555", "nominal" if i==0 else None)
        if level:
            P=load_calibration("G2",level)[i]@T
            frustum(ax,P,"#d62728", f"{level}°/{level}cm" if i==0 else None)
    ax.set_title("Clean" if level==0 else f"Scaled {level}° / {level} cm")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend(loc="upper left",fontsize=8)
    ax.set_box_aspect((1,1,0.8))
fig.suptitle("G2 camera-frustum displacement under exact evaluation perturbations",fontsize=15)
fig.tight_layout(); fig.savefig(OUT/"scaled_calibration_camera_frusta.png",dpi=220); fig.savefig(OUT/"scaled_calibration_camera_frusta.pdf")

# Geometric point-cloud overlay: points on a canonical depth sheet, transformed by each exact camera pose.
u,v=np.meshgrid(np.linspace(-.45,.45,18),np.linspace(-.35,.35,14)); local=np.c_[u.ravel(),v.ravel(),np.ones(u.size)*1.0,np.ones(u.size)]
fig=plt.figure(figsize=(13,6))
for j,level in enumerate([20,40,60],1):
    ax=fig.add_subplot(1,3,j,projection="3d")
    for i,T in enumerate(cams):
        clean=(T@local.T).T[:,:3]; noisy=(load_calibration("G2",level)[i]@T@local.T).T[:,:3]
        ax.scatter(clean[:,0],clean[:,1],clean[:,2],s=3,c="#777777",alpha=.28)
        ax.scatter(noisy[:,0],noisy[:,1],noisy[:,2],s=5,c="#d62728",alpha=.65)
    ax.set_title(f"20/20 → scaled {level}/{level}"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_box_aspect((1,1,.8))
fig.suptitle("Clean (gray) vs perturbed (red) projected point-cloud geometry, G2",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"scaled_calibration_pointcloud_overlays.png",dpi=220); fig.savefig(OUT/"scaled_calibration_pointcloud_overlays.pdf")

# Translation-only view: deliberately ignore every rotation component.
fig, ax = plt.subplots(figsize=(10, 7))
colors={20:"#1f77b4",30:"#ff7f0e",40:"#2ca02c",50:"#d62728",60:"#9467bd"}
for i,T in enumerate(cams):
    p=T[:3,3]; ax.scatter(p[0],p[1],color="#222222",s=55,marker="o",label="nominal camera center" if i==0 else None)
    ax.text(p[0]+.012,p[1]+.012,f"C{i}",fontsize=9)
    for level in [20,30,40,50,60]:
        P=load_calibration("G2",level)[i]@T; q=P[:3,3]; d=q-p
        ax.arrow(p[0],p[1],d[0],d[1],color=colors[level],width=.0015,head_width=.025,length_includes_head=True,alpha=.8)
        ax.scatter(q[0],q[1],color=colors[level],s=20)
        if i==0: ax.text(q[0]+.01,q[1]+.01,f"{level}°/{level}cm",color=colors[level],fontsize=8)
ax.set_xlabel("camera-center x (m)"); ax.set_ylabel("camera-center y (m)"); ax.set_title("Translation-only camera-center displacement (G2; rotations ignored)")
ax.grid(alpha=.25); ax.set_aspect("equal",adjustable="box"); ax.legend(loc="best",fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"scaled_calibration_translation_only.png",dpi=220); fig.savefig(OUT/"scaled_calibration_translation_only.pdf")

fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d")
for i,T in enumerate(cams):
    p=T[:3,3]; ax.scatter(*p,color="#222222",s=55,label="nominal camera center" if i==0 else None); ax.text(*p,f" C{i}")
    for level in [20,30,40,50,60]:
        q=(load_calibration("G2",level)[i]@T)[:3,3]; d=q-p
        ax.quiver(*p,*d,color=colors[level],arrow_length_ratio=.08,linewidth=1.5)
        ax.scatter(*q,color=colors[level],s=22)
        if i==0: ax.text(*q,f" {level}/{level}",color=colors[level],fontsize=8)
ax.set_xlabel("camera-center x (m)"); ax.set_ylabel("camera-center y (m)"); ax.set_zlabel("camera-center z (m)")
ax.set_title("3D translation-only camera-center displacement (G2; rotations ignored)"); ax.legend(loc="upper left",fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"scaled_calibration_translation_only_3d.png",dpi=220); fig.savefig(OUT/"scaled_calibration_translation_only_3d.pdf")
print(OUT/"scaled_calibration_camera_frusta.png")
print(OUT/"scaled_calibration_pointcloud_overlays.png")
print(OUT/"scaled_calibration_translation_only.png")
print(OUT/"scaled_calibration_translation_only_3d.png")
