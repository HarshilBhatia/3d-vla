#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'docs/figures'; OUT.mkdir(parents=True,exist_ok=True)
rows=json.loads((ROOT/'instructions/orbital_cameras_grouped.json').read_text()); g=next(x for x in rows if x['group']=='G7')
p=[np.asarray(g['left']['pos'],float),np.asarray(g['right']['pos'],float)]
d=p[0]-p[1]; d=d/np.linalg.norm(d)
colors={10:'#1f77b4',20:'#ff7f0e',40:'#2ca02c',60:'#9467bd'}
fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection='3d')
for i,origin in enumerate(p):
 ax.scatter(*origin,c='#222',s=65,label='nominal camera center' if i==0 else None); ax.text(*origin,f' C{i}')
 for cm in [10,20,40,60]:
  delta=(d if i==0 else -d)*cm/100; q=origin+delta
  ax.quiver(*origin,*delta,color=colors[cm],arrow_length_ratio=.08,linewidth=2)
  ax.scatter(*q,c=colors[cm],s=28); ax.text(*q,f' {cm}cm',color=colors[cm],fontsize=9)
ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
ax.set_title('G7 unseen group: opposing-camera translation only (rotation unchanged)')
ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(OUT/'g7_opposite_camera_translation_3d.png',dpi=220);fig.savefig(OUT/'g7_opposite_camera_translation_3d.pdf')
print(OUT/'g7_opposite_camera_translation_3d.png')

# Canonical point-cloud overlay: a one-metre depth sheet in each camera frame.
u,v=np.meshgrid(np.linspace(-.45,.45,20),np.linspace(-.32,.32,15))
local=np.c_[u.ravel(),v.ravel(),np.ones(u.size),np.ones(u.size)]
def pose(cam):
 T=np.eye(4);T[:3,:3]=np.asarray(cam['R']);T[:3,3]=cam['pos'];return T
poses=[pose(g['left']),pose(g['right'])]
fig=plt.figure(figsize=(14,8))
for j,cm in enumerate([10,20,40,60],1):
 ax=fig.add_subplot(2,2,j,projection='3d')
 for i,T in enumerate(poses):
  clean=(T@local.T).T[:,:3]; delta=(d if i==0 else -d)*cm/100
  E=np.eye(4);E[:3,3]=delta; noisy=(E@T@local.T).T[:,:3]
  ax.scatter(clean[:,0],clean[:,1],clean[:,2],s=3,c='#777777',alpha=.22)
  ax.scatter(noisy[:,0],noisy[:,1],noisy[:,2],s=4,c=colors[cm],alpha=.62)
 ax.set_title(f'clean (gray) vs opposing {cm} cm (color)');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.set_box_aspect((1,1,.8))
fig.suptitle('G7 canonical point-cloud motion under opposing-camera translations',fontsize=15)
fig.tight_layout();fig.savefig(OUT/'g7_opposite_camera_pointcloud_3d.png',dpi=220);fig.savefig(OUT/'g7_opposite_camera_pointcloud_3d.pdf')
print(OUT/'g7_opposite_camera_pointcloud_3d.png')
