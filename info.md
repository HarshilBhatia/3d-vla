Here's the full picture of how orbital camera poses were computed:

  ---
  Orbital Camera Pose Computation

  The poses are defined in scripts/rlbench/visualize_cameras_rerun.py:59 (orbital_cameras()) and the
  resulting poses are stored in orbital_cameras.json.

  Position (spherical coordinates)

  Each camera is placed on a sphere of radius 2 × workspace_radius around the workspace centroid. Two
  azimuths are used:

  ┌──────┬──────────────┬────────────────┐
  │ Side │ Base azimuth │ Interpretation │
  ├──────┼──────────────┼────────────────┤
  │ L    │ 45°          │ Front-left     │
  ├──────┼──────────────┼────────────────┤
  │ R    │ 315° (−45°)  │ Front-right    │
  └──────┴──────────────┴────────────────┘

  For each side, cameras are placed at 3 elevation levels: 10°, 35°, 60°. At each elevation, 2 cameras are
  generated with azimuth offsets of −5° and +5° from the base, giving 12 cameras total (2 sides × 3
  elevations × 2 per elevation).

  Cartesian position:
  pos = [
      cx + r * cos(el) * cos(az),
      cy + r * cos(el) * sin(az),
      cz + r * sin(el),
  ]
  pos[2] = max(pos[2], z_table + 0.05)  # floor clamp

  Small random elevation jitter U(−2°, 2°) is applied to each elevation angle.

  Rotation (look-at)

  Each camera's rotation matrix is computed by look_at(pos, center) (visualize_cameras_rerun.py:42):

  1. Z-axis (forward): unit vector from pos → center
  2. X-axis (right): normalize(world_up × Z)
  3. Y-axis (down-in-image): Z × X

  Columns of the 3×3 rotation matrix are [X, Y, Z] — this is a camera-to-world rotation.

  Workspace geometry (center/radius)

  The center and radius are estimated from a merged point cloud of actual scene observations:
  - center = median XYZ of all points
  - radius = 80th percentile of distances from centroid
  - z_table = 15th percentile of Z values
