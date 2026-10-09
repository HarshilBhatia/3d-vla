# Run board

`index.html` is a self-contained page: runs first, toggles that include or exclude each run/checkpoint from every chart, and the analysis TODOs with results and conclusions. Its data is embedded, so it also works opened straight from disk. Chart.js and the fonts load from public CDNs.

- Rebuild after new results land: `python scripts/analysis/build_dashboard.py`. It reads the eval campaign folders under `/grogu/datasets/hbhatia/3dfa_online_eval_{100,20}rollouts` and `todos.json`, and writes `data.json` and `index.html` from `template.html`.
- Edit the TODO entries in `todos.json`, and the run list in `RUNS` / `SERIES` / `LADDERS` at the top of the build script.
- Serve on the cluster login node: `python -m http.server 8765 --bind 127.0.0.1 --directory docs/research/dashboard`, then from a laptop `ssh -N -L 8765:127.0.0.1:8765 harshilb@grogu.ri.cmu.edu` and open http://localhost:8765.
