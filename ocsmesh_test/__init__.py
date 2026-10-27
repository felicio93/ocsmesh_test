"""ocsmesh_test: benchmark and regression-check harness for OCSMesh.

Pipeline (one YAML config drives everything):

    ocsmesh-test data    -c configs/nj_gaea.yaml      # download/crop DEMs, write manifest
    ocsmesh-test slurm   -c configs/nj_gaea.yaml      # write (and optionally submit) SLURM scripts
    ocsmesh-test run     -c ... --test 0 --mode serial --method exact   # one case (called by SLURM)
    ocsmesh-test report  -c configs/nj_gaea.yaml      # summary.json files -> markdown report
    ocsmesh-test check                                 # regression checks for OCSMesh fixes -> markdown

Nothing here changes OCSMesh. Everything is driven through its public API.
"""

__version__ = "0.1.0"
