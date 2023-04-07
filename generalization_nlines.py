"""
See how we can use our saccade + correction setup to do transfer / generalization.

Two envs: one with 1-4 lines, one with only 4 lines 

Two agents, same init.

One is trained only on 4 lines, the other is trained on 1-4 lines.

At the end, compare 4 possibilities:
- trained 1-4 test 1-4 (baseline for success)
- trained 4 test 1-4 (baseline for failure)
- put the 4 lines fovea with the 1-4 periphery (should succeed)
- put the 1-4 lines fovea with the 4 periphery (should fail)

"""
