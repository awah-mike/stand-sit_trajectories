# Scripted Tripod Gait Sandbox

This folder is intentionally separate from the RL environment and reward code.
It is for open-loop tripod-gait experiments that can later become a reference
trajectory, imitation target, or residual-RL prior.

The renderer uses the existing Isaac Lab DirectRLEnv only as a simulator/video
wrapper. It does not load a policy and does not edit `insectoid_mini_rl`.

## First render

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_scripted_tripod.py \
  --headless --enable_cameras
```

The script starts from the standing pose, waits briefly, then alternates:

- Tripod A: `BR`, `FR`, `ML`
- Tripod B: `BL`, `FL`, `MR`

The stance tripod sweeps its coxas backward against the ground while the swing
tripod lifts, advances, and places its feet forward.

## Useful tuning flags

- `--cycle-s`: total gait cycle duration.
- `--coxa-amp-deg`: forward/backward sweep amplitude.
- `--femur-lift-deg`: femur lift during swing.
- `--tibia-lift-deg`: tibia fold/unfold during swing.
- `--manual-coxa-signs BL BR FL FR ML MR`: override automatically calibrated
  coxa signs if the sweep direction is wrong.
- `--video-length`: number of env steps to record.

## Straight-Line Task-Space Version

`render_task_space_tripod.py` generates desired foot positions first, then uses
per-leg IK to solve coxa/femur/tibia targets. This keeps each foot's lateral
coordinate fixed in the base frame for forward walking, or keeps each foot's
forward coordinate fixed for sideways walking.

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_task_space_tripod.py \
  --headless --enable_cameras \
  --step-length-m 0.10 \
  --lift-height-m 0.035 \
  --video-name task_space_tripod_v1
```

For this script:

- stance path is a straight line from front to rear at standing foot height;
- swing path is a straight line from rear to front with sinusoidal vertical
  clearance;
- `--step-length-m` controls the full front-to-rear displacement.

Sideways gait uses the same tripod timing and IK solver, but moves the foot
line along body `X` instead of body `Y`:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_task_space_tripod.py \
  --headless --enable_cameras \
  --travel-axis lateral \
  --travel-sign 1 \
  --step-length-m 0.08 \
  --lift-height-m 0.055 \
  --video-name task_space_sideways_xpos_v1
```

Use `--travel-sign -1` for the opposite lateral direction. The `lateral` axis
is inferred from the named left/right feet in the robot geometry; raw `x` and
`y` axes are still available for debugging.

## Residual RL Integration

The DirectRL environment now has a residual reference-gait scaffold:

- `insectoid_mini_rl.gait_reference` builds a phase-indexed task-space tripod
  joint table.
- `InsectoidMiniDirectEnvCfg.use_reference_gait = True` enables residual
  control.
- Policy actions are interpreted as bounded corrections:

```text
joint_target = q_reference(phase) + reference_residual_scale * action
```

The initial settings use:

```text
step_length_m = 0.10
lift_height_m = 0.055
cycle_s = 1.2
reference_residual_scale = 0.18 rad
fixed_reference_phase_reset = True
```

The observation keeps the existing 74D shape, but joint position observations
become `joint_pos - q_reference(phase)` while reference mode is enabled. The
reward logs these additional terms:

- `reference_joint_tracking`
- `reference_coxa_tracking`
- `reference_residual_action_l2`

The next training command is the normal DirectRL PPO command:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/train_rsl_rl.py \
  --task Isaac-InsectoidMini-Flat-Direct-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 100
```

## Scripted Startup / Stand-Up Controller

`render_scripted_standup.py` is a deterministic startup routine, not a policy.
It exists because the learned stand-height controller tended to find awkward
solutions. The script starts from a low zero-joint floor pose, locks all coxas
to zero, and uses only femur/tibia IK.

Current intended startup sequence:

```text
phase 1: all six feet move simultaneously to stance XY through an arc
phase 2: stance XY is held fixed while the femurs/tibias extend to lift the torso
```

Current test renders:

```text
scripted_standup_v1-step-0.mp4
root_start_z=0.055 m, root_end_z=0.157 m, root_max_z=0.252 m, lift=0.050 m

scripted_standup_v2_lift_025m-step-0.mp4
root_start_z=0.055 m, root_end_z=0.163 m, root_max_z=0.252 m, lift=0.025 m

scripted_standup_v3_slow_start-step-0.mp4
root_start_z=0.075 m, root_end_z=0.165 m, root_max_z=0.249 m, lift=0.025 m, pair_move_s=1.6

scripted_standup_two_phase_v1-step-0.mp4
root_start_z=0.055 m, root_end_z=0.164 m, root_max_z=0.263 m, lift=0.025 m,
foot_place_s=1.4, body_lift_s=1.6

scripted_standup_two_phase_no_reset_v1-step-0.mp4
root_start_z=0.055 m, root_end_z=0.161 m, root_max_z=0.161 m, lift=0.025 m,
foot_place_s=1.4, body_lift_s=1.6

scripted_standup_h020_random_coxa_v1-step-0.mp4
root_start_z=0.055 m, root_end_z=0.210 m, root_max_z=0.210 m, target_height=0.200 m,
random coxa start enabled, femur/tibia start at zero

scripted_sitdown_h020_random_coxa_v3-step-0.mp4
mode=sitdown, root_start_z=0.200 m, root_end_z=0.061 m, root_max_z=0.242 m,
random coxa start enabled, final all joints exactly zero
```

Important playback detail:

```text
The startup pose has root_start_z below the locomotion task's normal fall
threshold. For startup renders, render_scripted_standup.py disables
base-contact/low-height resets and clears episode_length_buf after the manual
zero-pose reset. Without this, the DirectRL env repeatedly resets the robot and
randomizes yaw, producing a bad video with frame-to-frame yaw jumps.
```

Example command:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_scripted_standup.py \
  --headless --enable_cameras \
  --video-name scripted_standup_two_phase_no_reset_v1 \
  --video-length 420 \
  --root-start-height 0.055 \
  --lift-height-m 0.025 \
  --foot-place-s 1.4 \
  --body-lift-s 1.6
```

Random coxa startup test:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_scripted_standup.py \
  --headless --enable_cameras \
  --video-name scripted_standup_h020_random_coxa_v1 \
  --video-length 480 \
  --root-start-height 0.055 \
  --target-height 0.20 \
  --nominal-stance-height 0.15 \
  --lift-height-m 0.025 \
  --foot-place-s 1.6 \
  --body-lift-s 1.8 \
  --random-start-coxa \
  --start-coxa-range-deg 35 \
  --start-seed 11
```

Sit-down test from a high stance:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/insectoid_mini/scripted_tripod_gait/render_scripted_standup.py \
  --headless --enable_cameras \
  --mode sitdown \
  --video-name scripted_sitdown_h020_random_coxa_v3 \
  --video-length 480 \
  --root-start-height 0.055 \
  --target-height 0.20 \
  --nominal-stance-height 0.15 \
  --lift-height-m 0.025 \
  --foot-place-s 1.4 \
  --body-lift-s 1.8 \
  --random-start-coxa \
  --start-coxa-range-deg 25 \
  --start-seed 13
```

Policy handoff note:

```text
Startup should remain a separate deterministic controller. The locomotion
policy should be responsible for walking after the robot is already standing.
If we need different torso heights during locomotion, the right path is to make
the reference gait/body-height parameter configurable and test the existing
walking policy at those heights, not to train a startup policy.
```

## Exporting Stand/Sit Trajectories

`export_stand_sit_trajectories.py` writes the deterministic stand-up and
sit-down motions as time-indexed joint targets without launching Isaac Sim.
The default export is 50 Hz and includes both joint positions and finite-
difference joint velocities in the policy/deployment joint order.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripted_tripod_gait/export_stand_sit_trajectories.py \
  --mode both \
  --output scripted_tripod_gait/trajectories/stand_sit_default.json
```

CSV is also supported:

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripted_tripod_gait/export_stand_sit_trajectories.py \
  --mode both \
  --format csv \
  --output /tmp/stand_sit_default.csv
```
