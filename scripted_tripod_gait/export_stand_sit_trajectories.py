"""Export deterministic stand-up and sit-down joint trajectories.

This is the non-rendering companion to ``render_scripted_standup.py``.  It
generates the same time-indexed 18-joint position targets without launching
Isaac Sim, so the output can be handed to another simulator or robot-side
integration agent.

Example:

    /workspace/isaaclab/_isaac_sim/python.sh \
      scripted_tripod_gait/export_stand_sit_trajectories.py \
      --mode both \
      --output scripted_tripod_gait/trajectories/stand_sit_default.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from insectoid_mini_rl.gait_reference import (  # noqa: E402
    LEG_ORDER,
    NEUTRAL_REFERENCE_RAD,
    SEGMENTS,
    _build_leg_chains,
    _foot_fk,
)


JOINT_ORDER = [
    "BL_coxa_joint",
    "BL_femur_joint",
    "BL_tibia_joint",
    "BR_coxa_joint",
    "BR_femur_joint",
    "BR_tibia_joint",
    "FL_coxa_joint",
    "FL_femur_joint",
    "FL_tibia_joint",
    "FR_coxa_joint",
    "FR_femur_joint",
    "FR_tibia_joint",
    "ML_coxa_joint",
    "ML_femur_joint",
    "ML_tibia_joint",
    "MR_coxa_joint",
    "MR_femur_joint",
    "MR_tibia_joint",
]


# Match Isaac's soft_joint_pos_limit_factor=0.95 for the URDF hard limits.
SOFT_LIMITS_RAD = {
    "coxa": (-0.994838, 0.994838),
    "femur": (-1.191187, -0.030543),
    "tibia": (0.065450, 2.552544),
}


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def joint_limits_for_order(joint_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(len(joint_order), dtype=float)
    upper = np.zeros(len(joint_order), dtype=float)
    for joint_id, joint_name in enumerate(joint_order):
        segment = joint_name.split("_")[1]
        lower[joint_id], upper[joint_id] = SOFT_LIMITS_RAD[segment]
    return lower, upper


def solve_femur_tibia(
    chain,
    desired_pos: np.ndarray,
    q_seed: np.ndarray,
    posture_weight: float,
    coxa_target: float = 0.0,
    posture_reference: np.ndarray | None = None,
) -> np.ndarray:
    lower = chain.lower[1:3]
    upper = chain.upper[1:3]
    default = chain.default_q[1:3] if posture_reference is None else posture_reference[1:3]

    def residual(q_ft: np.ndarray) -> np.ndarray:
        q = np.array((coxa_target, q_ft[0], q_ft[1]), dtype=float)
        position_error = _foot_fk(chain, q) - desired_pos
        posture_error = posture_weight * (q_ft - default)
        return np.concatenate((position_error, posture_error))

    result = least_squares(
        residual,
        x0=np.clip(q_seed[1:3], lower, upper),
        bounds=(lower, upper),
        xtol=1.0e-5,
        ftol=1.0e-5,
        gtol=1.0e-5,
        max_nfev=40,
    )
    return np.array((coxa_target, result.x[0], result.x[1]), dtype=float)


def phase_at_time(
    elapsed_s: float,
    *,
    hold_zero_s: float,
    foot_place_s: float,
    body_lift_s: float,
) -> tuple[float, float, float]:
    """Return foot-placement progress, body-lift progress, and arc phase."""

    if elapsed_s < hold_zero_s:
        return 0.0, 0.0, 0.0
    t = elapsed_s - hold_zero_s
    if t < foot_place_s:
        local = t / max(foot_place_s, 1.0e-6)
        return smoothstep(local), 0.0, math.sin(math.pi * local)
    t -= foot_place_s
    if t < body_lift_s:
        local = t / max(body_lift_s, 1.0e-6)
        return 1.0, smoothstep(local), 0.0
    return 1.0, 1.0, 0.0


def desired_startup_foot_position(
    start_pos: np.ndarray,
    stand_pos: np.ndarray,
    foot_place_progress: float,
    body_lift_progress: float,
    arc_phase: float,
    lift_height_m: float,
) -> np.ndarray:
    desired = start_pos.copy()
    desired[:2] = (1.0 - foot_place_progress) * start_pos[:2] + foot_place_progress * stand_pos[:2]
    desired[2] = (1.0 - body_lift_progress) * start_pos[2] + body_lift_progress * stand_pos[2]
    desired[2] += lift_height_m * arc_phase
    return desired


def vector_from_leg_q(joint_ids: dict[str, dict[str, int]], q_by_leg: dict[str, np.ndarray]) -> list[float]:
    q = np.zeros(len(JOINT_ORDER), dtype=float)
    for leg in LEG_ORDER:
        for segment_i, segment in enumerate(SEGMENTS):
            q[joint_ids[leg][segment]] = q_by_leg[leg][segment_i]
    return q.tolist()


def build_pose_sets(
    *,
    urdf_path: Path,
    target_height: float,
    nominal_stance_height: float,
    ik_posture_weight: float,
) -> tuple[dict, dict, dict, dict, dict]:
    lower, upper = joint_limits_for_order(JOINT_ORDER)
    chains, joint_ids = _build_leg_chains(JOINT_ORDER, lower, upper, urdf_path)

    zero_q = {leg: np.zeros(3, dtype=float) for leg in LEG_ORDER}
    stand_seed = {
        leg: np.array((0.0, NEUTRAL_REFERENCE_RAD["femur"], NEUTRAL_REFERENCE_RAD["tibia"]), dtype=float)
        for leg in LEG_ORDER
    }
    stand_foot_pos = {leg: _foot_fk(chains[leg], stand_seed[leg]) for leg in LEG_ORDER}
    height_delta = target_height - nominal_stance_height
    for leg in LEG_ORDER:
        stand_foot_pos[leg] = stand_foot_pos[leg].copy()
        stand_foot_pos[leg][2] -= height_delta

    high_stand_q = {
        leg: solve_femur_tibia(chains[leg], stand_foot_pos[leg], stand_seed[leg], ik_posture_weight)
        for leg in LEG_ORDER
    }
    low_foot_pos = {leg: _foot_fk(chains[leg], zero_q[leg]) for leg in LEG_ORDER}
    return chains, joint_ids, zero_q, high_stand_q, low_foot_pos


def coxa_start_values(args, nominal_start_q: dict[str, np.ndarray]) -> dict[str, float]:
    if not args.random_start_coxa:
        return {leg: float(nominal_start_q[leg][0]) for leg in LEG_ORDER}
    rng = np.random.default_rng(args.start_seed)
    coxa_range = math.radians(args.start_coxa_range_deg)
    return {leg: float(rng.uniform(-coxa_range, coxa_range)) for leg in LEG_ORDER}


def generate_trajectory(mode: str, args) -> dict:
    chains, joint_ids, zero_q, high_stand_q, low_foot_pos = build_pose_sets(
        urdf_path=Path(args.urdf),
        target_height=args.target_height,
        nominal_stance_height=args.nominal_stance_height,
        ik_posture_weight=args.ik_posture_weight,
    )
    nominal_start_q = high_stand_q if mode == "sitdown" else zero_q
    coxa_start = coxa_start_values(args, nominal_start_q)
    start_q = {
        leg: np.array((coxa_start[leg], nominal_start_q[leg][1], nominal_start_q[leg][2]), dtype=float)
        for leg in LEG_ORDER
    }
    start_foot_pos = {leg: _foot_fk(chains[leg], start_q[leg]) for leg in LEG_ORDER}
    stand_foot_pos = {leg: _foot_fk(chains[leg], high_stand_q[leg]) for leg in LEG_ORDER}
    q_seed = {leg: start_q[leg].copy() for leg in LEG_ORDER}

    total_s = args.hold_zero_s + args.foot_place_s + args.body_lift_s + args.settle_s
    sample_count = int(round(total_s / args.dt)) + 1
    samples = []
    for sample_idx in range(sample_count):
        time_s = min(sample_idx * args.dt, total_s)
        foot_progress, body_progress, arc_phase = phase_at_time(
            time_s,
            hold_zero_s=args.hold_zero_s,
            foot_place_s=args.foot_place_s,
            body_lift_s=args.body_lift_s,
        )
        if mode == "standup":
            root_height_reference_m = (
                (1.0 - body_progress) * args.root_start_height + body_progress * args.target_height
            )
        else:
            root_height_reference_m = (
                (1.0 - body_progress) * args.target_height + body_progress * args.root_start_height
            )
        for leg in LEG_ORDER:
            coxa_target = (1.0 - foot_progress) * coxa_start[leg]
            if mode == "standup":
                desired = desired_startup_foot_position(
                    start_foot_pos[leg],
                    stand_foot_pos[leg],
                    foot_progress,
                    body_progress,
                    arc_phase,
                    args.lift_height_m,
                )
                posture_reference = None
            elif body_progress <= 0.0:
                desired = (1.0 - foot_progress) * start_foot_pos[leg] + foot_progress * stand_foot_pos[leg]
                posture_reference = None
            else:
                desired = (1.0 - body_progress) * stand_foot_pos[leg] + body_progress * low_foot_pos[leg]
                desired[2] += args.lift_height_m * math.sin(math.pi * body_progress)
                posture_reference = zero_q[leg]

            if mode == "sitdown" and body_progress >= 1.0:
                q_seed[leg] = zero_q[leg].copy()
            else:
                q_seed[leg] = solve_femur_tibia(
                    chains[leg],
                    desired,
                    q_seed[leg],
                    args.ik_posture_weight,
                    coxa_target=coxa_target,
                    posture_reference=posture_reference,
                )

        samples.append(
            {
                "time_s": round(time_s, 6),
                "root_height_reference_m": round(float(root_height_reference_m), 9),
                "phase": {
                    "foot_place": round(float(foot_progress), 9),
                    "body_lift": round(float(body_progress), 9),
                    "arc": round(float(arc_phase), 9),
                },
                "joint_position_rad": vector_from_leg_q(joint_ids, q_seed),
            }
        )

    add_velocities(samples, args.dt)
    return {
        "mode": mode,
        "dt_s": args.dt,
        "duration_s": total_s,
        "joint_order": JOINT_ORDER,
        "samples": samples,
    }


def add_velocities(samples: list[dict], dt: float) -> None:
    positions = np.array([sample["joint_position_rad"] for sample in samples], dtype=float)
    velocities = np.zeros_like(positions)
    if len(samples) > 1:
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
    if len(samples) > 2:
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    for sample, velocity in zip(samples, velocities, strict=True):
        sample["joint_velocity_radps"] = velocity.tolist()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    trajectories = payload["trajectories"]
    for mode, trajectory in trajectories.items():
        for sample in trajectory["samples"]:
            row = {
                "mode": mode,
                "time_s": sample["time_s"],
                "root_height_reference_m": sample["root_height_reference_m"],
            }
            for joint_name, q, qd in zip(
                payload["joint_order"],
                sample["joint_position_rad"],
                sample["joint_velocity_radps"],
                strict=True,
            ):
                row[f"{joint_name}_pos_rad"] = q
                row[f"{joint_name}_vel_radps"] = qd
            rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else ["mode", "time_s"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standup", "sitdown", "both"), default="both")
    parser.add_argument("--output", default="scripted_tripod_gait/trajectories/stand_sit_default.json")
    parser.add_argument("--format", choices=("json", "csv"), default=None)
    parser.add_argument("--urdf", default=str(REPO_ROOT / "URDF_description" / "urdf" / "URDF.urdf"))
    parser.add_argument("--dt", type=float, default=0.02, help="Sample period. Default is 50 Hz.")
    parser.add_argument("--root-start-height", type=float, default=0.055)
    parser.add_argument("--target-height", type=float, default=0.15)
    parser.add_argument("--nominal-stance-height", type=float, default=0.15)
    parser.add_argument("--hold-zero-s", type=float, default=0.50)
    parser.add_argument("--foot-place-s", type=float, default=1.50)
    parser.add_argument("--body-lift-s", type=float, default=1.50)
    parser.add_argument("--settle-s", type=float, default=1.00)
    parser.add_argument("--lift-height-m", type=float, default=0.050)
    parser.add_argument("--random-start-coxa", action="store_true")
    parser.add_argument("--start-coxa-range-deg", type=float, default=35.0)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--ik-posture-weight", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ("standup", "sitdown") if args.mode == "both" else (args.mode,)
    trajectories = {mode: generate_trajectory(mode, args) for mode in modes}
    payload = {
        "schema": "insectoid_mini_stand_sit_trajectory_v1",
        "description": "Time-indexed joint position targets generated from the deterministic stand/sit controller.",
        "joint_order": JOINT_ORDER,
        "units": {"time": "s", "position": "rad", "velocity": "rad/s"},
        "parameters": {
            "dt_s": args.dt,
            "root_start_height_m": args.root_start_height,
            "target_height_m": args.target_height,
            "nominal_stance_height_m": args.nominal_stance_height,
            "hold_zero_s": args.hold_zero_s,
            "foot_place_s": args.foot_place_s,
            "body_lift_s": args.body_lift_s,
            "settle_s": args.settle_s,
            "lift_height_m": args.lift_height_m,
            "random_start_coxa": args.random_start_coxa,
            "start_coxa_range_deg": args.start_coxa_range_deg,
            "start_seed": args.start_seed,
            "ik_posture_weight": args.ik_posture_weight,
        },
        "trajectories": trajectories,
    }

    output = Path(args.output)
    output_format = args.format or output.suffix.lstrip(".").lower() or "json"
    if output_format == "json":
        write_json(output, payload)
    elif output_format == "csv":
        write_csv(output, payload)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    for mode, trajectory in trajectories.items():
        print(
            f"EXPORTED {mode} samples={len(trajectory['samples'])} "
            f"duration_s={trajectory['duration_s']:.3f} dt_s={trajectory['dt_s']:.3f}",
            flush=True,
        )
    print(f"OUTPUT {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
