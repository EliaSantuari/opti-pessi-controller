#!/usr/bin/env python3
"""Fast closed-loop test of the obstacle formulations, without Gazebo (seconds instead of hours).

Same OCP of the controller (lipm_model.create_lipm_ocp, generated in a temporary folder: c_generated_code is not
touched), same loop of AcadosLeggedController::mpcThreadLoop (parameters, heading, warm start of the hyperplanes)
and same obstacles of ObstacleManager.hpp (trapezoidal back and forth, circular). One solve per step, the robot
follows the LIPM of the OCP with a random tracking error (--noise) on CoM position and velocity at each step.

Collision: the obstacle disc (true radius, no safety factor) touches the rectangle of the hips (the robot body of the
OCP), checked at 10 instants inside each step with the LIPM trajectory and the obstacle at the same time.

Usage (acados python env, see instruction.md):
  LD_LIBRARY_PATH=$HOME/acados/lib PYTHONPATH= ~/opti_pessi_acados/acados_env/bin/python closed_loop_test.py
  ... closed_loop_test.py --variants leonardelli,support,new --scenarios A,B,C,D --reps 5 --plot

Variants: margin_kind of config.yaml (leonardelli, min_max, support), the rest of config.yaml unchanged.
Solve time: internal time of acados (time_tot), as the controller log.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VARIANTS = ("leonardelli", "min_max", "support")

# Scenarios of the note (sec. 3): robot (0, 0) -> goal, obstacle (type, start, end/centre), speeds
SCENARIOS = {
    "A": dict(name="crossing",  goal=(4.0, 0.0), type="dynamic",  pos=(2.0, 1.2), end=(2.0, -5.0), speeds=(0.4, 0.6, 0.8)),
    "B": dict(name="lane",      goal=(4.0, 0.0), type="dynamic",  pos=(5.0, 0.7), end=(-4.0, 0.7), speeds=(0.4, 0.6, 0.8)),
    "C": dict(name="chase",     goal=(6.0, 0.0), type="dynamic",  pos=(1.2, 0.0), end=(10.0, 0.0), speeds=(0.3, 0.5, 0.7)),
    "D": dict(name="circle",    goal=(4.0, 0.0), type="circular", pos=(2.0, 0.9), center=(2.0, 0.0), speeds=(0.3, 0.5, 0.7)),
    "E": dict(name="head-on",   goal=(4.0, 0.0), type="dynamic",  pos=(5.0, 0.1), end=(-4.0, 0.1), speeds=(0.4, 0.6, 0.8)),
    # G: walks parallel to the lane of the robot (y = 0.9, towards -x); when the robot CoM is closer than d_trig it lunges at the robot for
    # t_lunge s (desired velocity v_max towards the robot, |dv/dt| <= a_max), then goes on straight: the worst case
    # the pessimistic set is made for
    "G": dict(name="ambush", goal=(4.0, 0.0), type="ambush", start=(4.0, 0.9), d_trig=1.5, t_lunge=2.0,
              speeds=(0.4, 0.6, 0.8)),
    # H: crossing like A (x = 2, towards -y); when the robot CoM is closer than d_trig it lunges at the robot
    "H": dict(name="crossing lunge", goal=(4.0, 0.0), type="ambush", start=(2.0, 1.2), dir=(0.0, -1.0), d_trig=1.0,
              t_lunge=1.5, speeds=(0.4, 0.6, 0.8)),
}
R_OBS, V_MAX_OBS, A_MAX_OBS = 0.2, 1.0, 1.0
T_MAX = 40.0
DT_SUB = 0.03            # s, the MPC solves again every DT_SUB (the controller solves continuously)
VERBOSE = bool(int(os.environ.get("CLT_VERBOSE", "0")))


class DictToObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, DictToObject(v) if isinstance(v, dict) else v)


def load_cfg(variant):
    with open(os.path.join(HERE, "..", "config", "config.yaml")) as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    raw["obstacles"]["margin_kind"] = variant
    return raw, DictToObject(raw)


# ---------------- obstacles (same as ObstacleManager.hpp) ----------------
class DynamicObs:
    def __init__(self, pos, end, speed, a_max):
        self.s0, self.e = np.array(pos, float), np.array(end, float)
        self.L = np.linalg.norm(self.e - self.s0)
        self.acc = max(a_max, 1e-3)
        self.vp = min(speed, math.sqrt(self.acc * self.L))
        self.ta = self.vp / self.acc
        self.tl = 2 * self.ta + (self.L - self.vp * self.ta) / self.vp

    def state(self, t):
        if t <= 0:
            return self.s0.copy(), np.zeros(2)
        tau = math.fmod(t, 2 * self.tl)
        back = tau > self.tl
        if back:
            tau -= self.tl
        if tau < self.ta:
            s, v = 0.5 * self.acc * tau**2, self.acc * tau
        elif tau < self.tl - self.ta:
            s, v = 0.5 * self.vp * self.ta + self.vp * (tau - self.ta), self.vp
        else:
            r = self.tl - tau
            s, v = self.L - 0.5 * self.acc * r**2, self.acc * r
        if back:
            s, v = self.L - s, -v
        u = (self.e - self.s0) / self.L
        return self.s0 + s * u, v * u


class CircularObs:
    def __init__(self, pos, center, speed, clockwise=False):
        self.c = np.array(center, float)
        d = np.array(pos, float) - self.c
        self.R = np.linalg.norm(d)
        self.a0 = math.atan2(d[1], d[0])
        self.w = (-1 if clockwise else 1) * speed / self.R

    def state(self, t):
        t = max(t, 0.0)
        a = self.a0 + self.w * t
        p = self.c + self.R * np.array([math.cos(a), math.sin(a)])
        v = self.w * self.R * np.array([-math.sin(a), math.cos(a)])
        return p, v


class AmbushObs:
    """Stateful: set_robot() before each state(t) of the loop, state(t) for t ahead extrapolates at constant velocity."""
    def __init__(self, pos, speed, d_trig, t_lunge, v_max, a_max, direction=(-1.0, 0.0)):
        self.p = np.array(pos, float)
        self.v = speed * np.array(direction, float)
        self.dir0 = np.array(direction, float)
        self.t = 0.0
        self.speed, self.d_trig, self.t_lunge, self.v_max, self.a_max = speed, d_trig, t_lunge, v_max, a_max
        self.t_start = None
        self.robot = np.zeros(2)

    def set_robot(self, xy):
        self.robot = np.array(xy, float)

    def _advance(self, t):
        h = 0.005
        while self.t + 1e-9 < t:
            dt = min(h, t - self.t)
            to_r = self.robot - self.p
            dist = np.linalg.norm(to_r)
            if self.t_start is None and dist < self.d_trig:
                self.t_start = self.t
            lunging = self.t_start is not None and self.t - self.t_start < self.t_lunge
            if lunging:
                vd = self.v_max * to_r / max(dist, 1e-6)
            else:
                nv = np.linalg.norm(self.v)
                vd = self.speed * self.v / nv if nv > 1e-6 else self.speed * self.dir0
            dv = vd - self.v
            nd = np.linalg.norm(dv)
            if nd > self.a_max * dt:
                dv *= self.a_max * dt / nd
            self.v = self.v + dv
            self.p = self.p + self.v * dt
            self.t += dt

    def state(self, t):
        if t > self.t:
            if t - self.t > 0.1:            # loop time: really advance
                self._advance(t)
            else:                           # sample inside the step: extrapolate
                return self.p + self.v * (t - self.t), self.v.copy()
        return self.p.copy(), self.v.copy()


def make_obstacle(sc, speed):
    if sc["type"] == "ambush":
        return AmbushObs(sc["start"], speed, sc["d_trig"], sc["t_lunge"], V_MAX_OBS, A_MAX_OBS,
                         sc.get("dir", (-1.0, 0.0)))
    if sc["type"] == "dynamic":
        return DynamicObs(sc["pos"], sc["end"], speed, A_MAX_OBS)
    return CircularObs(sc["pos"], sc["center"], speed)


# ---------------- support margin (same as ObstacleManager.hpp) ----------------
def support_margin(u0, t, v_max, a_max):
    u0 = min(max(u0, -v_max), v_max)
    a_max = max(a_max, 1e-3)
    ts = (v_max - u0) / a_max
    return v_max * t - 0.5 * a_max * ts**2 + 0.5 * a_max * max(ts - t, 0.0)**2


def obstacle_margin(kind, ax, ay, o, t):
    u0 = -(ax * o["vx"] + ay * o["vy"]) + o["e_v"]
    if kind == "leonardelli":
        return o["vmax"] * t
    if kind == "min_max":
        return min(o["vmax"] * t, u0 * t + 0.5 * o["amax"] * t * t)
    return support_margin(u0, t, o["vmax"], o["amax"])


def dist_point_rect(p, c, th, hx, hy):
    """Distance of point p from the rectangle (centre c, yaw th, half sizes hx, hy), 0 inside."""
    d = p - c
    ct, st = math.cos(th), math.sin(th)
    lx, ly = ct * d[0] + st * d[1], -st * d[0] + ct * d[1]
    qx, qy = max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0)
    return math.hypot(qx, qy)


# ---------------- worker: one variant, all the runs ----------------
def worker(variant, scen_keys, reps, noise, seed, out_json, export_root, record=False, speed_idx=None):
    from lipm_model import create_lipm_ocp

    raw, cfg = load_cfg(variant)
    kind = cfg.obstacles.margin_kind
    k_safe = cfg.obstacles.safety_factor
    N = cfg.simulation.N_horizon
    NO = cfg.obstacles.N_OBS_MAX
    nub = 8 + 3 * NO
    off_x, off_y = cfg.robot.off_x, cfg.robot.off_y
    w = math.sqrt(cfg.robot.g / cfg.robot.h_com)
    step_nom = (cfg.limits.dt_min + cfg.limits.dt_max) / (2 * cfg.simulation.step_per_phase)
    hip_off = {0: (off_x, off_y), 1: (off_x, -off_y), 2: (-off_x, off_y), 3: (-off_x, -off_y)}  # LF RF LH RH
    pairs = [(1, 2), (0, 3)]

    export_dir = os.path.join(export_root, f"codegen_{variant}")
    cwd = os.getcwd()
    os.makedirs(export_dir, exist_ok=True)
    os.chdir(export_root)
    solver = create_lipm_ocp(cfg=cfg, N=N, c_target=np.array([4.0, 0.0]), theta_target=0.0,
                             x_init=np.zeros(22), export_dir=export_dir)
    os.chdir(cwd)
    lbu0 = np.array(solver.ocp.constraints.lbu, float)
    ubu0 = np.array(solver.ocp.constraints.ubu, float)
    nb_branch = 4 + 3 * NO
    ny = solver.ocp.cost.yref.size
    nyn = solver.ocp.cost.yref_e.size

    rng = np.random.default_rng(seed)
    results = []

    for sk in scen_keys:
        sc = SCENARIOS[sk]
        goal = np.array(sc["goal"])
        speeds = sc["speeds"] if speed_idx is None else [sc["speeds"][speed_idx]]
        for speed in speeds:
            for rep in range(reps):
                obs_model = make_obstacle(sc, speed)
                t_shift = rng.uniform(-0.3, 0.3) if reps > 1 else 0.0     # start of the obstacle not always the same
                stance = 1
                x = np.zeros(11)
                x[6:8] = hip_off[pairs[stance][0]]
                x[8:10] = hip_off[pairs[stance][1]]
                x_prev = [np.concatenate([x, x]) for _ in range(N + 1)]
                u_init = np.zeros(2 * nub)
                u_init[7] = u_init[nub + 7] = cfg.limits.dt_max / cfg.simulation.step_per_phase
                u_prev = [u_init.copy() for _ in range(N)]
                step_T = step_nom         # duration of the step in progress
                t_left = step_T           # time left of the step in progress (gait.timeLeft())
                new_step = False          # warm start shifted by one node only after a touchdown
                u0 = u_init.copy()
                u0[4] = 0.5
                t = 0.0
                path = [x[0:2].copy()]
                obs_path = []
                min_clear = 1e9
                collided = False
                fails = 0
                solve_times = []
                reached = False
                last_ok = True
                frames = []

                while t < T_MAX:
                    if hasattr(obs_model, "set_robot"):
                        obs_model.set_robot(x[0:2])
                        obs_model._advance(t + t_shift)
                    yo, vo = obs_model.state(t + t_shift)
                    o = dict(x=yo[0], y=yo[1], r=R_OBS, vmax=V_MAX_OBS, amax=A_MAX_OBS,
                             vx=vo[0], vy=vo[1], e_v=cfg.obstacles.vel_uncertainty, t0=0.0)
                    slots = [o] + [dict(x=15.0 + 2 * i, y=15.0, r=0.0, vmax=0.0, amax=1.0, vx=0.0, vy=0.0,
                                        e_v=0.0, t0=0.0) for i in range(1, NO)]

                    dist_goal = np.linalg.norm(goal - x[0:2])
                    if dist_goal < cfg.simulation.goal_tolerance:
                        reached = True
                        break

                    # ---- parameters (as mpcThreadLoop) ----
                    bearing = math.atan2(goal[1] - x[1], goal[0] - x[0])
                    d_th = math.remainder(bearing - x[2], 2 * math.pi)
                    th_path = bearing + math.pi if abs(d_th) > math.pi / 2 else bearing
                    w_path = cfg.heading.path_weight if dist_goal > cfg.heading.blend_dist[0] else 0.0
                    x[10] = 0.0
                    x0 = np.concatenate([x, x])
                    solver.set(0, "lbx", x0)
                    solver.set(0, "ubx", x0)
                    for k in range(N + 1):
                        pp = pairs[(1 - stance + k) % 2]
                        p = [*hip_off[pp[0]], *hip_off[pp[1]]]
                        for s in slots:
                            p += [s["x"], s["y"], s["r"], s["vmax"], s["amax"], s["vx"], s["vy"], s["e_v"], s["t0"]]
                        p += [1.0 - w_path, th_path]
                        solver.set(k, "p", np.array(p))
                    lbu, ubu = lbu0.copy(), ubu0.copy()
                    lbu[3] = ubu[3] = lbu[3 + nb_branch] = ubu[3 + nb_branch] = max(t_left, 1e-3)
                    solver.constraints_set(0, "lbu", lbu)
                    solver.constraints_set(0, "ubu", ubu)
                    yref = np.zeros(ny)
                    yref[0:2] = yref[17:19] = goal
                    yref_e = np.zeros(nyn)
                    yref_e[0:2] = yref_e[6:8] = goal
                    for k in range(N):
                        solver.cost_set(k, "yref", yref)
                    solver.cost_set(N, "yref", yref_e)

                    # ---- warm start (as setWarmStart: shift only after a touchdown) ----
                    shift = 1 if new_step else 0
                    new_step = False
                    t_node = [0.0, 0.0]
                    for k in range(N):
                        xp = x_prev[min(k + shift, N)]
                        up = u_prev[min(k + shift, N - 1)]
                        ug = np.zeros(2 * nub)
                        xg = xp.copy()
                        pp = pairs[(1 - stance + k) % 2]
                        for br in range(2):
                            ix, iu = 11 * br, nub * br
                            c = xp[ix:ix + 2]
                            ct, st = math.cos(xp[ix + 2]), math.sin(xp[ix + 2])
                            hips = [c + np.array([ct * hip_off[f][0] - st * hip_off[f][1],
                                                  st * hip_off[f][0] + ct * hip_off[f][1]]) for f in pp]
                            ug[iu:iu + 2], ug[iu + 2:iu + 4] = hips[0], hips[1]
                            ug[iu + 4] = 0.5
                            ug[iu + 7] = max(t_left, 1e-3) if k == 0 else up[iu + 7]
                            for i, s in enumerate(slots):
                                tt = s["t0"] + t_node[br]
                                pess = br == 1
                                if pess:      # towards the nominal position y0 + v0 t, as setWarmStart
                                    tgt = np.array([s["x"] + s["vx"] * tt, s["y"] + s["vy"] * tt])
                                else:
                                    tgt = np.array([s["x"], s["y"]])
                                cands = [c, hips[0], hips[1]]
                                dd = [np.linalg.norm(tgt - q) for q in cands]
                                j = int(np.argmin(dd))
                                pt = cands[j]
                                a = (tgt - pt) / (dd[j] + 1e-5)
                                ref = np.array([s["x"], s["y"]])
                                if pess:      # set at the end of node k (lipm_model.py)
                                    m = k_safe * (s["r"] + obstacle_margin(kind, a[0], a[1], s, tt + ug[iu + 7]))
                                else:         # optimistic: obstacle standing still
                                    m = k_safe * s["r"]
                                gap = a @ (ref - pt) - m
                                b = m + 0.5 * max(gap, 0.0) - a @ ref
                                ug[iu + 8 + 3 * i: iu + 11 + 3 * i] = [a[0], a[1], b]
                            xg[ix + 10] = t_node[br]
                            t_node[br] += ug[iu + 7]
                        if k == 0:
                            ug[0:8] = ug[nub:nub + 8]
                        else:
                            solver.set(k, "x", xg)
                        solver.set(k, "u", ug)
                    solver.set(N, "x", xg)

                    tic = time.perf_counter()
                    status = solver.solve()
                    solve_times.append(solver.get_stats("time_tot"))    # internal acados time, as the controller

                    if status in (0, 2):
                        x_prev = [solver.get(k, "x") for k in range(N + 1)]
                        u_prev = [solver.get(k, "u") for k in range(N)]
                        u0 = u_prev[0].copy()
                        last_ok = True
                    else:
                        # as the controller: keep applying the last good plan, restart the warm start
                        fails += 1
                        last_ok = False
                        x_prev = [np.concatenate([x, x]) for _ in range(N + 1)]
                        u_prev = [u_init.copy() for _ in range(N)]

                    if record:
                        r3 = lambda v: round(float(v), 3)
                        frames.append([r3(t), r3(x[0]), r3(x[1]), r3(x[2]), r3(yo[0]), r3(yo[1]), r3(vo[0]), r3(vo[1]),
                                       [[r3(xx[0]), r3(xx[1]), r3(xx[10])] for xx in x_prev],
                                       [[r3(xx[11]), r3(xx[12])] for xx in x_prev]])

                    # ---- apply DT_SUB of the step: LIPM of the OCP (the MPC solves again mid-step) ----
                    h = min(DT_SUB, t_left)
                    c, cd = x[0:2].copy(), x[3:5].copy()
                    th, thd = x[2], x[5]
                    p0, p1 = x[6:8], x[8:10]
                    alpha, beta, gamma = u0[4], u0[5], u0[6]
                    cop = p0 + alpha * (p1 - p0)
                    for tau in np.linspace(0.0, h, 4):
                        ch, sh = math.cosh(w * tau), math.sinh(w * tau)
                        cc = ch * c + sh / w * cd + (1 - ch) * cop
                        yo_t, _ = obs_model.state(t + t_shift + tau)
                        dclear = dist_point_rect(yo_t, cc, th + thd * tau, off_x, off_y) - R_OBS
                        min_clear = min(min_clear, dclear)
                        if dclear < 0.0:
                            collided = True
                    ch, sh = math.cosh(w * h), math.sinh(w * h)
                    F = cfg.robot.m * w**2 * (c - cop)
                    f0 = np.array([beta * F[0], gamma * F[1]])
                    f1 = np.array([(1 - beta) * F[0], (1 - gamma) * F[1]])
                    tau_z = ((p0[0] - c[0]) * f0[1] - (p0[1] - c[1]) * f0[0]
                             + (p1[0] - c[0]) * f1[1] - (p1[1] - c[1]) * f1[0])
                    x_new = x.copy()
                    x_new[0:2] = ch * c + sh / w * cd + (1 - ch) * cop
                    x_new[3:5] = w * sh * c + ch * cd - w * sh * cop
                    x_new[2] = th + thd * h
                    x_new[5] = float(np.clip(thd + h / cfg.robot.Iz * tau_z, -1.5, 1.5))
                    x = x_new
                    t += h
                    t_left -= h
                    if t_left <= 1e-9:
                        # touchdown: swing feet on the footholds of the last solve, new step
                        x[6:10] = u0[0:4]
                        stance = 1 - stance
                        if last_ok:
                            step_T = float(np.clip(u_prev[1][7], cfg.limits.dt_min, cfg.limits.dt_max))
                        t_left = step_T
                        new_step = True
                        if noise > 0:       # tracking error of the WBC
                            x[0:2] += rng.normal(0.0, noise, 2)
                            x[3:5] += rng.normal(0.0, 2 * noise, 2)
                        path.append(x[0:2].copy())
                        obs_path.append(yo.copy())
                        if VERBOSE:
                            print(f"[dbg] t={t:5.2f} st={status} c=({x[0]:+.2f},{x[1]:+.2f}) th={x[2]:+.2f} "
                                  f"v=({x[3]:+.2f},{x[4]:+.2f}) thd={x[5]:+.2f} T={step_T:.3f} a={u0[4]:.2f} "
                                  f"obs=({yo[0]:+.2f},{yo[1]:+.2f}) fails={fails}", flush=True)

                P = np.array(path)
                results.append(dict(
                    variant=variant, scenario=sk, speed=speed, rep=rep,
                    reached=reached, collided=collided, time=t if reached else float("nan"),
                    path_len=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                    min_clear=float(min_clear), fails=fails, steps=len(path) - 1,
                    solve_ms=float(np.mean(solve_times) * 1e3) if solve_times else float("nan"),
                    solve_all_ms=[round(float(v) * 1e3, 3) for v in solve_times],
                    path=P.tolist(), obs_path=np.array(obs_path).tolist(),
                    frames=frames,
                    params=dict(kind=kind, k_safe=k_safe, r=R_OBS, vmax=V_MAX_OBS, amax=A_MAX_OBS,
                                off_x=off_x, off_y=off_y, goal=list(goal), step_nom=step_nom)))
                print(f"[{variant:11s}] {sk} v={speed:.1f} rep={rep}: "
                      f"{'GOAL' if reached else 'TIMEOUT'} t={t:5.1f}s clear={min_clear:+.2f} m "
                      f"{'COLLISION ' if collided else ''}fails={fails}", flush=True)

    with open(out_json, "w") as f:
        json.dump(results, f)


# ---------------- main: one process per variant (one acados library per process) ----------------
def summarize(results, variants, scen_keys):
    def fmt(v, f="{:.2f}"):
        return "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)

    print("\n" + "=" * 96)
    hdr = f"{'scen':5s}{'variant':13s}{'goal':>7s}{'coll':>7s}{'time[s]':>10s}{'path[m]':>9s}" \
          f"{'clear_min':>11s}{'clear_mean':>11s}{'fails':>7s}{'solve[ms]':>11s}"
    print(hdr)
    print("-" * 96)
    for sk in scen_keys + ["ALL"]:
        for v in variants:
            rs = [r for r in results if r["variant"] == v and (sk == "ALL" or r["scenario"] == sk)]
            if not rs:
                continue
            n = len(rs)
            ok = [r for r in rs if r["reached"]]
            tm = np.mean([r["time"] for r in ok]) if ok else float("nan")
            pl = np.mean([r["path_len"] for r in ok]) if ok else float("nan")
            print(f"{sk:5s}{v:13s}{sum(r['reached'] for r in rs):>4d}/{n:<2d}{sum(r['collided'] for r in rs):>4d}/{n:<2d}"
                  f"{fmt(tm):>10s}{fmt(pl):>9s}{fmt(min(r['min_clear'] for r in rs)):>11s}"
                  f"{fmt(np.mean([r['min_clear'] for r in rs])):>11s}{sum(r['fails'] for r in rs):>7d}"
                  f"{fmt(np.mean([r['solve_ms'] for r in rs]), '{:.1f}'):>11s}")
        print("-" * 96)


def plot(results, variants, scen_keys, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(variants), len(scen_keys), figsize=(4.2 * len(scen_keys), 3.4 * len(variants)),
                             squeeze=False)
    cols = {0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd"}
    for i, v in enumerate(variants):
        for j, sk in enumerate(scen_keys):
            ax = axes[i][j]
            speeds = SCENARIOS[sk]["speeds"]
            for r in results:
                if r["variant"] != v or r["scenario"] != sk:
                    continue
                P, O = np.array(r["path"]), np.array(r["obs_path"])
                ci = speeds.index(r["speed"])
                ax.plot(P[:, 0], P[:, 1], color="red" if r["collided"] else cols[ci], lw=1.0, alpha=0.8,
                        label=f"{r['speed']} m/s" if r["rep"] == 0 else None)
                if r["rep"] == 0:
                    ax.plot(O[:, 0], O[:, 1], color="gray", lw=0.8, ls="--")
            ax.plot(*SCENARIOS[sk]["goal"], "k*", ms=10)
            ax.plot(0, 0, "ko", ms=4)
            ax.set_aspect("equal")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-3, 2)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{sk}: {SCENARIOS[sk]['name']}")
            if j == 0:
                ax.set_ylabel(v)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Closed-loop paths (red = collision, dashed = obstacle)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"plot: {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="leonardelli,support,new")
    ap.add_argument("--scenarios", default="A,B,C,D")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.01, help="m, std of the CoM tracking error per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="folder of results.json / paths.png (default: temporary)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--record", action="store_true", help="save every solve (robot, obstacle, plans) for animations")
    ap.add_argument("--speed-index", type=int, default=None, help="only this speed of each scenario (0, 1, 2)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--export-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    scen_keys = args.scenarios.split(",")

    if args.worker:
        worker(args.worker, scen_keys, args.reps, args.noise, args.seed, args.worker_out, args.export_root,
               args.record, args.speed_index)
        return

    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant {v}: {list(VARIANTS)}")
    out = args.out or tempfile.mkdtemp(prefix="closed_loop_")
    os.makedirs(out, exist_ok=True)
    results = []
    procs = []
    # one process per variant, in parallel (acados codegen/build output in codegen_<variant>.log)
    for v in variants:
        wj = os.path.join(out, f"res_{v}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", v, "--worker-out", wj,
               "--export-root", out, "--scenarios", args.scenarios, "--reps", str(args.reps),
               "--noise", str(args.noise), "--seed", str(args.seed)]
        if args.record:
            cmd.append("--record")
        if args.speed_index is not None:
            cmd += ["--speed-index", str(args.speed_index)]
        log = open(os.path.join(out, f"codegen_{v}.log"), "w")
        procs.append((v, wj, log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True)))
    for v, wj, log, proc in procs:
        for line in proc.stdout:
            if line.startswith("["):
                print(line, end="", flush=True)
            else:
                log.write(line)
        proc.wait()
        log.close()
        if proc.returncode != 0:
            sys.exit(f"variant {v} failed, see {log.name}")
        with open(wj) as f:
            results += json.load(f)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f)
    summarize(results, variants, scen_keys)
    if args.plot:
        plot(results, variants, scen_keys, os.path.join(out, "paths.png"))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
