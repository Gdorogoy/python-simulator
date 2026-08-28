import math


def compute_grade(success_rate, avg_final_dist, avg_hit_time_sec, avg_grad_norm,
                   oob_radius, episode_time_budget_sec, grad_norm_ceiling=50.0,
                   w_success=1.0, w_error=0.3, w_time=0.2, w_grad=0.1):
    """Single scalar score for comparing checkpoints/trials: success rate, rewarded,
    minus normalized penalties for final distance error, time-to-hit, and grad norm.
    Returns (grade, breakdown_dict)."""
    error_ratio = min(max(avg_final_dist / oob_radius, 0.0), 1.0)
    if avg_hit_time_sec is None:
        time_ratio = 1.0
    else:
        time_ratio = min(max(avg_hit_time_sec / episode_time_budget_sec, 0.0), 1.0)
    grad_ratio = min(math.log1p(max(avg_grad_norm, 0.0)) / math.log1p(grad_norm_ceiling), 1.0)

    grade = (w_success * success_rate) - (w_error * error_ratio) - (w_time * time_ratio) - (w_grad * grad_ratio)
    breakdown = {
        "success_rate": success_rate,
        "error_ratio": error_ratio,
        "time_ratio": time_ratio,
        "grad_ratio": grad_ratio,
        "grade": grade,
    }
    return grade, breakdown


def calc_drone_state(drone_state_arr, n=10):
    """Averages position/velocity/orientation/rotor_rpm over the last n drone states."""
    if len(drone_state_arr) < 0:
        return


    elif len(drone_state_arr) < n:

        recent = drone_state_arr[-len(drone_state_arr) - 1:]
        count = len(recent)

        avg_pos = [
            sum(s.position.x for s in recent) / count,
            sum(s.position.y for s in recent) / count,
            sum(s.position.z for s in recent) / count,
        ]
        avg_vel = [
            sum(s.velocity.x for s in recent) / count,
            sum(s.velocity.y for s in recent) / count,
            sum(s.velocity.z for s in recent) / count,
        ]
        avg_orient = [
            sum(s.orientation.x for s in recent) / count,
            sum(s.orientation.y for s in recent) / count,
            sum(s.orientation.z for s in recent) / count,
            sum(s.orientation.w for s in recent) / count,
        ]
        avg_rotor_rpm = [
            sum(s.rotor_rpm[i] for s in recent) / count
            for i in range(len(recent[0].rotor_rpm))
        ]

        return {
            "position": avg_pos,
            "velocity": avg_vel,
            "orientation": avg_orient,
            "rotor_rpm": avg_rotor_rpm,
            "n_averaged": count,
        }

    else:

        recent = drone_state_arr[-n:]
        count = len(recent)

        avg_pos = [
            sum(s.position.x for s in recent) / count,
            sum(s.position.y for s in recent) / count,
            sum(s.position.z for s in recent) / count,
        ]
        avg_vel = [
            sum(s.velocity.x for s in recent) / count,
            sum(s.velocity.y for s in recent) / count,
            sum(s.velocity.z for s in recent) / count,
        ]
        avg_orient = [
            sum(s.orientation.x for s in recent) / count,
            sum(s.orientation.y for s in recent) / count,
            sum(s.orientation.z for s in recent) / count,
            sum(s.orientation.w for s in recent) / count,
        ]
        avg_rotor_rpm = [
            sum(s.rotor_rpm[i] for s in recent) / count
            for i in range(len(recent[0].rotor_rpm))
        ]

        return {
            "position": avg_pos,
            "velocity": avg_vel,
            "orientation": avg_orient,
            "rotor_rpm": avg_rotor_rpm,
            "n_averaged": count,
        }
