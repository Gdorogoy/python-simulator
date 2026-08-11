# ---------------------------------------------------------------------------
# Helper method helping to display the drone state in the last 10 eps
# ---------------------------------------------------------------------------
def calc_drone_state(drone_state_arr, n=10):
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
