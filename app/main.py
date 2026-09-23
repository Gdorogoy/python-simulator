from app.test.test_config import test

# demo.py (PyBullet GUI demo) moved to deprecated/demo/demo.py -- Isaac Sim is
# the only visualization/simulation path now, see scripts/isaac_lab/*.py
# (e.g. random_agent_smoke.py, zero_agent.py) for the equivalent live view.


def main():
    print(f"Running Engine(Physics,RL)")
    inp=0

    while inp not in (1, 2, 3):
        print(f"==========================\n"
              f"Enter 1 to start Demo (removed -- see scripts/isaac_lab/*.py) \n"
              f"Enter 2 to start Tests \n"
              f"Enter 3 to start RL \n"
              f"==========================\n")
        inp = int(input("Please enter your choice: "))

    if inp == 1:
        print("Demo removed -- PyBullet visualization retired, use scripts/isaac_lab/random_agent_smoke.py "
              "or zero_agent.py (run with the Isaac py3.11 venv) instead.")
    elif inp == 2:
        test()
    elif inp == 3:
        print("RL")


if __name__ == "__main__":
    main()