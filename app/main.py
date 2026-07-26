from app.demo.demo import simulation
from app.test.test_config import test


def main():
    print(f"Running Engine(Physics,RL)")
    inp=0

    while inp not in (1, 2, 3):
        print(f"==========================\n"
              f"Enter 1 to start Demo \n"
              f"Enter 2 to start Tests \n"
              f"Enter 3 to start RL \n"
              f"==========================\n")
        inp = int(input("Please enter your choice: "))

    if inp == 1:
        simulation()
    elif inp == 2:
        test()
    elif inp == 3:
        print("RL")


if __name__ == "__main__":
    main()