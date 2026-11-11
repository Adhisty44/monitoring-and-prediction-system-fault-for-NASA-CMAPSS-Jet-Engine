import argparse

from training.train_models import TRAINERS


def main():
    parser = argparse.ArgumentParser(description="Latih model status mesin.")
    parser.add_argument("--model", default="all", choices=[*TRAINERS, "all"])
    args = parser.parse_args()

    names = list(TRAINERS) if args.model == "all" else [args.model]
    for name in names:
        TRAINERS[name]().run()      # polymorphism: semua dipanggil sama


if __name__ == "__main__":
    main()