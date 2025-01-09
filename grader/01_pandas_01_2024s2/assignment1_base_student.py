import pandas as pd


def main():
    file = input()
    func = input()

    df = pd.read_csv(file)

    if func == "Q1":
        # Do something
        print(df.shape)
    elif func == "Q2":
        # Do something
        print(df.max()["score"])
    elif func == "Q3":
        # Do something
        print(df[df["score"] >= 80].shape[0])
    else:
        # Do something
        print("No Output")


if __name__ == "__main__":
    main()
