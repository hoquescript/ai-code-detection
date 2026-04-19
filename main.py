import pandas as pd
from model.embedding import kfold_embedding_model


def main():
    df = pd.read_csv("data/sample/python.csv")
    report = kfold_embedding_model(df, ["code", "ast", "combined"])
    print(report)


if __name__ == "__main__":
    main()
