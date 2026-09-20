import pandas as pd


def analyze_csv(csv_path):

    df = pd.read_csv(csv_path)

    result = {}

    # 基本信息
    result["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


    # 缺失率
    missing = df.isnull().mean()

    result["missing_values"] = (
        missing[missing > 0]
        .to_dict()
    )


    # 重复值
    result["duplicate_rows"] = int(
        df.duplicated().sum()
    )


    # 数据类型
    result["data_types"] = (
        df.dtypes.astype(str)
        .to_dict()
    )


    return result



def print_report(csv_path):

    report = analyze_csv(csv_path)


    print("="*40)
    print("数据质量报告")
    print("="*40)


    print("\n数据规模:")
    print(report["shape"])


    print("\n缺失值:")
    if report["missing_values"]:
        for k,v in report["missing_values"].items():
            print(
                k,
                ":",
                round(v*100,2),
                "%"
            )
    else:
        print("无")


    print("\n重复值:")
    print(report["duplicate_rows"])


    print("\n字段类型:")
    for k,v in report["data_types"].items():
        print(k,":",v)



if __name__ == "__main__":

    csv_path = "demo_data/train.csv"

    print_report(csv_path)