import nbformat
import ast


def read_notebook(notebook_path):
    """
    读取Notebook，提取代码cell
    """

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    code_cells = []

    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells.append(cell.source)

    return code_cells


def analyze_code_with_ast(code):
    """
    使用AST分析代码
    """

    issues = []

    try:
        tree = ast.parse(code)

    except SyntaxError:
        issues.append({
            "type": "Syntax Error",
            "message": "代码存在语法错误"
        })

        return issues


    has_fit_transform = False
    has_train_test_split = False
    has_dropna=False


    for node in ast.walk(tree):

        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Attribute):

                func_name = node.func.attr


                # 规则1:
                # 检测fit_transform

                if func_name == "fit_transform":

                    has_fit_transform = True


                # 规则2:
                # 检测train_test_split

                if func_name == "train_test_split":

                    has_train_test_split = True
                # 规则3:
                # 检测dropna

                if func_name == "dropna":

                    has_dropna = True



    # 如果发现标准化在数据划分之前
    if has_fit_transform and not has_train_test_split:

        issues.append({

            "type":
            "Data Leakage Risk",

            "message":
            "检测到fit_transform操作，可能在训练测试集划分前使用全部数据进行拟合"

        })


    elif has_fit_transform and has_train_test_split:
       

        issues.append({

            "type":
            "Potential Data Leakage",

            "message":
            "检测到数据标准化和数据划分操作，请检查执行顺序"

        })

    if has_dropna:

        issues.append({

            "type":
            "Missing Value Handling Risk",

             "message":
            "检测到dropna操作，直接删除缺失值可能导致样本损失，建议评估缺失比例和处理策略"

        })
    return issues


def analyze_notebook(notebook_path):

    code_cells = read_notebook(notebook_path)

    results = []


    for index, code in enumerate(code_cells):

        cell_result = analyze_code_with_ast(code)


        if cell_result:

            results.append({
                "cell": index + 1,
                "issues": cell_result
            })


    return results



def print_result(notebook_path):

    results = analyze_notebook(notebook_path)


    print("=" * 40)

    if not results:

        print("未发现明显问题")

    else:

        print("发现问题：")

        for result in results:

            print(f"\nCell {result['cell']}")

            for issue in result["issues"]:

                print(
                    "类型:",
                    issue["type"]
                )

                print(
                    "说明:",
                    issue["message"]
                )

    print("=" * 40)



if __name__ == "__main__":

    notebook_path = "demo_data/test.ipynb"

    print_result(notebook_path)