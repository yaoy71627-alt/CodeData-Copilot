import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ======================
# 加载环境变量
# ======================

load_dotenv()


api_key = os.getenv("DASHSCOPE_API_KEY")


if not api_key:
    raise ValueError(
        "未检测到DASHSCOPE_API_KEY，请配置.env文件"
    )


# ======================
# 千问客户端
# ======================

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)



# ======================
# Agent核心
# ======================

def run_agent(code_result, data_result):
    """
    数据科学项目Review Agent

    输入:
        code_result:
            代码分析结果

        data_result:
            数据质量分析结果

    输出:
        LLM生成的诊断报告
    """


    prompt = f"""

你是一名资深数据科学家和机器学习工程师。

你的任务是审查一个机器学习项目。

下面提供两个自动分析模块的结果：

=====================
代码流程分析结果：

{json.dumps(
    code_result,
    ensure_ascii=False,
    indent=2
)}

=====================
数据质量分析结果：

{json.dumps(
    data_result,
    ensure_ascii=False,
    indent=2
)}


请从数据科学项目规范角度生成Review报告。

重点分析：

1. 是否存在数据泄露风险；
2. 数据预处理流程是否合理；
3. 数据质量问题；
4. 可能影响模型训练和评估的问题；
5. 给出具体修改建议；
6. 给出推荐的数据科学流程。


输出格式：

# 项目诊断报告

## 发现的问题

## 原因分析

## 潜在影响

## 修改建议

## 推荐流程

"""


    response = client.chat.completions.create(

        model="qwen-plus",

        messages=[

            {
                "role":"system",
                "content":
                "你是一个机器学习项目审查Agent。"
            },

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.3

    )


    return response.choices[0].message.content



# ======================
# 本地测试
# ======================

if __name__ == "__main__":


    test_code_result = [

        {
            "cell":3,

            "issues":[

                {
                    "type":
                    "Data Leakage Risk",

                    "message":
                    "检测到fit_transform操作，可能在数据划分前使用全部数据拟合"
                }

            ]
        }

    ]


    test_data_result = {

        "shape":
        {
            "rows":1000,
            "columns":20
        },


        "missing_values":
        {
            "Age":0.2
        },


        "duplicate_rows":5

    }


    report = run_agent(
        test_code_result,
        test_data_result
    )


    print(report)