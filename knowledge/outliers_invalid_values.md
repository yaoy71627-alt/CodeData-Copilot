# Outliers, Invalid Values and Type Consistency

- IQR 与 MAD 标记的是统计意义上的潜在极端值；极端值可能是真实样本，不能自动当作非法值删除。
- +inf 与 -inf 是确定性的非有限数值，应在统计和建模前追踪来源并显式处理。
- NaN/None 是真实缺失；NA、NULL、unknown 等字符串只能先标记为疑似缺失 token，再由业务语义确认。
- 没有显式业务约束时，不应仅凭字段名断言年龄、分数或温度的合法范围。
- 只有用户提供 min/max 或枚举约束后，越界值才能被确定为 Constraint Violation。
- 大部分值可解析而少量失败时，应保留失败样本并报告 Numeric/Datetime Parsing Anomaly。
- 混合类型、空白字符串、首尾空格与类别大小写差异应先诊断，再采用可追踪的标准化规则。
- 样本量过小时不要强行执行异常值判断；IQR 为零时可使用 MAD，但不应重复输出两套相同问题。

