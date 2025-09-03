"""
路径管理模块
统一管理项目中所有数据文件的路径生成和组织
"""

import os


def get_data_path(data_type, filename=None, auto_create=True, **kwargs):
    """
    统一的数据路径管理函数

    参数:
        data_type: 数据类型
            - 'combo_mask': 组合掩码数据 -> data/cache/combo_masks/
            - 'return_1d': 日收益率数据 -> data/cache/returns/
            - 'industry_market': 行业市值数据 -> data/cache/industry/
            - 'open_price': 开盘价数据 -> data/cache/open_price/
            - 'factor_raw': 原始因子数据 -> data/factor_lib/raw/
            - 'factor_processed': 处理后因子数据 -> data/factor_lib/processed/
            - 'ic_report': IC分析报告 -> outputs/reports/IC_analysis/
            - 'layered_report': 分层分析报告 -> outputs/reports/layered_analysis/
            - 'metrics_table': 绩效指标表 -> outputs/reports/metrics_tables/
            - 'performance_chart': 绩效曲线图 -> outputs/reports/performance_charts/
            - 'account_result': 回测结果 -> data/account_result/
        filename: 文件名（可选，如果不提供则根据kwargs自动生成）
        auto_create: 是否自动创建目录
        **kwargs: 用于生成文件名的参数

    返回:
        完整的文件路径
    """

    # 路径映射
    path_mapping = {
        "combo_mask": "data/cache/combo_masks",
        "return_1d": "data/cache/returns",
        "industry_market": "data/cache/industry",
        "open_price": "data/cache/open_price",
        "factor_raw": "data/factor_lib/raw",
        "factor_processed": "data/factor_lib/processed",
        "limit_up": "data/cache/limit_up",
        "stock_universe": "data/cache/stock_universe",
        "ic_report": "outputs/reports/IC_analysis",
        "layered_report": "outputs/reports/layered_analysis",
        "metrics_table": "outputs/reports/metrics_tables",
        "performance_chart": "outputs/reports/performance_charts",
        "strategy_comparison": "outputs/reports/strategy_comparison",
        "hot_corr": "outputs/reports/hot_corr",
        "account_result": "data/account_result",
        "IC_df": "data/IC_df",
    }

    # 文件名模板
    filename_templates = {
        "combo_mask": "combo_mask_{index_item}_{start_date}_{end_date}.pkl",
        "return_1d": "return_1d_{index_item}_{start}_{end}.pkl",
        "industry_market": "df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl",
        "open_price": "open_{index_item}_{start}_{end}.pkl",
        "limit_up": "limit_up_{index_item}_{start_date}_{end_date}.pkl",
        "stock_universe": "stock_universe_{index_item}_{start}_{end}.pkl",
        "ic_report": "{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}_IC.png",
        "layered_report": "{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}_{rebalance_days}_layered.png",
        "metrics_table": "{factor_name}_{benchmark_index}_{direction}_{neutralize}_{start_date}_{end_date}_metrics_table.png",
        "performance_chart": "{factor_name}_{benchmark_index}_{direction}_{neutralize}_{start_date}_{end_date}_performance_chart.png",
        "strategy_comparison": "{benchmark_name}_{benchmark_neutralize}_vs_{strategy_name}_{strategy_neutralize}_{index_item}_{start_date}_{end_date}_comparison.png",
        "hot_corr": "factors_IC_correlation_{index_item}_{neutralize}_{start_date}_{end_date}.png",
        "account_result": "{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}_account_result.pkl",
        "IC_df": "{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}_IC_values.pkl",
    }

    if data_type not in path_mapping:
        raise ValueError(
            f"不支持的数据类型: {data_type}。支持的类型: {list(path_mapping.keys())}"
        )

    # 生成文件名
    if filename is None:
        if data_type in filename_templates:
            filename = filename_templates[data_type].format(**kwargs)
        else:
            raise ValueError(f"数据类型 {data_type} 需要提供 filename 参数")

    # 使用固定的绝对路径
    alpha_local_path = "/Users/didi/KDCJ/alpha_local"

    # 对于报告类型，根据index_item和日期创建子文件夹
    report_types = [
        "ic_report",
        "layered_report",
        "metrics_table",
        "performance_chart",
        "strategy_comparison",
        "hot_corr",
    ]
    # 对于因子数据类型，根据index_item创建子文件夹
    factor_types = ["factor_raw", "factor_processed", "IC_df"]
    # 对于回测结果，按指数、日期和中性化状态分类
    account_result_types = ["account_result"]

    if data_type in report_types and "index_item" in kwargs:
        index_folder = kwargs["index_item"]

        # 获取因子名称，如果没有提供则使用默认值
        factor_name = kwargs.get("factor_name", "unknown_factor")

        # 添加日期文件夹（格式：YYYYMMDD）
        from datetime import datetime

        date_folder = datetime.now().strftime("%Y%m%d")

        # 简化路径结构，移除neutralize文件夹（文件名中已包含中性化信息）
        full_path = os.path.join(
            alpha_local_path,
            path_mapping[data_type],
            index_folder,
            factor_name,
            date_folder,
            filename,
        )
    elif data_type in account_result_types:
        # 回测结果按指数、日期和中性化状态分类存放
        from datetime import datetime

        if "index_item" not in kwargs:
            raise ValueError(f"数据类型 {data_type} 需要提供 index_item 参数")

        index_folder = kwargs["index_item"]
        date_folder = datetime.now().strftime("%Y%m%d")

        # 如果提供了neutralize参数，则进一步按中性化状态分类
        if "neutralize" in kwargs:
            neutralize_folder = "True" if kwargs["neutralize"] else "False"
            full_path = os.path.join(
                alpha_local_path,
                path_mapping[data_type],
                index_folder,
                date_folder,
                neutralize_folder,
                filename,
            )
        else:
            full_path = os.path.join(
                alpha_local_path,
                path_mapping[data_type],
                index_folder,
                date_folder,
                filename,
            )
    elif data_type in factor_types and "index_item" in kwargs:
        # 因子数据按指数分类存放，不需要日期文件夹
        index_folder = kwargs["index_item"]

        # 对于processed因子和IC_df，进一步按neutralize分类
        if data_type in ["factor_processed", "IC_df"] and "neutralize" in kwargs:
            neutralize_folder = "True" if kwargs["neutralize"] else "False"
            full_path = os.path.join(
                alpha_local_path,
                path_mapping[data_type],
                index_folder,
                neutralize_folder,
                filename,
            )
        else:
            full_path = os.path.join(
                alpha_local_path,
                path_mapping[data_type],
                index_folder,
                filename,
            )
    else:
        full_path = os.path.join(alpha_local_path, path_mapping[data_type], filename)

    # 自动创建目录
    if auto_create:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

    return full_path


def load_processed_factors(factor_names, neutralize, index_item, start_date, end_date):
    """
    从 processed 文件夹加载处理后的因子，支持单个或多个因子

    :param factor_names: 因子名称或因子名称列表
    :param neutralize: 是否中性化
    :param index_item: 指数代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 单个因子返回DataFrame，多个因子返回字典
    """
    import pandas as pd
    from alpha_local.core.factors import get_factor_config

    # 统一处理为列表格式
    if isinstance(factor_names, str):
        factor_names = [factor_names]
        return_single = True
    else:
        return_single = False

    factors_dict = {}

    for factor_name in factor_names:
        try:
            # 获取因子配置信息
            factor_info = get_factor_config(factor_name, neutralize=neutralize)
            direction = factor_info["direction"]

            # 构建文件名
            filename = f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl"

            # 使用统一路径管理生成文件路径
            file_path = get_data_path(
                "factor_processed",
                factor_name=factor_name,
                index_item=index_item,
                direction=direction,
                neutralize=neutralize,
                start_date=start_date,
                end_date=end_date,
                filename=filename,
            )

            # 加载因子数据
            factor_df = pd.read_pickle(file_path)
            factors_dict[factor_name] = factor_df
            print(f"✅加载因子: {factor_name} (中性化: {neutralize})")

        except FileNotFoundError:
            print(f"❌未找到因子文件: {factor_name}")
        except Exception as e:
            print(f"❌加载因子 {factor_name} 失败: {e}")

    # 根据输入类型返回结果
    if return_single:
        if len(factors_dict) == 1:
            return list(factors_dict.values())[0]
        else:
            return None
    else:
        print(f"\n📊成功加载 {len(factors_dict)} 个因子")
        return factors_dict
