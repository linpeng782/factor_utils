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
        "account_result": "data/account_result",
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
        "account_result": "{start_date}_{end_date}_{factor_name}_account_result.pkl",
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
    report_types = ["ic_report", "layered_report", "metrics_table", "performance_chart"]
    # 对于因子数据类型，根据index_item创建子文件夹
    factor_types = ["factor_raw", "factor_processed"]
    # 对于回测结果，添加日期文件夹
    date_classified_types = ["account_result"]
    
    if data_type in report_types and "index_item" in kwargs:
        index_folder = kwargs["index_item"]
        # 添加日期文件夹（格式：YYYYMMDD）
        from datetime import datetime

        date_folder = datetime.now().strftime("%Y%m%d")
        full_path = os.path.join(
            alpha_local_path,
            path_mapping[data_type],
            index_folder,
            date_folder,
            filename,
        )
    elif data_type in date_classified_types:
        # 回测结果按日期分类存放
        from datetime import datetime
        
        date_folder = datetime.now().strftime("%Y%m%d")
        full_path = os.path.join(
            alpha_local_path,
            path_mapping[data_type],
            date_folder,
            filename,
        )
    elif data_type in factor_types and "index_item" in kwargs:
        # 因子数据按指数分类存放，不需要日期文件夹
        index_folder = kwargs["index_item"]
        
        # 对于processed因子，进一步按neutralize分类
        if data_type == "factor_processed" and "neutralize" in kwargs:
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
