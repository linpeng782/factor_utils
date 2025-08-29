import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import os
from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *

init("13522652015", "123456")
import rqdatac

from tqdm import *

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import warnings
from factor_utils.path_manager import get_data_path

warnings.filterwarnings("ignore")


# 热力图
def hot_corr(name, ic_df):
    """
    :param name: 因子名称 -> list
    :param ic_df: ic序列表 -> dataframe
    :return fig: 热力图 -> plt
    """
    ax = plt.subplots(figsize=(len(name), len(name)))  # 调整画布大小
    ax = sns.heatmap(
        ic_df[name].corr(), vmin=0.4, square=True, annot=True, cmap="Greens"
    )  # annot=True 表示显示系数
    plt.title("Factors_IC_CORRELATION")
    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


# 动态券池
def INDEX_FIX(start_date, end_date, index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str
    :param index_item: 指数代码 -> str
    :return index_fix: 动态因子值 -> unstack
    """

    index_fix = pd.DataFrame(
        {
            k: dict.fromkeys(v, True)
            for k, v in index_components(
                index_item, start_date=start_date, end_date=end_date
            ).items()
        }
    ).T

    index_fix.fillna(False, inplace=True)
    index_fix.index.names = ["datetime"]

    return index_fix


# 新股过滤
def get_new_stock_filter(stock_list, datetime_period, newly_listed_threshold=252):
    """
    :param stock_list: 股票队列 -> list
    :param datetime_period: 研究周期 -> list
    :param newly_listed_threshold: 新股日期阈值 -> int
    :return newly_listed_window: 新股过滤券池 -> unstack
    """

    datetime_period_tmp = datetime_period.copy()
    # 多添加一天
    datetime_period_tmp += [
        pd.to_datetime(get_next_trading_date(datetime_period[-1], 1))
    ]
    # 获取上市日期
    listed_datetime_period = [instruments(stock).listed_date for stock in stock_list]
    # 获取上市后的第252个交易日（新股和老股的分界点）
    newly_listed_window = pd.Series(
        index=stock_list,
        data=[
            pd.to_datetime(get_next_trading_date(listed_date, n=newly_listed_threshold))
            for listed_date in listed_datetime_period
        ],
    )
    # 防止分割日在研究日之后，后续填充不存在
    for k, v in enumerate(newly_listed_window):
        if v > datetime_period_tmp[-1]:
            newly_listed_window.iloc[k] = datetime_period_tmp[-1]

    # 标签新股，构建过滤表格
    newly_listed_window.index.names = ["order_book_id"]
    newly_listed_window = newly_listed_window.to_frame("date")
    newly_listed_window["signal"] = True
    newly_listed_window = (
        newly_listed_window.reset_index()
        .set_index(["date", "order_book_id"])
        .signal.unstack("order_book_id")
        .reindex(index=datetime_period_tmp)
    )
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False).iloc[:-1]

    return newly_listed_window


# st过滤（风险警示标的默认不进行研究）
def get_st_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return st_filter: st过滤券池 -> unstack
    """

    # 当st时返回1，非st时返回0
    st_filter = is_st_stock(stock_list, date_list[0], date_list[-1]).reindex(
        columns=stock_list, index=date_list
    )
    st_filter = st_filter.shift(-1).ffill()

    return st_filter


# 停牌过滤 （无法交易）
def get_suspended_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return suspended_filter: 停牌过滤券池 -> unstack
    """

    # 当停牌时返回1，非停牌时返回0
    suspended_filter = is_suspended(stock_list, date_list[0], date_list[-1]).reindex(
        columns=stock_list, index=date_list
    )
    suspended_filter = suspended_filter.shift(-1).ffill()

    return suspended_filter


# 涨停过滤 （开盘无法买入）
def get_limit_up_filter(stock_list, date_list, index_item):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return limit_up_filter: 涨停过滤券池 -> unstack∏
    """

    new_path = get_data_path(
        "limit_up",
        index_item=index_item,
        start_date=date_list[0].strftime("%F"),
        end_date=date_list[-1].strftime("%F"),
    )

    try:
        limit_up_mask = pd.read_pickle(new_path)
        print(f"✅过滤涨停：加载limit_up_mask {new_path}")
    except:
        print(f"✅过滤涨停：计算新的limit_up_mask...")

        price = get_price(
            stock_list,
            date_list[0],
            date_list[-1],
            adjust_type="none",
            fields=["open", "limit_up"],
        )
        limit_up_mask = (
            (price["open"] == price["limit_up"])
            .unstack("order_book_id")
            .shift(-1)
            .fillna(False)
        )

        # 保存limit_up_mask
        limit_up_mask.to_pickle(new_path)
        print(f"✅过滤涨停：保存limit_up_mask {new_path}")

    return limit_up_mask


# 离群值处理
def mad_vectorized(df, n=3 * 1.4826):

    # 计算每行的中位数
    median = df.median(axis=1)
    # 计算每行的MAD
    mad_values = (df.sub(median, axis=0).abs()).median(axis=1)
    # 计算上下界
    lower_bound = median - n * mad_values
    upper_bound = median + n * mad_values

    return df.clip(lower_bound, upper_bound, axis=0)


# 标准化处理
def standardize(df):
    df_standardize = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_standardize


# 获取行业暴露度矩阵
def get_industry_exposure(order_book_ids, datetime_period, industry_type="zx"):
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :param industry_type: 行业分类标准 二选一 中信/申万 zx/sw -> str
    :return result: 虚拟变量 -> df_unstack
    """

    # 获取行业特征数据
    if industry_type == "zx":
        industry_map_dict = rqdatac.client.get_client().execute(
            "__internal__zx2019_industry"
        )
        # 构建个股/行业map
        df = pd.DataFrame(
            industry_map_dict,
            columns=["first_industry_name", "order_book_id", "start_date"],
        )
        df.sort_values(["order_book_id", "start_date"], ascending=True, inplace=True)
        df = df.pivot(
            index="start_date", columns="order_book_id", values="first_industry_name"
        ).ffill()
    else:
        industry_map_dict = rqdatac.client.get_client().execute(
            "__internal__shenwan_industry"
        )
        df = pd.DataFrame(
            industry_map_dict,
            columns=["index_name", "order_book_id", "version", "start_date"],
        )
        df = df[df.version == 2]
        df = df.drop_duplicates()
        df = (
            df.set_index(["start_date", "order_book_id"])
            .index_name.unstack("order_book_id")
            .ffill()
        )

    # 匹配交易日
    datetime_period_base = pd.to_datetime(
        get_trading_dates(get_previous_trading_date(df.index[0], 2), df.index[-1])
    )
    df.index = datetime_period_base.take(
        datetime_period_base.searchsorted(df.index, side="right") - 1
    )
    # 切片所需日期
    df = df.reset_index().drop_duplicates(subset=["index"]).set_index("index")
    df = (
        df.reindex(index=datetime_period_base)
        .ffill()
        .reindex(index=datetime_period)
        .ffill()
    )
    inter_stock_list = list(set(df.columns) & set(order_book_ids))
    df = df[inter_stock_list].sort_index(axis=1)

    # 生成行业虚拟变量
    return df.stack()


# 行业市值中性化
def neutralization_vectorized(
    factor,
    order_book_ids,
    index_item="",
    industry_type="zx",
):

    datetime_period = factor.index.tolist()
    start = datetime_period[0].strftime("%F")
    end = datetime_period[-1].strftime("%F")

    # 尝试从新路径读取industry_market数据
    new_path = get_data_path(
        "industry_market",
        industry_type=industry_type,
        index_item=index_item,
        start=start,
        end=end,
    )

    df_industry_market = None

    # 尝试加载数据
    try:
        df_industry_market = pd.read_pickle(new_path)
        print(f"✅行业市值中性化:加载industry_market: {new_path}")
    except:
        print(f"✅行业市值中性化:计算新的industry_market...")

    # 如果没有找到，则重新计算
    if df_industry_market is None:
        # 获取市值暴露度
        market_cap = (
            execute_factor(LOG(Factor("market_cap_3")), order_book_ids, start, end)
            .stack()
            .to_frame("market_cap")
        )
        # 获取行业暴露度
        industry_df = pd.get_dummies(
            get_industry_exposure(order_book_ids, datetime_period, industry_type)
        ).astype(int)
        # 合并市值行业暴露度
        industry_df["market_cap"] = market_cap
        df_industry_market = industry_df
        df_industry_market.index.names = ["datetime", "order_book_id"]
        df_industry_market.dropna(axis=0, inplace=True)

        # 保存到新路径
        df_industry_market.to_pickle(new_path)
        print(f"✅行业市值中性化:保存industry_market: {new_path}")

    df_industry_market["factor"] = factor.stack()
    df_industry_market.dropna(subset="factor", inplace=True)

    # 将factor列移到第一列
    cols = df_industry_market.columns.tolist()
    cols.remove("factor")
    df_industry_market = df_industry_market[["factor"] + cols]

    # 截面回归
    def neutralize_cross_section(group):

        try:
            y = group["factor"]
            x = group.iloc[:, 1:]
            model = sm.OLS(y, x, hasconst=False, missing="drop").fit()
            return pd.Series(model.resid)
        except:
            return pd.Series(dtype=float)

    factor_resid = df_industry_market.groupby(level=0).apply(neutralize_cross_section)
    factor_resid = factor_resid.reset_index(level=0, drop=True)
    factor_resid = factor_resid.unstack(level=0).T
    factor_resid.index = pd.to_datetime(factor_resid.index)

    return factor_resid


# 单因子检验
def calc_ic(
    df,
    index_item,
    direction,
    neutralize,
    rebalance_days,
    factor_name="",
    Rank_IC=True,
):
    """
    计算因子IC
    :param df: 因子数据 DataFrame
    :param rebalance_days: 换手周期（天数），可以是单个数字或列表
    :param index_item: 基准指数
    :param factor_name: 因子名称
    :param Rank_IC: 是否使用排名IC
    :return: IC结果和报告
    """
    # 基础数据获取
    order_book_ids = df.columns.tolist()
    date_list = df.index
    start = date_list.min().strftime("%F")
    end = date_list.max().strftime("%F")

    # 尝试从路径读取open_price数据
    new_path = get_data_path("open_price", index_item=index_item, start=start, end=end)
    open = None

    # 尝试加载数据
    try:
        open = pd.read_pickle(new_path)
        print(f"✅加载open_price: {new_path}")
    except:
        print(f"✅计算新的open_price...")

    # 如果没有找到，则重新计算
    if open is None:
        index_fix = INDEX_FIX(start, end, index_item)
        order_book_ids = index_fix.columns.tolist()
        # 获取开盘价
        open = get_price(
            order_book_ids,
            start_date=start,
            end_date=end,
            frequency="1d",
            fields="open",
        ).open.unstack("order_book_id")
        # 保存到新路径
        open.to_pickle(new_path)
        print(f"保存open_price: {new_path}")

    # 未来一段收益股票的累计收益率计算
    future_returns = open.pct_change(rebalance_days).shift(-rebalance_days - 1)

    # 计算IC
    if Rank_IC:
        ic_values = df.corrwith(future_returns, axis=1, method="spearman").dropna(
            how="all"
        )
    else:
        ic_values = df.corrwith(future_returns, axis=1, method="pearson").dropna(
            how="all"
        )

    # t检验 单样本
    t_stat, _ = stats.ttest_1samp(ic_values, 0)

    # 因子报告
    ic_report = {
        "factor_name": factor_name,
        "direction": direction,
        "neutralized": neutralize,
        "rebalance_days": rebalance_days,
        "IC_mean": round(ic_values.mean(), 4),
        "IC_std": round(ic_values.std(), 4),
        "ICIR": round(ic_values.mean() / ic_values.std(), 4),
        "IC_>0": round(len(ic_values[ic_values > 0].dropna()) / len(ic_values), 4),
        "ABS_IC_>2%": round(
            len(ic_values[abs(ic_values) > 0.02].dropna()) / len(ic_values), 4
        ),
        "t_statistic": round(t_stat, 4),
    }

    # 转换为DataFrame格式
    ic_report_df = pd.DataFrame([ic_report]).set_index("factor_name")

    # 生成PNG表格图片
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 设置中文字体
    rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("tight")
    ax.axis("off")

    # 准备表格数据，包含行索引
    # 重置索引，将factor_name作为第一列
    display_df = ic_report_df.reset_index()
    table_data = display_df.values
    col_labels = display_df.columns

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # 设置数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            else:
                table[(i, j)].set_facecolor("white")

    # 设置标题
    plt.title(f"{factor_name} IC分析报告", fontsize=16, fontweight="bold", pad=20)

    # 保存合并报告到指定目录
    import os
    from datetime import datetime

    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

    # 使用get_data_path生成路径，根据index_item分类存放
    png_save_path = get_data_path(
        "ic_report",
        factor_name=factor_name,
        index_item=index_item,
        direction=direction,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
    )

    # 保存PNG图片
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"IC表格图片已保存到: {png_save_path}")
    print(f"\nIC报告:")
    print(ic_report_df)

    return ic_values, ic_report_df


# 数据清洗封装函数：券池清洗、离群值处理、标准化处理、中性化处理、涨停过滤
def preprocess_raw_factor(
    factor_name,
    raw_factor,
    index_item,
    direction,
    neutralize,
    stock_universe,
):

    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()
    universe_start = date_list[0].strftime("%F")
    universe_end = date_list[-1].strftime("%F")

    # 尝试从路径读取combo_mask
    new_path = get_data_path(
        "combo_mask",
        index_item=index_item,
        start_date=universe_start,
        end_date=universe_end,
    )

    # 尝试加载数据
    try:
        combo_mask = pd.read_pickle(new_path)
        print(f"✅过滤新股、ST、停牌：加载combo_mask {new_path}")
    except:
        print(f"✅过滤新股、ST、停牌：计算新的combo_mask...")

        #  新股过滤
        new_stock_filter = get_new_stock_filter(stock_list, date_list)
        # st过滤
        st_filter = get_st_filter(stock_list, date_list)
        # 停牌过滤
        suspended_filter = get_suspended_filter(stock_list, date_list)

        combo_mask = (
            new_stock_filter.astype(int)
            + st_filter.astype(int)
            + suspended_filter.astype(int)
            + (~stock_universe).astype(int)
        ) == 0

        # 保存到新路径
        combo_mask.to_pickle(new_path)
        print(f"✅保存combo_mask: {new_path}")

    # axis=1,过滤掉所有日期截面都是nan的股票
    # factor = factor.mask(~combo_mask).dropna(axis=1, how="all")
    factor = raw_factor.mask(~combo_mask)
    # factor = factor.dropna(axis=1, how="all")
    # print("删除全列nan后的因子 shape:", factor.shape)

    if neutralize:
        # 离群值处理
        factor = mad_vectorized(factor)

        # 标准化处理
        factor = standardize(factor)

        # 中性化处理
        factor = neutralization_vectorized(factor, stock_list)
        print("✅中性化后的因子 shape:", factor.shape)
    else:
        print(f"✅{factor_name}_{index_item}_{direction}未进行中性化处理")

    # 涨停过滤
    limit_up_filter = get_limit_up_filter(stock_list, date_list, index_item)
    factor = factor.mask(limit_up_filter)

    # 4. 保存因子数据
    processed_path = get_data_path(
        "factor_processed",
        filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{universe_start}_{universe_end}.pkl",
        index_item=index_item,
        neutralize=neutralize,
    )
    factor.to_pickle(processed_path)
    print(f"✅processed_factor已保存到: {processed_path}")

    return factor


def factor_layered_backtest(
    df,
    index_item,
    direction,
    neutralize,
    factor_name,
    g=10,
    rebalance_days=20,
    rebalance=False,
):
    """
    因子分层回测函数（优化版）

    :param df: 因子值 DataFrame (unstack格式)
    :param g: 分组数量
    :param index_item: 券池名称
    :param direction: 因子方向（1或-1）
    :param neutralize: 是否中性化
    :param rebalance_days: 调仓频率（天数），默认20天
    :param name: 因子名称
    :param rebalance: 是否每日重新平衡权重
        - True: 每日等权重平均（高交易成本）
        - False: 买入持有策略（低交易成本）
    :return: (group_return, turnover_ratio)
        - group_return: 各分组的累计净值表现
        - turnover_ratio: 各分组的换手率
    """

    # 信号下移一个交易日
    df = df.shift(1).iloc[1:]
    order_book_ids = df.columns.tolist()
    date_list = df.index.tolist()
    start = date_list[0].strftime("%F")
    end = date_list[-1].strftime("%F")

    stock_universe = INDEX_FIX(start, end, index_item)

    # 尝试从路径读取return_1d
    new_path = get_data_path("return_1d", index_item=index_item, start=start, end=end)

    return_1d = None

    # 尝试加载数据
    try:
        return_1d = pd.read_pickle(new_path)
        print(f"✅加载return_1d: {new_path}")
    except:
        print(f"✅计算新的return_1d...")

        # 如果没有找到，则重新计算
        if return_1d is None:
            order_book_ids = stock_universe.columns.tolist()
            open = get_price(
                order_book_ids,
                start,
                get_next_trading_date(end, 1),
                "1d",
                "open",
                "pre",
                False,
                True,
            ).open.unstack("order_book_id")
            return_1d = open.pct_change().shift(-1).dropna(axis=0, how="all").stack()

            # 保存到新路径
            return_1d.to_pickle(new_path)
            print(f"✅保存return_1d: {new_path}")

    # 数据合并，使用multiindex
    factor_data = df.stack().to_frame("factor")
    factor_data["current_return"] = return_1d
    factor_data = factor_data.dropna()

    # 获取调仓日期和数据边界
    actual_rebalance_dates = date_list[::rebalance_days]  # 真正的调仓日期

    # 批量分组
    all_groups = {}
    turnover_data = []

    ##########计算调仓日分组和换手率##########
    # 为所有调仓日期构建分组信息
    for i, rebalance_date in enumerate(actual_rebalance_dates):
        # 获取当前因子值
        current_factors = factor_data.loc[rebalance_date, "factor"]

        # 使用qcut进行分组
        groups = pd.qcut(current_factors, g, labels=range(1, g + 1))
        current_groups = (
            current_factors.groupby(groups).apply(lambda x: x.index.tolist()).to_dict()
        )

        # 计算换手率（除了第一次）
        if i > 0:
            turnover_rates = []
            for group_id in range(1, g + 1):
                if group_id in all_groups[i - 1] and group_id in current_groups:
                    old_stocks = set(all_groups[i - 1][group_id])
                    new_stocks = set(current_groups[group_id])
                    turnover = (
                        len(old_stocks - new_stocks) / len(old_stocks)
                        if old_stocks
                        else 0
                    )
                    turnover_rates.append(turnover)
                else:
                    turnover_rates.append(np.nan)

            turnover_data.append({"date": rebalance_date, "turnover": turnover_rates})

        all_groups[i] = current_groups

    if turnover_data:
        turnover_ratio = pd.DataFrame(
            [d["turnover"] for d in turnover_data],
            index=[d["date"] for d in turnover_data],
            columns=[f"G{i}" for i in range(1, g + 1)],
        )
    else:
        turnover_ratio = pd.DataFrame()

    ##########计算分组收益##########
    group_returns_list = []

    # 处理所有调仓周期（周期含义：调仓日）
    for i, start_date in enumerate(actual_rebalance_dates):

        # 确定周期结束日期
        is_last_period = i == len(actual_rebalance_dates) - 1

        if not is_last_period:
            # 正常周期：到下一个调仓日
            end_date = actual_rebalance_dates[i + 1]
            period_mask = (factor_data.index.get_level_values(0) >= start_date) & (
                factor_data.index.get_level_values(0) < end_date
            )
        else:
            # 最后一个周期：到数据结束日
            period_mask = factor_data.index.get_level_values(0) >= start_date

        period_data = factor_data[period_mask]

        if i not in all_groups:
            continue

        group_dict = all_groups[i]
        period_returns = {}

        for group_id in range(1, g + 1):
            if group_id not in group_dict:
                continue

            stocks = group_dict[group_id]
            # 获取该组股票的收益率数据
            group_data = period_data[period_data.index.get_level_values(1).isin(stocks)]

            if len(group_data) == 0:
                continue

            # 根据rebalance参数选择计算方式，有每日重新平衡和买入持有两种方式
            if rebalance:
                # 每日等权重平均（每天重新平衡权重）
                portfolio_daily_returns = group_data.groupby(level=0)[
                    "current_return"
                ].mean()
            else:

                # 将数据重新整理为日期×股票的格式
                stock_daily_returns = group_data.reset_index().pivot(
                    index="datetime", columns="order_book_id", values="current_return"
                )
                # 等权重买入股票，不重新平衡
                group_cum_returns = (1 + stock_daily_returns).cumprod().mean(axis=1)
                portfolio_daily_returns = group_cum_returns.pct_change()  # 转为日收益率

                # 处理第一天的收益率
                if not portfolio_daily_returns.empty:
                    portfolio_daily_returns.iloc[0] = stock_daily_returns.iloc[0].mean()

            period_returns[f"G{group_id}"] = portfolio_daily_returns

        # 将该期间收益率添加到总列表中
        if period_returns:
            period_df = pd.DataFrame(period_returns)
            group_returns_list.append(period_df)

    if group_returns_list:
        group_return = pd.concat(group_returns_list, axis=0)
    else:
        group_return = pd.DataFrame()

    if not group_return.empty:
        # 基准收益
        group_return["Benchmark"] = group_return.mean(axis=1)
        group_return = (group_return + 1).cumprod()

        # 计算年化收益率
        group_annual_ret = group_return.iloc[-1] ** (252 / len(group_return)) - 1
        group_annual_ret = group_annual_ret - group_annual_ret.Benchmark
        group_annual_ret = group_annual_ret.drop("Benchmark").to_frame("annual_ret")
        group_annual_ret["group"] = list(range(1, g + 1))
        corr_value = round(group_annual_ret.corr(method="spearman").iloc[0, 1], 4)

    # 绘制分层回测结果
    print(f"✅生成分层分析图表...")
    plot_factor_layered_analysis(
        group_return,
        index_item,
        direction,
        neutralize,
        rebalance_days=rebalance_days,
        stock_universe=stock_universe,
        factor_name=factor_name,
    )

    return group_return, turnover_ratio


# 因子分层回测结果可视化 - 三图合一
def plot_factor_layered_analysis(
    group_cumulative,
    index_item,
    direction,
    neutralize,
    rebalance_days,
    stock_universe,
    factor_name="",
    save_path=None,
    show_plot=False,
):
    """
    基于factor_layered_backtest的结果生成三图合一的综合分析图

    :param group_cumulative: 分组累积收益数据（包含Benchmark列） -> DataFrame
    :param index_item: 基准指数 -> str
    :param factor_name: 因子名称 -> str
    :param stock_universe: 股票池数据 -> DataFrame
    :param save_path: 保存路径，如果为None则使用默认路径 -> str
    :param show_plot: 是否显示图形 -> bool
    """
    # 导入必要的库
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    import pandas as pd

    # 设置中文字体
    rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # 直接使用传入的累积收益数据
    g = len(group_cumulative.columns) - 1  # 减去 Benchmark 列

    # 计算年化收益和超额收益
    group_annual_ret = group_cumulative.iloc[-1] ** (252 / len(group_cumulative)) - 1
    group_annual_ret_excess = group_annual_ret - group_annual_ret.Benchmark
    group_annual_ret_excess = group_annual_ret_excess.drop("Benchmark")

    # 计算单调性
    group_numbers = list(range(1, g + 1))
    corr_value = round(
        pd.Series(group_annual_ret_excess.values).corr(
            pd.Series(group_numbers), method="spearman"
        ),
        4,
    )

    # 计算逐年分层收益（使用与原函数相同的方法）
    daily_returns = group_cumulative.pct_change().dropna()

    # 按年重采样计算年化收益，然后转置
    yby_performance = (
        daily_returns.resample("Y")
        .apply(lambda x: (1 + x).cumprod().iloc[-1])
        .T  # 转置：行为分组，列为年份
    )

    # 减去基准收益（计算超额收益）
    if "Benchmark" in yby_performance.index:
        yby_performance = yby_performance - yby_performance.loc["Benchmark"]
        yby_performance = yby_performance.drop("Benchmark")

    # 创建三图合一的图形
    fig = plt.figure(figsize=(16, 12))

    # 与原函数完全一致的颜色方案
    base_colors = [
        "#1f77b4",  # 蓝色
        "#ff7f0e",  # 橙色
        "#2ca02c",  # 绿色
        "#d62728",  # 红色
        "#9467bd",  # 紫色
        "#8c564b",  # 棕色
        "#e377c2",  # 粉色
        "#7f7f7f",  # 灰色
        "#bcbd22",  # 橄榄色
        "#000080",  # 深蓝色
        "#FF1493",  # 深粉色
        "#00CED1",  # 暗青色
        "#FF4500",  # 橙红色
        "#32CD32",  # 酸橙绿
        "#8A2BE2",  # 蓝紫色
        "#DC143C",  # 深红色
        "#00BFFF",  # 深天蓝
        "#FFD700",  # 金色
        "#FF69B4",  # 热粉色
        "#228B22",  # 森林绿
    ]

    # 子图1: 分层超额年化收益柱状图 (左上) - 使用默认颜色
    ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
    group_annual_ret_excess.plot(
        kind="bar",
        ax=ax1,
        title=f"{factor_name}_分层超额年化收益_单调性{corr_value}_{neutralize}_{direction}",
    )
    ax1.set_xlabel("分组")
    ax1.set_ylabel("超额年化收益")
    ax1.grid(True, alpha=0.3)

    # 子图2: 分层净值表现线图 (右上) - 突出G1和G10，弱化其他组
    ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
    for col in group_cumulative.columns:
        if col in ["G1", "G10"]:  # G1和G10组
            linewidth = 2
            alpha = 1.0
            if col == "G10":
                color = "#FF0000"  # 鲜红色
                label = "G10"
            else:  # G1组
                color = "#00AA00"  # 鲜绿色
                label = "G1"
        elif col == "Benchmark":
            linewidth = 2
            alpha = 0.8
            color = "#000000"  # 黑色
            label = "Benchmark"
        else:
            linewidth = 1
            alpha = 0.6
            color = "#888888"  # 中等灰色，更易识别
            label = col

        ax2.plot(
            group_cumulative.index,
            group_cumulative[col],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    ax2.set_title(
        f"{factor_name}_分层净值表现_单调性{corr_value}_{neutralize}_{direction}"
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("累积收益")

    # 子图3: 逐年分层年化收益柱状图 (下方跨两列)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)

    # 为逐年图设置颜色（按年份设置，与原函数一致）
    num_years = len(yby_performance.columns)
    if num_years > len(base_colors):
        year_colors = [plt.cm.tab20(i / num_years) for i in range(num_years)]
    else:
        year_colors = base_colors[:num_years]

    yby_performance.plot(
        kind="bar",
        ax=ax3,
        title=f"{factor_name}_逐年分层年化收益_单调性{corr_value}_{neutralize}_{direction}",
        color=year_colors,
    )

    # 修改图例显示：完全平铺，只显示年份
    legend_labels = [
        str(col.year) + "年" if hasattr(col, "year") else str(col)
        for col in yby_performance.columns
    ]
    ax3.legend(
        legend_labels,
        bbox_to_anchor=(0.5, -0.12),
        loc="upper center",
        ncol=len(yby_performance.columns),
        fontsize=8,
    )
    ax3.set_xlabel("分组")
    ax3.set_ylabel("年化收益")
    ax3.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # 保存图片
    if save_path is None:
        if factor_name and stock_universe is not None:
            import os
            from datetime import datetime

            start_date = stock_universe.index[0].strftime("%Y-%m-%d")
            end_date = stock_universe.index[-1].strftime("%Y-%m-%d")

            # 使用get_data_path生成路径，根据index_item分类存放
            save_path = get_data_path(
                "layered_report",
                factor_name=factor_name,
                index_item=index_item,
                direction=direction,
                neutralize=neutralize,
                start_date=start_date,
                end_date=end_date,
                rebalance_days=rebalance_days,
            )

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"图片已保存到: {save_path}")

    # 显示图形
    if show_plot:
        plt.show()
    else:
        plt.close()  # 如果不显示，则关闭图形以释放内存
