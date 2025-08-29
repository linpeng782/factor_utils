import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *
from datetime import datetime

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


# 创建分离式策略报告：收益曲线图 + 绩效指标表
from matplotlib import rcParams
import os

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
import warnings

warnings.filterwarnings("ignore")


def get_buy_list(df, top_type="rank", rank_n=100, quantile_q=0.8):
    """
    :param df: 因子值 -> dataframe/unstack
    :param top_type: 选择买入队列方式，从['rank','quantile']选择一种方式 -> str
    :param rank_n: 值最大的前n只的股票 -> int
    :param quantile_q: 值最大的前n分位数的股票 -> float
    :return df: 买入队列 -> dataframe/unstack
    """

    if top_type == "rank":
        df = df.rank(axis=1, ascending=False) <= rank_n
    elif top_type == "quantile":
        df = df.sub(df.quantile(quantile_q, axis=1), axis=0) > 0
    else:
        print("select one from ['rank','quantile']")

    df = df.astype(int)
    df = df.replace(0, np.nan).dropna(how="all", axis=1)

    return df


def calc_transaction_fee(
    transaction_value, min_transaction_fee, sell_cost_rate, buy_cost_rate
):
    """
    计算单笔交易的手续费

    :param transaction_value: 交易金额（正数为买入，负数为卖出）
    :param min_transaction_fee: 最低交易手续费
    :param sell_cost_rate: 卖出成本费率
    :param buy_cost_rate: 买入成本费率
    :return: 交易手续费
    """
    if pd.isna(transaction_value) or transaction_value == 0:
        return 0  # 无交易时手续费为0
    elif transaction_value < 0:  # 卖出交易（负数）
        fee = -transaction_value * sell_cost_rate  # 卖出手续费：印花税 + 过户费 + 佣金
    else:  # 买入交易（正数）
        fee = transaction_value * buy_cost_rate  # 买入手续费：过户费 + 佣金

    # 应用最低手续费限制
    return max(fee, min_transaction_fee)  # 返回实际手续费和最低手续费中的较大值


def calculate_target_holdings(
    target_weights, available_cash, stock_prices, min_trade_units, sell_cost_rate
):
    """
    计算目标持仓数量

    :param target_weights: 目标权重 Series
    :param available_cash: 可用资金
    :param stock_prices: 股票价格 Series
    :param min_trade_units: 最小交易单位 Series
    :param sell_cost_rate: 卖出成本费率（用于预留手续费）
    :return: 目标持仓数量 Series
    """
    # 按权重分配资金
    allocated_cash = target_weights * available_cash

    # 计算调整后价格（预留卖出手续费）
    adjusted_prices = stock_prices * (1 + sell_cost_rate)

    # 计算可购买的最小交易单位数量（向下取整）
    units_to_buy = allocated_cash / adjusted_prices // min_trade_units

    # 转换为实际股数
    target_holdings = units_to_buy * min_trade_units

    return target_holdings


def get_bar(df, adjust):
    """
    获取股票价格数据
    :param df: 投资组合权重矩阵
    :param adjust: 复权方式 ('none', 'pre', 'post')
    :return: 价格数据
    """
    start_date = df.index[0]
    end_date = df.index[-1]
    order_book_ids = df.columns.tolist()

    prices = get_price(
        order_book_ids,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields="open",
        adjust_type=adjust,
    ).open.unstack("order_book_id")

    return prices


def backtest(
    portfolio_weights,
    rebalance_frequency=20,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
    backtest_start_date=None,
):
    """
    量化策略回测框架

    :param portfolio_weights: 投资组合权重矩阵 -> DataFrame
    :param rebalance_frequency: 调仓频率（天数） -> int
    :param initial_capital: 初始资本 -> float
    :param stamp_tax_rate: 印花税率 -> float
    :param transfer_fee_rate: 过户费率 -> float
    :param commission_rate: 佣金率 -> float
    :param min_transaction_fee: 最低交易手续费 -> float
    :param cash_annual_yield: 现金年化收益率 -> float
    :param start_date: 回测起始日期，格式为'YYYY-MM-DD'或None（从数据开始日期开始） -> str or None
    :return: 账户历史记录 -> DataFrame
    """

    # =========================== 基础参数初始化 ===========================
    # 保存初始资本备份，用于最后的统计计算
    cash = initial_capital
    # 初始化历史持仓，第一次调仓时为0
    previous_holdings = 0
    # 买入成本费率：过户费 + 佣金
    buy_cost_rate = transfer_fee_rate + commission_rate
    # 卖出成本费率：印花税 + 过户费 + 佣金
    sell_cost_rate = stamp_tax_rate + transfer_fee_rate + commission_rate
    # 现金账户日利率（年化收益率转换为日收益率）
    daily_cash_yield = (1 + cash_annual_yield) ** (1 / 252) - 1
    # 筛选出回测期间的交易日
    all_signal_dates = portfolio_weights.index.tolist()
    # 处理起始日期参数
    if backtest_start_date is not None:
        # 将字符串日期转换为pandas Timestamp
        start_timestamp = pd.to_datetime(backtest_start_date)
        # 筛选出大于等于起始日期的信号日期
        filtered_signal_dates = [
            date for date in all_signal_dates if date >= start_timestamp
        ]
        if not filtered_signal_dates:
            raise ValueError(f"指定的起始日期 {backtest_start_date} 大于所有数据日期")
        all_signal_dates = filtered_signal_dates
        print(
            f"回测起始日期: {backtest_start_date}, 实际开始日期: {all_signal_dates[0].strftime('%Y-%m-%d')}"
        )
    # =========================== 数据结构初始化 ===========================
    # 创建账户历史记录表，索引为回测期间的交易日
    account_history = pd.DataFrame(
        index=pd.Index(all_signal_dates),  # 使用筛选后的日期范围
        columns=["total_account_asset", "holding_market_cap", "cash_account"],
    )
    # 获取所有股票的开盘价格数据（未复权）
    open_prices = get_bar(portfolio_weights, "none")
    # 获取所有股票的后复权价格数据
    adjusted_prices = get_bar(portfolio_weights, "post")
    # 获取每只股票的最小交易单位（通常为100股）
    min_trade_units = pd.Series(
        dict(
            [
                (stock, instruments(stock).round_lot)
                for stock in portfolio_weights.columns.tolist()
            ]
        )
    )

    # 生成调仓日期列表：每 rebalance_frequency 天调仓一次，最后一天也被包含在调仓日中
    rebalance_dates = sorted(
        set(all_signal_dates[::rebalance_frequency] + [all_signal_dates[-1]])
    )

    # =========================== 开始逐期调仓循环 ===========================
    for i in tqdm(range(0, len(rebalance_dates) - 1)):

        rebalance_date = rebalance_dates[i]  # 当前调仓日期
        next_rebalance_date = rebalance_dates[i + 1]  # 下一个调仓日期

        # if rebalance_date == pd.Timestamp("2016-05-27"):
        #     breakpoint()

        # =========================== 获取当前调仓日的目标权重 ===========================
        # 获取当前调仓日的目标权重，并删除缺失值
        target_weights = portfolio_weights.loc[rebalance_date].dropna()
        # 获取目标股票列表
        target_stocks = target_weights.index.tolist()

        # =========================== 计算目标持仓数量 ===========================
        target_holdings = calculate_target_holdings(
            target_weights,
            cash,
            open_prices.loc[rebalance_date, target_stocks],
            min_trade_units.loc[target_stocks],
            sell_cost_rate,
        )

        # =========================== 仓位变动计算 ===========================
        ## 步骤1：计算持仓变动量（目标持仓 - 历史持仓）
        # fill_value=0 确保新增股票（历史持仓为空）和清仓股票（目标持仓为空）都能正确计算
        holdings_change_raw = target_holdings.sub(
            previous_holdings, fill_value=0
        )  # 计算原始持仓变动量

        ## 步骤2：过滤掉无变动的股票（变动量为0的股票）
        # 将变动量为0的股票标记为NaN，然后删除，只保留需要调仓的股票
        holdings_change_filtered = holdings_change_raw.replace(0, np.nan)

        ## 步骤3：获取最终的交易执行列表
        # 正数表示需要买入的股数，负数表示需要卖出的股数
        # 删除NaN，只保留需要执行的交易
        trades_to_execute = holdings_change_filtered.dropna()

        # 获取当前调仓日的所有股票开盘价
        current_prices = open_prices.loc[rebalance_date]

        # =========================== 计算交易成本 ===========================
        # 计算总交易成本：交易金额 = 价格 * 交易股数
        # 计算每笔交易的交易金额
        # 对每笔交易计算手续费
        # 求和得到总手续费
        total_transaction_cost = (
            (current_prices * trades_to_execute)
            .apply(
                lambda x: calc_transaction_fee(
                    x, min_transaction_fee, sell_cost_rate, buy_cost_rate
                )
            )
            .sum()
        )

        # =========================== 价格复权调整 ===========================
        # 从调仓日到下一调仓日的后复权价格
        period_adj_prices = adjusted_prices.loc[rebalance_date:next_rebalance_date]
        # 调仓日的后复权价格(基准)
        base_adj_prices = adjusted_prices.loc[rebalance_date]
        # 价格变动倍数
        price_multipliers = period_adj_prices.div(base_adj_prices, axis=1)
        # 模拟未复权价格
        simulated_prices = price_multipliers.mul(current_prices, axis=1).dropna(
            axis=1, how="all"
        )

        # =========================== 计算投资组合市值 ===========================
        # 投资组合市值 = 每只股票的(模拟未复权价格 * 持仓数量)的总和
        # 按日计算投资组合市值
        portfolio_market_value = (simulated_prices * target_holdings).sum(axis=1)

        # =========================== 计算现金账户余额 ===========================
        # 初始现金余额 = 可用资金 - 交易成本 - 初始投资金额
        initial_cash_balance = (
            cash - total_transaction_cost - portfolio_market_value.loc[rebalance_date]
        )

        # 计算期间现金账户的复利增长（按日计息）
        cash_balance = pd.Series(
            [
                initial_cash_balance
                * ((1 + daily_cash_yield) ** (day + 1))  # 复利计息公式
                for day in range(0, len(portfolio_market_value))
            ],  # 对每一天计算
            index=portfolio_market_value.index,
        )  # 使用相同的日期索引

        # =========================== 计算账户总资产 ===========================
        total_portfolio_value = (
            portfolio_market_value + cash_balance
        )  # 总资产 = 持仓市值 + 现金余额

        # =========================== 更新历史数据为下一次调仓做准备 ===========================
        previous_holdings = target_holdings  # 更新历史持仓为当前目标持仓
        cash = total_portfolio_value.loc[
            next_rebalance_date
        ]  # 更新可用资金为下一调仓日的账户总值

        # =========================== 保存账户历史记录 ===========================
        # 将当前期间的账户数据保存到历史记录中（保留2位小数）
        account_history.loc[
            rebalance_date:next_rebalance_date, "total_account_asset"
        ] = round(total_portfolio_value, 2)
        account_history.loc[
            rebalance_date:next_rebalance_date, "holding_market_cap"
        ] = round(portfolio_market_value, 2)
        account_history.loc[rebalance_date:next_rebalance_date, "cash_account"] = round(
            cash_balance, 2
        )

    # =========================== 添加初始日记录并排序 ===========================
    # 在第一个交易日之前添加初始资本记录
    initial_date = pd.to_datetime(
        get_previous_trading_date(account_history.index.min(), 1)
    )
    account_history.loc[initial_date] = [
        initial_capital,
        0,
        initial_capital,
    ]  # [总资产, 持仓市值, 现金余额]
    account_history = account_history.sort_index()  # 按日期排序

    return account_history  # 返回完整的账户历史记录


def get_benchmark(df, benchmark, benchmark_type="mcw"):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(), 1).strftime("%F")
    end_date = df.index.max().strftime("%F")
    if benchmark_type == "mcw":
        price_open = get_price(
            [benchmark], start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
    else:
        index_fix = INDEX_FIX(start_date, end_date, benchmark)
        stock_list = index_fix.columns.tolist()
        price_open = get_price(
            stock_list, start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
        price_open = price_open.pct_change().mask(~index_fix).mean(axis=1)
        price_open = (1 + price_open).cumprod().to_frame(benchmark)

    return price_open


def get_performance_analysis(
    account_result,
    direction,
    neutralize,
    rf=0.03,
    benchmark_index="000985.XSHG",
    factor_name=None,
    start_date=None,
    end_date=None,
    show_plot=False,
):

    # 加入基准
    performance = pd.concat(
        [
            account_result["total_account_asset"].to_frame("strategy"),
            get_benchmark(account_result, benchmark_index),
        ],
        axis=1,
    )
    daily_returns = performance.pct_change().dropna(how="all")  # 日收益率
    cumulative_returns = (1 + daily_returns).cumprod()  # 累计收益
    cumulative_returns["alpha"] = (
        cumulative_returns["strategy"] / cumulative_returns[benchmark_index]
    )
    cumulative_returns = cumulative_returns.fillna(1)

    # 指标计算
    daily_pct_change = cumulative_returns.pct_change().dropna()

    # 策略收益
    strategy_name, benchmark_name, alpha_name = cumulative_returns.columns.tolist()
    Strategy_Final_Return = cumulative_returns[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (
        252 / len(cumulative_returns)
    ) - 1

    # 策略年化收益(算术)
    Strategy_Annualized_Return_AM = daily_pct_change[strategy_name].mean() * 252

    # 基准收益
    Benchmark_Final_Return = cumulative_returns[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (
        252 / len(cumulative_returns)
    ) - 1

    # alpha
    ols_result = sm.OLS(
        daily_pct_change[strategy_name] * 252 - rf,
        sm.add_constant(daily_pct_change[benchmark_name] * 252 - rf),
    ).fit()
    Alpha = ols_result.params[0]

    # beta
    Beta = ols_result.params[1]

    # 波动率
    Strategy_Volatility = daily_pct_change[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf) / Strategy_Volatility

    # 夏普(算术)
    Strategy_Sharpe_AM = (Strategy_Annualized_Return_AM - rf) / Strategy_Volatility

    # 下行波动率
    strategy_ret = daily_pct_change[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf) / Strategy_Down_Volatility

    # 跟踪误差
    Tracking_Error = (
        daily_pct_change[strategy_name] - daily_pct_change[benchmark_name]
    ).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (
        Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR
    ) / Tracking_Error

    # 最大回撤计算（分步骤进行）
    # 第1步：获取策略累计收益序列
    strategy_cumulative = cumulative_returns[strategy_name]

    # 第2步：计算到每个时点的历史最高点
    running_max = np.maximum.accumulate(strategy_cumulative)

    # 第3步：计算每个时点的回撤比例
    drawdown_ratio = (running_max - strategy_cumulative) / running_max

    # 第4步：识别所有回撤周期并计算前三大回撤
    drawdown_periods = []

    # 找到所有局部峰值点（新高点）
    is_new_high = strategy_cumulative == running_max
    peak_indices = strategy_cumulative[is_new_high].index

    # 对每个峰值，找到后续的最大回撤
    for k in range(len(peak_indices) - 1):
        peak_idx = peak_indices[k]
        next_peak_idx = peak_indices[k + 1]

        # 在这个峰值到下个峰值之间找最大回撤
        period_mask = (strategy_cumulative.index >= peak_idx) & (
            strategy_cumulative.index < next_peak_idx
        )
        if period_mask.sum() > 1:  # 确保有足够的数据点
            period_cumulative = strategy_cumulative[period_mask]
            period_drawdown = drawdown_ratio[period_mask]

            if (
                len(period_drawdown) > 0 and period_drawdown.max() > 0.01
            ):  # 只考虑回撤超过1%的情况
                trough_idx = period_drawdown.idxmax()
                peak_value = strategy_cumulative[peak_idx]
                trough_value = strategy_cumulative[trough_idx]
                drawdown_pct = period_drawdown.max()

                drawdown_periods.append(
                    {
                        "peak_date": peak_idx,
                        "trough_date": trough_idx,
                        "peak_value": peak_value,
                        "trough_value": trough_value,
                        "drawdown": drawdown_pct,
                    }
                )

    # 处理最后一个峰值到结束的回撤
    if len(peak_indices) > 0:
        last_peak_idx = peak_indices[-1]
        period_mask = strategy_cumulative.index >= last_peak_idx
        period_cumulative = strategy_cumulative[period_mask]
        period_drawdown = drawdown_ratio[period_mask]

        if len(period_drawdown) > 0 and period_drawdown.max() > 0.01:
            trough_idx = period_drawdown.idxmax()
            peak_value = strategy_cumulative[last_peak_idx]
            trough_value = strategy_cumulative[trough_idx]
            drawdown_pct = period_drawdown.max()

            drawdown_periods.append(
                {
                    "peak_date": last_peak_idx,
                    "trough_date": trough_idx,
                    "peak_value": peak_value,
                    "trough_value": trough_value,
                    "drawdown": drawdown_pct,
                }
            )

    # 按回撤幅度排序，取前三大
    drawdown_periods.sort(key=lambda x: x["drawdown"], reverse=True)
    top_3_drawdowns = drawdown_periods[:3]

    print("\n=== 前三大回撤分析 ===")
    for i, dd in enumerate(top_3_drawdowns, 1):
        print(f"第{i}大回撤:")
        print(f"  峰值日期: {dd['peak_date']} (净值: {dd['peak_value']:.4f})")
        print(f"  谷值日期: {dd['trough_date']} (净值: {dd['trough_value']:.4f})")
        print(f"  回撤幅度: {dd['drawdown']:.4f} ({dd['drawdown']*100:.2f}%)")
        print()

    # 第5步：获取最大回撤值
    Max_Drawdown = top_3_drawdowns[0]["drawdown"] if top_3_drawdowns else 0

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR) / Max_Drawdown

    # 超额收益
    Alpha_Final_Return = cumulative_returns[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (
        252 / len(cumulative_returns)
    ) - 1

    # 超额波动率
    Alpha_Volatility = daily_pct_change[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf) / Alpha_Volatility

    # 超额最大回撤
    i = np.argmax(
        (
            np.maximum.accumulate(cumulative_returns[alpha_name])
            - cumulative_returns[alpha_name]
        )
        / np.maximum.accumulate(cumulative_returns[alpha_name])
    )
    j = np.argmax(cumulative_returns[alpha_name][:i])
    Alpha_Max_Drawdown = (
        1 - cumulative_returns[alpha_name][i] / cumulative_returns[alpha_name][j]
    )

    # 胜率
    daily_pct_change["win"] = daily_pct_change[alpha_name] > 0
    Win_Ratio = daily_pct_change["win"].value_counts().loc[True] / len(daily_pct_change)

    # 盈亏比
    profit_lose = daily_pct_change.groupby("win")[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True] / profit_lose[False])

    result = {
        "策略累计收益": round(Strategy_Final_Return, 4),
        "策略年化收益": round(Strategy_Annualized_Return_EAR, 4),
        "策略年化收益(算术)": round(Strategy_Annualized_Return_AM, 4),
        "基准累计收益": round(Benchmark_Final_Return, 4),
        "基准年化收益": round(Benchmark_Annualized_Return_EAR, 4),
        "阿尔法": round(Alpha, 4),
        "贝塔": round(Beta, 4),
        "波动率": round(Strategy_Volatility, 4),
        "夏普比率": round(Strategy_Sharpe, 4),
        "夏普比率(算术)": round(Strategy_Sharpe_AM, 4),
        "下行波动率": round(Strategy_Down_Volatility, 4),
        "索提诺比率": round(Sortino, 4),
        "跟踪误差": round(Tracking_Error, 4),
        "信息比率": round(Information_Ratio, 4),
        "最大回撤": round(Max_Drawdown, 4),
        "卡玛比率": round(Calmar, 4),
        "超额累计收益": round(Alpha_Final_Return, 4),
        "超额年化收益": round(Alpha_Annualized_Return_EAR, 4),
        "超额波动率": round(Alpha_Volatility, 4),
        "超额夏普": round(Alpha_Sharpe, 4),
        "超额最大回撤": round(Alpha_Max_Drawdown, 4),
        "胜率": round(Win_Ratio, 4),
        "盈亏比": round(Profit_Lose_Ratio, 4),
    }

    # 使用get_data_path生成路径，根据index_item和日期分类存放
    from factor_utils.path_manager import get_data_path

    chart_path = get_data_path(
        "performance_chart",
        factor_name=factor_name,
        benchmark_index=benchmark_index,
        direction=direction,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
        index_item=benchmark_index,  # 使用benchmark_index作为index_item
    )

    table_path = get_data_path(
        "metrics_table",
        factor_name=factor_name,
        benchmark_index=benchmark_index,
        direction=direction,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
        index_item=benchmark_index,  # 使用benchmark_index作为index_item
    )

    # ==================== 图1：绩效指标表 ====================
    fig2, ax3 = plt.subplots(figsize=(12, 16))  # 竖向布局，适合表格
    ax3.axis("off")

    # 准备表格数据
    result_df = pd.DataFrame([result]).T
    result_df.columns = ["数值"]

    # 创建表格数据
    table_data = []
    for idx, row in result_df.iterrows():
        table_data.append([idx, f"{row['数值']:.4f}"])

    # 添加回撤区间信息
    try:
        if len(top_3_drawdowns) > 0:
            table_data.append(["回撤区间分析", ""])
            for i, dd in enumerate(top_3_drawdowns, 1):
                peak_date_str = dd["peak_date"].strftime("%Y-%m-%d")
                trough_date_str = dd["trough_date"].strftime("%Y-%m-%d")
                period_str = f"{peak_date_str} ~ {trough_date_str}"
                drawdown_str = f"{dd['drawdown']*100:.2f}%"
                table_data.append([f"第{i}大回撤区间", period_str])
                table_data.append([f"第{i}大回撤幅度", drawdown_str])
    except NameError:
        # 如果top_3_drawdowns变量不存在，则跳过
        pass

    # 绘制表格
    table = ax3.table(
        cellText=table_data,
        colLabels=["绩效指标", "数值"],
        cellLoc="left",
        loc="center",
        colWidths=[0.7, 0.3],
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # 更大的字体
    table.scale(1, 2.2)  # 更大的行高

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white", size=14)

    # 设置交替行颜色和样式
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor("#F8F9FA")
        # 设置数值列的字体为粗体
        table[(i, 1)].set_text_props(weight="bold")

    # 添加标题
    ax3.text(
        0.5,
        1.00,
        f'{factor_name or "策略"}_绩效指标表_{start_date}_{end_date}',
        transform=ax3.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    plt.tight_layout()

    # 保存绩效指标表
    if factor_name:
        plt.savefig(table_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"绩效指标表已保存到: {table_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ==================== 图1：收益曲线图 ====================
    fig1, ax1 = plt.subplots(figsize=(16, 9))  # 16:9比例，更适合时间序列

    # 绘制策略和基准收益曲线
    ax1.plot(
        cumulative_returns.index,
        cumulative_returns[strategy_name],
        color="#1f77b4",
        linewidth=2.5,
        label="策略收益",
        alpha=0.9,
    )
    ax1.plot(
        cumulative_returns.index,
        cumulative_returns[benchmark_name],
        color="#ff7f0e",
        linewidth=2.5,
        label="基准收益",
        alpha=0.9,
    )

    # 创建第二个y轴显示超额收益
    ax2 = ax1.twinx()
    ax2.plot(
        cumulative_returns.index,
        cumulative_returns[alpha_name],
        color="#2ca02c",
        linewidth=2,
        alpha=0.7,
        label="超额收益",
    )
    ax2.set_ylabel("超额收益", color="#2ca02c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # 设置主图样式
    ax1.set_title(
        f'{factor_name or "策略"}_收益曲线分析_{start_date}_{end_date}',
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xlabel("日期", fontsize=12)
    ax1.set_ylabel("累积收益", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=11)
    ax2.legend(loc="upper right", fontsize=11)

    # 美化图表
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()

    # 保存收益曲线图
    if factor_name:
        plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"收益曲线图已保存到: {chart_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 打印结果表格到控制台
    print(pd.DataFrame([result]).T)

    return cumulative_returns, result
