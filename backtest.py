import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")

from datetime import datetime


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

    # =========================== 数据结构初始化 ===========================
    # 创建账户历史记录表，索引为所有交易日
    account_history = pd.DataFrame(
        index=portfolio_weights.index,
        # 列1：账户总资产
        # 列2：持仓市值
        # 列3：现金账户余额
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
    all_signal_dates = portfolio_weights.index.tolist()
    # 生成调仓日期列表：每 rebalance_frequency 天调仓一次，最后一天也被包含在调仓日中
    rebalance_dates = sorted(
        set(all_signal_dates[::rebalance_frequency] + [all_signal_dates[-1]])
    )

    # =========================== 开始逐期调仓循环 ===========================
    for i in tqdm(range(0, len(rebalance_dates) - 1)):
        rebalance_date = rebalance_dates[i]  # 当前调仓日期
        next_rebalance_date = rebalance_dates[i + 1]  # 下一个调仓日期

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
        # 将当前期间的账户数据保存到历史记录中（保疙2位小数）
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
    rebalance_days=20,
    rf=0.03,
    benchmark_index="000985.XSHG",
    factor_name=None,
    stock_universe=None,
    save_path=None,
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
    performance_net = performance.pct_change().dropna(how="all")  # 清算至当日开盘
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet["alpha"] = (
        performance_cumnet["strategy"] / performance_cumnet[benchmark_index]
    )
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name, benchmark_name, alpha_name = performance_cumnet.columns.tolist()
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 策略年化收益(算术)
    Strategy_Annualized_Return_AM = performance_pct[strategy_name].mean() * 252

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # alpha
    ols_result = sm.OLS(
        performance_pct[strategy_name] * 252 - rf,
        sm.add_constant(performance_pct[benchmark_name] * 252 - rf),
    ).fit()
    Alpha = ols_result.params[0]

    # beta
    Beta = ols_result.params[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf) / Strategy_Volatility

    # 夏普(算术)
    Strategy_Sharpe_AM = (Strategy_Annualized_Return_AM - rf) / Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf) / Strategy_Down_Volatility

    # 跟踪误差
    Tracking_Error = (
        performance_pct[strategy_name] - performance_pct[benchmark_name]
    ).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (
        Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR
    ) / Tracking_Error

    # 最大回撤
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[strategy_name])
            - performance_cumnet[strategy_name]
        )
        / np.maximum.accumulate(performance_cumnet[strategy_name])
    )
    j = np.argmax(performance_cumnet[strategy_name][:i])
    Max_Drawdown = (
        1 - performance_cumnet[strategy_name][i] / performance_cumnet[strategy_name][j]
    )

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR) / Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf) / Alpha_Volatility

    # 超额最大回撤
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[alpha_name])
            - performance_cumnet[alpha_name]
        )
        / np.maximum.accumulate(performance_cumnet[alpha_name])
    )
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (
        1 - performance_cumnet[alpha_name][i] / performance_cumnet[alpha_name][j]
    )

    # 胜率
    performance_pct["win"] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct["win"].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby("win")[alpha_name].mean()
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

    # 创建分离式策略报告：收益曲线图 + 绩效指标表
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import os

    # 设置中文字体
    rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # 生成文件名和路径
    if factor_name and stock_universe is not None:
        from datetime import datetime

        start_date = stock_universe.index[0].strftime("%Y-%m-%d")
        end_date = stock_universe.index[-1].strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # 分别为收益曲线和指标表格创建独立目录
        charts_dir = "/Users/didi/KDCJ/alpha_local/outputs/reports/performance_charts"
        tables_dir = "/Users/didi/KDCJ/alpha_local/outputs/reports/metrics_tables"
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        chart_filename = f"{factor_name}_{benchmark_index}_{direction}_{neutralize}_{start_date}_{end_date}_{rebalance_days}_performance_chart.png"
        table_filename = f"{factor_name}_{benchmark_index}_{direction}_{neutralize}_{start_date}_{end_date}_{rebalance_days}_metrics_table.png"
        chart_path = os.path.join(charts_dir, chart_filename)
        table_path = os.path.join(tables_dir, table_filename)

    # ==================== 图1：收益曲线图 ====================
    fig1, ax1 = plt.subplots(figsize=(16, 9))  # 16:9比例，更适合时间序列

    # 绘制策略和基准收益曲线
    ax1.plot(
        performance_cumnet.index,
        performance_cumnet[strategy_name],
        color="#1f77b4",
        linewidth=2.5,
        label="策略收益",
        alpha=0.9,
    )
    ax1.plot(
        performance_cumnet.index,
        performance_cumnet[benchmark_name],
        color="#ff7f0e",
        linewidth=2.5,
        label="基准收益",
        alpha=0.9,
    )

    # 创建第二个y轴显示超额收益
    ax2 = ax1.twinx()
    ax2.plot(
        performance_cumnet.index,
        performance_cumnet[alpha_name],
        color="#2ca02c",
        linewidth=2,
        alpha=0.7,
        label="超额收益",
    )
    ax2.set_ylabel("超额收益", color="#2ca02c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # 设置主图样式
    ax1.set_title(
        f'{factor_name or "策略"}_收益曲线分析', fontsize=18, fontweight="bold", pad=20
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
    if factor_name and stock_universe is not None:
        plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"收益曲线图已保存到: {chart_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ==================== 图2：绩效指标表 ====================
    fig2, ax3 = plt.subplots(figsize=(12, 16))  # 竖向布局，适合表格
    ax3.axis("off")

    # 准备表格数据
    result_df = pd.DataFrame([result]).T
    result_df.columns = ["数值"]

    # 创建表格数据
    table_data = []
    for idx, row in result_df.iterrows():
        table_data.append([idx, f"{row['数值']:.4f}"])

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
        0.95,
        f'{factor_name or "策略"}_绩效指标表',
        transform=ax3.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="top",
    )

    plt.tight_layout()

    # 保存绩效指标表
    if factor_name and stock_universe is not None:
        plt.savefig(table_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"绩效指标表已保存到: {table_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 打印结果表格到控制台
    print(pd.DataFrame([result]).T)

    return performance_cumnet, result
