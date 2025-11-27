import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.idx_bus import BUS_TYPE, REF
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_real_data(load_file, solar_file, wind_file, target_date='2017-06-27', auto_scale=True, target_capacity=6195):
    print(f"\n正在加载 {target_date} 的数据...")

    # 读取数据
    load_df = pd.read_csv(load_file)
    load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
    load_df['date'] = load_df['timestamp'].dt.date
    load_df['hour'] = load_df['timestamp'].dt.hour

    solar_df = pd.read_csv(solar_file)
    solar_df['timestamp'] = pd.to_datetime(solar_df['timestamp'])
    solar_cols = [col for col in solar_df.columns if col not in ['timestamp', 'BA']]
    solar_df['total_solar'] = solar_df[solar_cols].sum(axis=1)
    solar_df['date'] = solar_df['timestamp'].dt.date
    solar_df['hour'] = solar_df['timestamp'].dt.hour

    wind_df = pd.read_csv(wind_file)
    wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'])
    wind_df['date'] = wind_df['timestamp'].dt.date
    wind_df['hour'] = wind_df['timestamp'].dt.hour

    # 提取24小时数据
    target = pd.to_datetime(target_date).date()
    load_hourly = load_df[load_df['date'] == target].groupby('hour')['load_actuals_MW'].mean()
    solar_hourly = solar_df[solar_df['date'] == target].groupby('hour')['total_solar'].mean()
    wind_hourly = wind_df[wind_df['date'] == target].groupby('hour')['BA'].mean()

    load_24h = np.array([load_hourly.get(h, 0) for h in range(24)])
    solar_24h = np.array([solar_hourly.get(h, 0) for h in range(24)])
    wind_24h = np.array([wind_hourly.get(h, 0) for h in range(24)])

    print(f"✓ 原始负荷: {load_24h.min():.1f} - {load_24h.max():.1f} MW")
    print(f"✓ 原始光伏: {solar_24h.min():.1f} - {solar_24h.max():.1f} MW")
    print(f"✓ 原始风电: {wind_24h.min():.1f} - {wind_24h.max():.1f} MW")

    # 自动缩放
    if auto_scale and load_24h.max() > target_capacity * 0.95:
        scale_factor = (target_capacity * 0.90) / load_24h.max()
        print(f"\n峰值负荷 {load_24h.max():.1f} MW 超过系统容量 {target_capacity} MW")
        print(f"✓ 自动缩放到 {scale_factor * 100:.1f}% ({scale_factor:.4f}x)")
        load_24h, solar_24h, wind_24h = load_24h * scale_factor, solar_24h * scale_factor, wind_24h * scale_factor
        print(f"✓ 缩放后负荷: {load_24h.min():.1f} - {load_24h.max():.1f} MW")
        print(f"✓ 缩放后光伏: {solar_24h.min():.1f} - {solar_24h.max():.1f} MW")
        print(f"✓ 缩放后风电: {wind_24h.min():.1f} - {wind_24h.max():.1f} MW")

    # 能源平衡
    total_load = load_24h.sum()
    total_renewable = solar_24h.sum() + wind_24h.sum()
    print(
        f"\n能源平衡:总负荷: {total_load:.1f} MW, 可再生能源: {total_renewable:.1f} MW, 占比: {total_renewable / total_load * 100:.2f}%")

    return load_24h, solar_24h, wind_24h


class CFEMarket:
    def __init__(self, net, load_data, solar_data, wind_data, solver='gurobi', congestion_factor=0.65,
                 congestion_lines=None):
        self.net = net
        self.solver = solver
        self.T = 24
        self.congestion_factor = congestion_factor
        self.congestion_lines = congestion_lines if congestion_lines else []

        for idx in self.net.gen.index:
            bus_idx = self.net.gen.at[idx, 'bus']
            bus_max_vm = self.net.bus.at[bus_idx, 'max_vm_pu']
            if self.net.gen.at[idx, 'vm_pu'] > bus_max_vm:
                self.net.gen.at[idx, 'vm_pu'] = bus_max_vm

        pp.rundcpp(self.net)
        ppc = pp.converter.to_ppc(self.net)

        baseMVA, gen, bus, branch = ppc['baseMVA'], ppc['gen'], ppc['bus'], ppc['branch']

        # 初始化索引
        self.gen_idx = ['G' + str(int(gen[g, 0])) for g in range(gen.shape[0])]
        self.bus_idx = [str(b) for b in range(bus.shape[0])]
        self.branch_idx = [(str(int(branch[e, 0])), str(int(branch[e, 1]))) for e in range(branch.shape[0])]
        self.branch_num_idx = range(branch.shape[0])

        # 计算PTDF矩阵
        ref_bus = np.where(bus[:, BUS_TYPE] == REF)[0][0]
        self.PTDF_matrix = pd.DataFrame(
            makePTDF(baseMVA, bus, branch, slack=ref_bus, result_side=0, using_sparse_solver=True, branch_id=None,
                     reduced=False), index=self.branch_idx, columns=self.bus_idx)

        # 发电机和线路参数
        self.Pg_max = pd.DataFrame(gen[:, 8], index=self.gen_idx, columns=['Pg_max'])
        self.Pg_min = pd.DataFrame(gen[:, 9], index=self.gen_idx, columns=['Pg_min'])
        self.Pf_max = pd.DataFrame(branch[:, 5], index=self.branch_idx, columns=['Pf_max'])

        # 应用线路堵塞
        if self.congestion_lines:
            for line_idx in self.congestion_lines:
                if line_idx < len(self.branch_idx):
                    self.Pf_max.at[self.branch_idx[line_idx], 'Pf_max'] *= self.congestion_factor
            print(f"✓ 应用线路堵塞: {len(self.congestion_lines)}条线路, 容量系数={self.congestion_factor}")

        # 成本和排放
        np.random.seed(42)
        self.Cost = pd.DataFrame(np.linspace(30, 50, len(self.gen_idx)), index=self.gen_idx, columns=['Cost'])
        self.GCI = pd.DataFrame(np.linspace(0.4, 0.95, len(self.gen_idx)), index=self.gen_idx, columns=['GCI'])

        self._init_temporal_data(bus, load_data, solar_data, wind_data)

    def _init_temporal_data(self, bus, load_data, solar_data, wind_data):
        bus_load = bus[:, 2]
        total_load = bus_load.sum() if bus_load.sum() > 0 else len(self.bus_idx)

        # 分配负荷到各节点
        self.Pd = pd.DataFrame(index=self.bus_idx, columns=range(self.T))
        for t in range(self.T):
            for b, bus_idx in enumerate(self.bus_idx):
                ratio = bus_load[b] / total_load if bus_load.sum() > 0 else 1.0 / len(self.bus_idx)
                self.Pd.loc[bus_idx, t] = ratio * load_data[t]

        self.participant_demand = self.Pd * 0.09

        # 分配可再生能源
        self.solar_output = pd.DataFrame(0.0, index=self.bus_idx, columns=range(self.T))
        self.wind_output = pd.DataFrame(0.0, index=self.bus_idx, columns=range(self.T))
        solar_buses = [str(min(i, len(self.bus_idx) - 1)) for i in [5, 10, 15, 20, 25]]
        solar_buses = [b for b in solar_buses if int(b) < len(self.bus_idx)]

        wind_buses = [str(min(i, len(self.bus_idx) - 1)) for i in [7, 12, 17, 22, 27]]
        wind_buses = [b for b in wind_buses if int(b) < len(self.bus_idx)]

        for t in range(self.T):
            # 光伏风电均分到5个节点
            solar_per_bus = solar_data[t] / len(solar_buses)
            for solar_bus in solar_buses:
                self.solar_output.loc[solar_bus, t] = solar_per_bus
            wind_per_bus = wind_data[t] / len(wind_buses)
            for wind_bus in wind_buses:
                self.wind_output.loc[wind_bus, t] = wind_per_bus

        # 储能参数
        self.storage_eff = 0.95
        self.storage_duration = 2
        self.storage_opex = 0.5
        self.storage_buses = [str(min(i, len(self.bus_idx) - 1)) for i in [3, 8, 13, 18, 23]]
        self.storage_buses = [b for b in self.storage_buses if int(b) < len(self.bus_idx)]
        print(f"\n储能配置:{len(self.storage_buses)}个储能节点 - 节点{self.storage_buses}")
        # CFE诊断
        total_participant = self.participant_demand.sum().sum()
        total_renewable = self.solar_output.sum().sum() + self.wind_output.sum().sum()
        print(
            f"\nCFE市场配置:参与者需求(9%): {total_participant:.1f} MW, 可再生能源: {total_renewable:.1f} MW, 比例: {total_renewable / total_participant * 100:.1f}%")
        if total_renewable < total_participant * 0.95:
            shortage = total_participant - total_renewable
            print(
                f"  警告:可再生能源略有不足!缺口 {shortage:.1f} MW ({shortage / total_participant * 100:.1f}%), 储能可补充 ~{min(50 * 39 * 2, shortage):.0f} MW")

    def create_model(self, scenario='volumetric', matching_target=0.90):
        self.scenario = scenario
        m = ConcreteModel(name=scenario)

        # 定义集合和变量
        m.T = RangeSet(0, self.T - 1)
        m.G = Set(initialize=self.gen_idx)
        m.B = Set(initialize=self.bus_idx)
        m.E = Set(initialize=self.branch_num_idx)
        m.S = Set(initialize=self.storage_buses)
        m.P_gen = Var(m.G, m.T, bounds=(0, None))
        m.P_charge = Var(m.S, m.T, bounds=(0, None))
        m.P_discharge = Var(m.S, m.T, bounds=(0, None))
        m.E_storage = Var(m.S, m.T, bounds=(0, None))
        m.P_Cap = Param(initialize=300)
        m.Pf = Var(self.branch_idx, m.T, bounds=(None, None))
        m.E_storageinitial = Param(initialize=300)
        # 辅助函数
        BusGen = lambda g, b: 1 if g[1:] == b else 0
        PTDF = lambda n, b: self.PTDF_matrix[b][self.branch_idx[n]]

        # 目标函数
        m.obj = Objective(rule=lambda m: sum(self.Cost.at[g, 'Cost'] * m.P_gen[g, t] for g in m.G for t in m.T) + sum(
            self.storage_opex * (m.P_charge[b, t] + m.P_discharge[b, t]) for b in m.S for t in m.T), sense=minimize)

        # 约束
        m.power_balance = Constraint(m.T, rule=lambda m, t:
        sum(m.P_gen[g, t] for g in m.G) == sum(
            self.Pd.loc[b, t] - self.solar_output.loc[b, t] - self.wind_output.loc[b, t] for b in m.B) + sum(
            m.P_charge[s, t] - m.P_discharge[s, t] for s in m.S))
        m.dc_flow = Constraint(m.E, m.T, rule=lambda m, n, t: m.Pf[self.branch_idx[n], t] == sum(
            (sum(m.P_gen[g, t] * BusGen(g, b) for g in m.G)
             - self.Pd.loc[b, t] + self.solar_output.loc[b, t] + self.wind_output.loc[b, t]) * PTDF(n, b) for b in
            m.B) + sum((m.P_discharge[s, t] - m.P_charge[s, t]) * PTDF(n, s) for s in m.S))
        m.branch_upper = Constraint(m.E, m.T, rule=lambda m, n, t: m.Pf[self.branch_idx[n], t] <= self.Pf_max.at[
            self.branch_idx[n], 'Pf_max'])
        m.branch_lower = Constraint(m.E, m.T, rule=lambda m, n, t: m.Pf[self.branch_idx[n], t] >= -self.Pf_max.at[
            self.branch_idx[n], 'Pf_max'])
        m.gen_min = Constraint(m.G, m.T, rule=lambda m, g, t: m.P_gen[g, t] >= self.Pg_min.at[g, 'Pg_min'])
        m.gen_max = Constraint(m.G, m.T, rule=lambda m, g, t: m.P_gen[g, t] <= self.Pg_max.at[g, 'Pg_max'])

        m.storage_dynamic = Constraint(
            m.S, m.T,
            rule=lambda m, b, t:
            m.E_storage[b, t] ==
            (m.E_storageinitial if t == 0 else m.E_storage[b, t - 1]) +
            self.storage_eff * m.P_charge[b, t] -
            m.P_discharge[b, t] / self.storage_eff
        )

        m.storage_limit = Constraint(m.S, m.T,
                                     rule=lambda m, b, t: m.E_storage[b, t] <= m.P_Cap * self.storage_duration)
        m.charge_limit = Constraint(m.S, m.T, rule=lambda m, b, t: m.P_charge[b, t] <= m.P_Cap)
        m.discharge_limit = Constraint(m.S, m.T, rule=lambda m, b, t: m.P_discharge[b, t] <= m.P_Cap)
        m.storage_cyclic = Constraint(m.S, rule=lambda m, b: m.E_storageinitial == m.E_storage[b, self.T - 1])

        # CFE匹配约束
        if scenario == 'volumetric':
            m.matching = Constraint(rule=lambda m: sum(
                self.solar_output.loc[b, t] + self.wind_output.loc[b, t] for b in m.B for t in m.T) + sum(
                m.P_discharge[s, t] - m.P_charge[s, t]
                for s in m.S for t in m.T) >= sum(self.participant_demand.loc[b, t] for b in m.B for t in m.T))
        elif scenario == 'hourly':
            m.matching = Constraint(m.T, rule=lambda m, t: sum(
                self.solar_output.loc[b, t] + self.wind_output.loc[b, t] for b in m.B) + sum(
                m.P_discharge[s, t] - m.P_charge[s, t]
                for s in m.S) >= sum(self.participant_demand.loc[b, t] for b in m.B) * matching_target)

        self.model = m

    def solve(self):
        solver = SolverFactory(self.solver)
        solver.options['OutputFlag'] = 0
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.results = solver.solve(self.model, tee=False)
        if self.results.solver.termination_condition != TerminationCondition.optimal:
            print(f"⚠ 求解失败: {self.results.solver.termination_condition}")
            return None
        return self.extract_results()

    def extract_results(self):
        m = self.model
        total_charge = [sum(value(m.P_charge[b, t]) for b in m.S) for t in m.T]
        total_discharge = [sum(value(m.P_discharge[b, t]) for b in m.S) for t in m.T]
        total_storage = [value(m.E_storageinitial) * 5] + [sum(value(m.E_storage[b, t]) for b in m.S) for t in m.T]

        nodal_storage = {}
        for s in m.S:
            nodal_storage[s] = [value(m.E_storageinitial)] + [value(m.E_storage[s, t]) for t in m.T]

        nodal_LMP = {}
        avg_LMP = []

        # 获取CFE匹配约束的对偶变量
        if self.scenario == 'volumetric':
            # Volumetric场景:单个约束,所有时段共享一个拉格朗日乘子
            mu_matching = m.dual[m.matching]


        # participant_demand占总负荷的比例
        participant_ratio = 0.09

        for t in m.T:
            # 1. 功率平衡约束的拉格朗日乘子
            lambda_pb = m.dual[m.power_balance[t]]

            # 2. 获取CFE匹配约束的拉格朗日乘子
            if self.scenario == 'volumetric':
                # Volumetric:所有时段使用相同的mu
                mu_t = mu_matching
            elif self.scenario == 'hourly':
                # Hourly:每个时段有独立的mu
                mu_t = m.dual[m.matching[t]]
            else:
                mu_t = 0.0


            lmp_values = {}
            for b in m.B:
                # 3. 传输约束的拉格朗日乘子(堵塞成本)
                congestion_component = sum(
                    (m.dual[m.branch_lower[n, t]] - m.dual[m.branch_upper[n, t]]) *
                    self.PTDF_matrix[b][self.branch_idx[n]]
                    for n in self.branch_num_idx
                )

                # 4. CFE匹配约束的贡献
                # 当节点b的负荷增加1MW时,participant_demand增加0.09MW
                # 这使得CFE匹配约束的RHS增加,影响系统成本
                matching_component = mu_t * participant_ratio

                # 5. 完整的LMP = 能量成分 + 堵塞成分 + CFE匹配成分
                lmp = lambda_pb + congestion_component + matching_component
                lmp_values[b] = lmp


                # 存储节点LMP
                if b not in nodal_LMP:
                    nodal_LMP[b] = []
                nodal_LMP[b].append(lmp)

            # 按节点负荷加权平均计算avg_LMP
            total_load_t = sum(self.Pd.loc[b, t] for b in m.B)
            weighted_lmp = sum(lmp_values[b] * self.Pd.loc[b, t] for b in m.B) / total_load_t
            avg_LMP.append(weighted_lmp)



        # 计算储能收益
        revenue = sum(
            sum((- self.storage_opex - nodal_LMP[b][t]) * value(m.P_charge[b, t]) for b in m.S) for t in m.T) + \
                  sum(sum((nodal_LMP[b][t] - self.storage_opex) * value(m.P_discharge[b, t]) for b in m.S) for t in m.T)

        return {
            'scenario': self.scenario,
            'cost': value(m.obj),
            'storage_cap': sum(value(m.P_Cap) for b in m.S),
            'P_charge': total_charge,
            'P_discharge': total_discharge,
            'LMP': avg_LMP,
            'nodal_LMP': nodal_LMP,
            'emissions': sum(value(m.P_gen[g, t]) * self.GCI.at[g, 'GCI'] for g in m.G for t in m.T),
            'revenue': revenue,
            'E_storage': total_storage,
            'nodal_storage': nodal_storage
        }

def validate_cost_timeseries(market, scenario='volumetric', delta=1e-4):

    # 备份原始 Pd 和 participant_demand
    Pd_backup = market.Pd.copy()
    part_backup = market.participant_demand.copy()

    T = market.T
    buses = [4]
    participant_ratio = 0.09


    market.create_model(scenario)
    base = market.solve()
    base_cost = base['cost']
    base_nodal_LMP = base['nodal_LMP']

    print(f"\n基线场景 = {scenario}, 基线总成本 C_base = {base_cost:.4f}")

    # 存放每个时段的成本差
    cost_diff_bus = {}    # ΔC(t) = C_pert(t) - C_base
    # ------- 2) 对每个时段 t 做一次系统扰动 --------
    for bus in buses:
        cost_diff = []

        for t in range(T):
        # 每个时段从基线恢复一次
            market.Pd = Pd_backup.copy()
            market.participant_demand = part_backup.copy()
            market.Pd.loc[str(bus), t] += delta
            market.participant_demand.loc[str(bus), t] += participant_ratio * delta

        # 重新建模并求解
            market.create_model(scenario)
            pert = market.solve()
            pert_cost = pert['cost']



            diff_t = (pert_cost - base_cost)/delta
            cost_diff.append(diff_t)

        cost_diff_bus[bus] = cost_diff


    # ------- 3) 恢复原始数据 --------
    market.Pd = Pd_backup
    market.participant_demand = part_backup

    hours = np.arange(T)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, bus in enumerate(buses):
        ax = axes[idx]
        ax.plot(hours, cost_diff_bus[bus], 'b-o',linewidth=2,markersize=5)
        ax.plot(hours, base_nodal_LMP[str(bus)], 'r-s',linewidth=2,markersize=5)
        ax.set_xlabel('时段 t (h)', fontsize=10)
        ax.set_ylabel('ΔCost ($)', fontsize=10)
        ax.set_title(f'Bus {bus}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'各节点成本增量时序图 (scenario={scenario}, ΔLoad={delta} MW)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'cost_diff_timeseries_separate_{scenario}.png', dpi=300)
    print(f"各节点单独成本增量图已保存为 cost_diff_timeseries_separate_{scenario}.png")

    plt.close('all')

    return {
        "base_cost": base_cost,
        "cost_diff_bus": cost_diff_bus,
        "base_nodal_LMP": base_nodal_LMP,
    }


def plot_results(vol, hourly, market, target_date):
    hours = np.arange(24)
    hoursE = np.arange(25)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # LMP对比
    axes[0, 0].plot(hours, vol['LMP'], 'b-o', label='Volumetric', linewidth=2.5)
    axes[0, 0].plot(hours, hourly['LMP'], 'r-s', label='Hourly (98%)', linewidth=2.5)
    axes[0, 0].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 0].set_ylabel('平均LMP ($/MWh)', fontsize=12)
    axes[0, 0].set_title(f'边际电价对比 ({target_date})', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 储能充放电
    net_vol = [vol['P_discharge'][t] - vol['P_charge'][t] for t in range(24)]
    net_hourly = [hourly['P_discharge'][t] - hourly['P_charge'][t] for t in range(24)]
    axes[0, 1].bar(hours - 0.2, net_vol, width=0.35, alpha=0.4, color='blue')
    axes[0, 1].bar(hours + 0.2, net_hourly, width=0.35, alpha=0.4, color='red')
    axes[0, 1].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 1].set_ylabel('功率 (MW)', fontsize=12)
    axes[0, 1].set_title('储能充放电', fontsize=13, fontweight='bold')

    # 5个储能节点的储能状态变化(Volumetric场景)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    storage_nodes = sorted(vol['nodal_storage'].keys(), key=lambda x: int(x))

    for idx, node in enumerate(storage_nodes):
        axes[0, 2].plot(hoursE, vol['nodal_storage'][node],
                        marker='o', linewidth=2.0, markersize=4,
                        label=f'节点{node}', color=colors[idx])

    axes[0, 2].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 2].set_ylabel('储能状态 (MWh)', fontsize=12)
    axes[0, 2].set_title('储能节点状态变化 (Volumetric)', fontsize=13, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 5个储能节点的储能状态变化(Hourly场景)
    storage_nodes = sorted(hourly['nodal_storage'].keys(), key=lambda x: int(x))

    for idx, node in enumerate(storage_nodes):
        axes[1, 0].plot(hoursE, hourly['nodal_storage'][node],
                        marker='o', linewidth=2.0, markersize=4,
                        label=f'节点{node}', color=colors[idx])

    axes[1, 0].set_xlabel('时段 (h)', fontsize=12)
    axes[1, 0].set_ylabel('储能状态 (MWh)', fontsize=12)
    axes[1, 0].set_title('储能节点状态变化 (Hourly)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 关键指标
    metrics = ['成本', '储能充放电', '储能收益', '排放']
    vol_vals = [vol['cost'] / 1000, np.sum(vol['P_charge'] + vol['P_discharge']), vol['revenue'],
                vol['emissions'] / 100]
    hourly_vals = [hourly['cost'] / 1000, np.sum(hourly['P_charge'] + hourly['P_discharge']), hourly['revenue'],
                   hourly['emissions'] / 100]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width / 2, vol_vals, width, label='Vol', alpha=0.7)
    axes[1, 1].bar(x + width / 2, hourly_vals, width, label='Hourly', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, fontsize=11)
    axes[1, 1].set_title('关键指标', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 负荷和新能源变化情况
    total_load_24h = market.participant_demand.sum(axis=0).values
    total_solar_24h = market.solar_output.sum(axis=0).values
    total_wind_24h = market.wind_output.sum(axis=0).values
    total_renewable_24h = total_solar_24h + total_wind_24h
    axes[1, 2].plot(hours, total_load_24h, 'k-o', label='load', linewidth=2.5)
    axes[1, 2].plot(hours, total_renewable_24h, 'g-s', label='renewable', linewidth=2.5)
    axes[1, 2].set_xlabel('时段 (h)', fontsize=12)
    axes[1, 2].set_ylabel('容量 (MWh)', fontsize=12)
    axes[1, 2].set_title(f'负荷和新能源变化情况 ({target_date})', fontsize=13, fontweight='bold')
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lmp_results.png', dpi=300, bbox_inches='tight')

    print("\n生成各节点LMP折线图...")

    nodal_lmp_vol = vol['nodal_LMP']
    nodes = sorted(nodal_lmp_vol.keys(), key=lambda x: int(x))

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('nodal_lmp_lines.pdf') as pdf:
        for node in nodes:
            fig_node = plt.figure(figsize=(8, 4.5))
            plt.plot(hours, nodal_lmp_vol[node], '-o', linewidth=2.0, markersize=4, label='Volumetric')
            plt.plot(hours, hourly['nodal_LMP'][node], '-s', linewidth=2.0, markersize=4, label='Hourly (90%)')
            plt.xlabel('时段 (h)', fontsize=12)
            plt.ylabel('LMP ($/MWh)', fontsize=12)
            plt.title(f'节点{node} - LMP时序({target_date})', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            pdf.savefig(fig_node, bbox_inches='tight')
            plt.close(fig_node)

    print("各节点LMP折线图已保存为 nodal_lmp_lines.pdf")


if __name__ == "__main__":
    print("多节点CFE市场: Volumetric vs Hourly - 修正LMP计算版本")

    # 配置
    data_dir = r'C:\Users\86178\PycharmProjects\pythonProject\24 7 LMP'
    load_file = os.path.join(data_dir, 'BA_load_actuals_2017.csv')
    solar_file = os.path.join(data_dir, 'BA_solar_actuals_Existing_2017.csv')
    wind_file = os.path.join(data_dir, 'BA_wind_actuals_Existing_2017 - add.csv')
    target_date = '2017-06-27'
    critical_lines = [3, 7, 11, 15, 19, 23, 27, 31]
    congestion_factor = 0.65

    # 加载数据
    load_24h, solar_24h, wind_24h = load_real_data(load_file, solar_file, wind_file, target_date, auto_scale=True,
                                                   target_capacity=6195)

    net = pn.case39()
    market = CFEMarket(net, load_24h, solar_24h, wind_24h, solver='gurobi', congestion_factor=congestion_factor,
                       congestion_lines=critical_lines)

    # 求解Volumetric
    print("\n[1/2] 求解Volumetric...")
    market.create_model('volumetric')
    vol = market.solve()
    if vol:
        print(f"✓ 成本: ${vol['cost']:.2f}, 储能: {vol['storage_cap']:.1f} MW, 排放: {vol['emissions']:.1f} tCO2")
    else:
        print("求解失败")
        exit()

    # 求解Hourly
    print("\n[2/2] 求解Hourly (90%)...")
    market.create_model('hourly', matching_target=0.90)
    hourly = market.solve()
    if hourly:
        print(
            f"✓ 成本: ${hourly['cost']:.2f}, 储能: {hourly['storage_cap']:.1f} MW, 排放: {hourly['emissions']:.1f} tCO2")
    else:
        print("求解失败")
        exit()

    # 生成图表
    plot_results(vol, hourly, market, target_date)
    print(f"图片已保存为 lmp_results.png")

    #  Volumetric 场景
    validate_cost_timeseries(market, scenario='volumetric', delta=1e-4)

    #  Hourly 场景：
    validate_cost_timeseries(market, scenario='hourly', delta=1e-4)
