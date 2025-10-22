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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CFEMarket:
    def __init__(self, net, solver='gurobi', verbose=False):
        self.net = net
        self.solver = solver
        self.verbose = verbose
        self.T = 24

        # 修正发电机电压限制
        for idx in self.net.gen.index:
            bus_idx = self.net.gen.at[idx, 'bus']
            bus_max_vm = self.net.bus.at[bus_idx, 'max_vm_pu']
            if self.net.gen.at[idx, 'vm_pu'] > bus_max_vm:
                self.net.gen.at[idx, 'vm_pu'] = bus_max_vm

        # 提取网络数据
        pp.rundcpp(self.net)
        ppc = pp.converter.to_ppc(self.net)

        baseMVA = ppc['baseMVA']
        gen = ppc['gen']
        bus = ppc['bus']
        branch = ppc['branch']

        self.gen_idx = ['G' + str(int(gen[g, 0])) for g in range(gen.shape[0])]
        self.bus_idx = [str(b) for b in range(bus.shape[0])]
        self.branch_idx = [(str(int(branch[e, 0])), str(int(branch[e, 1])))
                           for e in range(branch.shape[0])]
        self.branch_num_idx = range(branch.shape[0])

        ref_bus = np.where(bus[:, BUS_TYPE] == REF)[0][0]
        self.PTDF_matrix = pd.DataFrame(
            makePTDF(baseMVA, bus, branch, slack=ref_bus, result_side=0,
                     using_sparse_solver=True, branch_id=None, reduced=False),
            index=self.branch_idx, columns=self.bus_idx
        )

        # 网络参数
        self.Pg_max = pd.DataFrame(gen[:, 8], index=self.gen_idx, columns=['Pg_max'])
        self.Pg_min = pd.DataFrame(gen[:, 9], index=self.gen_idx, columns=['Pg_min'])
        self.Pf_max = pd.DataFrame(branch[:, 5], index=self.branch_idx, columns=['Pf_max'])

        np.random.seed(42)
        self.Cost = pd.DataFrame(np.random.uniform(30, 50, len(self.gen_idx)),
                                 index=self.gen_idx, columns=['Cost'])
        self.GCI = pd.DataFrame(np.random.uniform(0.4, 0.95, len(self.gen_idx)),
                                index=self.gen_idx, columns=['GCI'])

        # 初始化时序数据
        self._init_temporal_data(bus)

    def _init_temporal_data(self, bus):
        """初始化24小时数据"""
        demand_profile = np.array([600, 550, 530, 520, 510, 550, 650, 800, 950, 1100,
                                   1150, 1200, 1250, 1200, 1150, 1100, 1150, 1250, 1350, 1400,
                                   1300, 1150, 950, 750])

        # 按节点负荷比例分配
        bus_load = bus[:, 2]
        total_load = bus_load.sum() if bus_load.sum() > 0 else len(self.bus_idx)

        self.Pd = pd.DataFrame(index=self.bus_idx, columns=range(self.T))
        for t in range(self.T):
            for b, bus_idx in enumerate(self.bus_idx):
                ratio = bus_load[b] / total_load if bus_load.sum() > 0 else 1.0 / len(self.bus_idx)
                self.Pd.loc[bus_idx, t] = ratio * demand_profile[t]

        self.participant_demand = self.Pd * 0.30

        # 可再生能源（分配到特定节点）
        solar_profile = np.array([0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.45, 0.7, 0.85, 0.95,
                                  1.0, 0.95, 0.85, 0.7, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0])
        wind_profile = np.array([0.6, 0.65, 0.7, 0.7, 0.65, 0.6, 0.5, 0.4, 0.35, 0.3,
                                 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                                 0.75, 0.7, 0.65, 0.6])

        self.solar_output = pd.DataFrame(0.0, index=self.bus_idx, columns=range(self.T))
        self.wind_output = pd.DataFrame(0.0, index=self.bus_idx, columns=range(self.T))

        solar_bus = str(min(5, len(self.bus_idx) - 1))
        wind_bus = str(min(10, len(self.bus_idx) - 1))

        for t in range(self.T):
            self.solar_output.loc[solar_bus, t] = solar_profile[t] * 400
            self.wind_output.loc[wind_bus, t] = wind_profile[t] * 300

        # 储能参数
        self.storage_eff = 0.95
        self.storage_duration = 2
        self.storage_capex = 150 / 365 / 24
        self.storage_opex = 2.0

    def create_model(self, scenario='volumetric', matching_target=0.90):
        self.scenario = scenario
        self.matching_target = matching_target
        m = ConcreteModel(name=scenario)

        m.T = RangeSet(0, self.T - 1)
        m.G = Set(initialize=self.gen_idx)
        m.B = Set(initialize=self.bus_idx)
        m.E = Set(initialize=self.branch_num_idx)

        m.P_gen = Var(m.G, m.T, bounds=(0, None))
        m.P_charge = Var(m.B, m.T, bounds=(0, None))
        m.P_discharge = Var(m.B, m.T, bounds=(0, None))
        m.E_storage = Var(m.B, m.T, bounds=(0, None))
        m.P_Cap = Param(initialize=50)
        m.Pf = Var(self.branch_idx, m.T, bounds=(None, None))

        # 辅助函数
        def BusGen(g, b):
            return 1 if g[1:] == b else 0

        def PTDF(n, b):
            return self.PTDF_matrix[b][self.branch_idx[n]]

        # 目标函数
        def obj_rule(m):
            gen_cost = sum(self.Cost.at[g, 'Cost'] * m.P_gen[g, t]
                           for g in m.G for t in m.T)
            # storage_capex = sum(self.storage_capex * m.P_Cap[b] for b in m.B)
            storage_opex = sum(self.storage_opex * (m.P_charge[b, t] + m.P_discharge[b, t])
                               for b in m.B for t in m.T)
            return gen_cost + storage_opex

        m.obj = Objective(rule=obj_rule, sense=minimize)

        # 功率平衡
        m.power_balance = Constraint(m.T, rule=lambda m, t:
        sum(m.P_gen[g, t] for g in m.G) == sum(
            self.Pd.loc[b, t] + m.P_charge[b, t] - m.P_discharge[b, t] -
            self.solar_output.loc[b, t] - self.wind_output.loc[b, t]
            for b in m.B))

        # DC潮流
        def dc_flow_rule(m, n, t):
            e = self.branch_idx[n]
            return m.Pf[e, t] == sum(
                (sum(m.P_gen[g, t] * BusGen(g, b) for g in m.G) -
                 self.Pd.loc[b, t] - m.P_charge[b, t] + m.P_discharge[b, t] +
                 self.solar_output.loc[b, t] + self.wind_output.loc[b, t]) * PTDF(n, b)
                for b in m.B)

        m.dc_flow = Constraint(m.E, m.T, rule=dc_flow_rule)

        # 支路容量
        def branch_upper_rule(m, n, t):
            e = self.branch_idx[n]
            return m.Pf[e, t] <= self.Pf_max.at[e, 'Pf_max']

        def branch_lower_rule(m, n, t):
            e = self.branch_idx[n]
            return -self.Pf_max.at[e, 'Pf_max'] <= m.Pf[e, t]

        m.branch_upper = Constraint(m.E, m.T, rule=branch_upper_rule)
        m.branch_lower = Constraint(m.E, m.T, rule=branch_lower_rule)

        # 机组约束
        def gen_min_rule(m, g, t):
            return m.P_gen[g, t] >= self.Pg_min.at[g, 'Pg_min']

        def gen_max_rule(m, g, t):
            return m.P_gen[g, t] <= self.Pg_max.at[g, 'Pg_max']

        m.gen_min = Constraint(m.G, m.T, rule=gen_min_rule)
        m.gen_max = Constraint(m.G, m.T, rule=gen_max_rule)

        # 储能约束
        def storage_rule(m, b, t):
            E_prev = 0.5 * m.P_Cap * self.storage_duration if t == 0 else m.E_storage[b, t - 1]
            return m.E_storage[b, t] == E_prev + self.storage_eff * m.P_charge[b, t] - m.P_discharge[
                b, t] / self.storage_eff

        m.storage_dynamic = Constraint(m.B, m.T, rule=storage_rule)
        m.storage_limit = Constraint(m.B, m.T,
                                     rule=lambda m, b, t: m.E_storage[b, t] <= m.P_Cap * self.storage_duration)
        m.charge_limit = Constraint(m.B, m.T, rule=lambda m, b, t: m.P_charge[b, t] <= m.P_Cap)
        m.discharge_limit = Constraint(m.B, m.T, rule=lambda m, b, t: m.P_discharge[b, t] <= m.P_Cap)
        m.storage_cyclic = Constraint(m.B, rule=lambda m, b: m.E_storage[b, 0] == m.E_storage[b, self.T - 1])

        # CFE匹配约束
        if scenario == 'volumetric':
            m.matching = Constraint(rule=lambda m:
            sum(self.solar_output.loc[b, t] + self.wind_output.loc[b, t] +
                m.P_discharge[b, t] - m.P_charge[b, t] for b in m.B for t in m.T)
            >= sum(self.participant_demand.loc[b, t] for b in m.B for t in m.T))
        elif scenario == 'hourly':
            m.matching = Constraint(m.T, rule=lambda m, t:
            sum(self.solar_output.loc[b, t] + self.wind_output.loc[b, t] +
                m.P_discharge[b, t] - m.P_charge[b, t] for b in m.B)
            >= sum(self.participant_demand.loc[b, t] for b in m.B) * matching_target)

        self.model = m

    def solve(self):
        solver = SolverFactory(self.solver)
        solver.options['OutputFlag'] = 0
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.results = solver.solve(self.model, tee=False)

        if self.results.solver.termination_condition != TerminationCondition.optimal:
            if self.verbose:
                print(f"求解失败: {self.results.solver.termination_condition}")
            return None

        return self.extract_results()

    def extract_results(self):
        m = self.model

        # 聚合结果
        total_charge = [sum(value(m.P_charge[b, t]) for b in m.B) for t in m.T]
        total_discharge = [sum(value(m.P_discharge[b, t]) for b in m.B) for t in m.T]

        # 计算平均LMP
        avg_LMP = []
        for t in m.T:
            lambda_pb = m.dual[m.power_balance[t]]
            lmp_values = []

            for b in m.B:
                congestion = sum(
                    (m.dual[m.branch_lower[n, t]] - m.dual[m.branch_upper[n, t]]) *
                    self.PTDF_matrix[b][self.branch_idx[n]]
                    for n in self.branch_num_idx
                )
                lmp_values.append(lambda_pb + congestion)

            avg_LMP.append(np.mean(lmp_values))

        return {
            'scenario': self.scenario,
            'cost': value(m.obj),
            'storage_cap': sum(value(m.P_Cap) for b in m.B),
            'P_charge': total_charge,
            'P_discharge': total_discharge,
            'LMP': avg_LMP,
            'emissions': sum(value(m.P_gen[g, t]) * self.GCI.at[g, 'GCI']
                             for g in m.G for t in m.T)
        }


def plot_results(vol, hourly, market):
    hours = np.arange(24)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. LMP对比
    axes[0, 0].plot(hours, vol['LMP'], 'b-o', label='Volumetric', linewidth=2.5)
    axes[0, 0].plot(hours, hourly['LMP'], 'r-s', label='Hourly (90%)', linewidth=2.5)
    axes[0, 0].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 0].set_ylabel('平均LMP ($/MWh)', fontsize=12)
    axes[0, 0].set_title('边际电价对比', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 储能充电
    axes[0, 1].bar(hours - 0.2, vol['P_charge'], width=0.35,
                   label='Vol', alpha=0.7, color='blue')
    axes[0, 1].bar(hours + 0.2, hourly['P_charge'], width=0.35,
                   label='Hourly', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 1].set_ylabel('功率 (MW)', fontsize=12)
    axes[0, 1].set_title('储能充电', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. 储能放电
    axes[0, 2].bar(hours - 0.2, vol['P_discharge'], width=0.35,
                   label='Vol', alpha=0.7, color='blue')
    axes[0, 2].bar(hours + 0.2, hourly['P_discharge'], width=0.35,
                   label='Hourly', alpha=0.7, color='red')
    axes[0, 2].set_xlabel('时段 (h)', fontsize=12)
    axes[0, 2].set_ylabel('功率 (MW)', fontsize=12)
    axes[0, 2].set_title('储能放电', fontsize=13, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 4. 可再生能源
    total_solar = market.solar_output.sum(axis=0).values
    total_wind = market.wind_output.sum(axis=0).values
    total_demand = market.participant_demand.sum(axis=0).values

    axes[1, 0].fill_between(hours, 0, total_solar, label='光伏', alpha=0.6, color='yellow')
    axes[1, 0].fill_between(hours, total_solar, total_solar + total_wind,
                            label='风电', alpha=0.6, color='green')
    axes[1, 0].plot(hours, total_demand, 'k--', label='参与者需求', linewidth=2.5)
    axes[1, 0].set_xlabel('时段 (h)', fontsize=12)
    axes[1, 0].set_ylabel('功率 (MW)', fontsize=12)
    axes[1, 0].set_title('可再生能源', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 指标对比
    metrics = ['成本', '储能', '排放']
    vol_vals = [vol['cost'] / 1000, vol['storage_cap'], vol['emissions'] / 100]
    hourly_vals = [hourly['cost'] / 1000, hourly['storage_cap'], hourly['emissions'] / 100]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width / 2, vol_vals, width, label='Vol', alpha=0.7)
    axes[1, 1].bar(x + width / 2, hourly_vals, width, label='Hourly', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, fontsize=11)
    axes[1, 1].set_title('关键指标', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. LMP与储能
    axes[1, 2].plot(hours, vol['LMP'], 'b-', linewidth=2.5, alpha=0.7, label='Vol-LMP')
    axes[1, 2].plot(hours, hourly['LMP'], 'r-', linewidth=2.5, alpha=0.7, label='Hourly-LMP')
    ax2 = axes[1, 2].twinx()
    net_vol = [vol['P_discharge'][t] - vol['P_charge'][t] for t in range(24)]
    net_hourly = [hourly['P_discharge'][t] - hourly['P_charge'][t] for t in range(24)]
    ax2.bar(hours - 0.2, net_vol, width=0.35, alpha=0.4, color='blue')
    ax2.bar(hours + 0.2, net_hourly, width=0.35, alpha=0.4, color='red')
    axes[1, 2].set_xlabel('时段 (h)', fontsize=12)
    axes[1, 2].set_ylabel('LMP ($/MWh)', fontsize=12)
    ax2.set_ylabel('净放电 (MW)', fontsize=12)
    axes[1, 2].set_title('LMP与储能套利', fontsize=13, fontweight='bold')
    axes[1, 2].legend(loc='upper left', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multinode_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("多节点CFE市场: Volumetric vs Hourly")
    print("=" * 60)

    net = pn.case39()
    print(f"\n✓ 加载39节点系统")

    market = CFEMarket(net, solver='gurobi', verbose=False)

    print("\n[1/2] 求解Volumetric...")
    market.create_model('volumetric')
    vol = market.solve()
    if vol:
        print(f"✓ 成本: ${vol['cost']:.2f}, 储能: {vol['storage_cap']:.1f} MW, 排放: {vol['emissions']:.1f} tCO2")
    else:
        exit()

    print("\n[2/2] 求解Hourly (90%)...")
    market.create_model('hourly', matching_target=0.90)
    hourly = market.solve()
    if hourly:
        print(
            f"✓ 成本: ${hourly['cost']:.2f}, 储能: {hourly['storage_cap']:.1f} MW, 排放: {hourly['emissions']:.1f} tCO2")
    else:
        exit()

    print("\n生成对比图...")
    plot_results(vol, hourly, market)
    print("✓ 完成！")