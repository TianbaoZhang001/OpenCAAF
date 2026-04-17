import math
from typing import Dict, Any, Optional

class UAISolver:
    """
    Generalized Deterministic Mathematical Solver (UAI Tool).
    Returns EXACT physical boundaries. No implicit margins.
    """
    
    @staticmethod
    def solve_kinematic_velocity_from_distance(d: float, mu: float, g: float, f: float = 3.6) -> float:
        return f * math.sqrt(max(0, d * 2 * mu * g))
        
    @staticmethod
    def solve_kinematic_distance_from_velocity(v: float, mu: float, g: float, f: float = 3.6) -> float:
        return ((max(0, v) / f) ** 2) / (2 * mu * g)
        
    @staticmethod
    def solve_acceleration_from_velocities(v_start: float, v_end: float, t: float, f: float = 3.6) -> float:
        return (v_start - v_end) / (t * f)
        
    @staticmethod
    def solve_velocity_from_acceleration(v_start: float, a: float, t: float, f: float = 3.6) -> float:
        return v_start - (a * t * f)

    @classmethod
    def analyze_domain_paradox(cls, domain_id: str, state: Dict[str, Any]) -> str:
        if domain_id == "ad_degradation":
            try:
                v0 = float(state.get('vehicle_speed_kmph_t0', 120))
                mu = float(state.get('road_friction_mu', 0.4))
                g_val = float(state.get('g', 9.8))
                t = float(state.get('transition_window_seconds', 5))
                f = float(state.get('m_per_sec_to_km_per_h_factor', 3.6))
                p_limit = float(state.get('perception_range_limit', 30))
                a_max = float(state.get('max_deceleration_limit', 2.0))
                
                # EXACT Calculations
                v_fwd_boundary = cls.solve_kinematic_velocity_from_distance(p_limit, mu, g_val, f)
                req_a_boundary = cls.solve_acceleration_from_velocities(v0, v_fwd_boundary, t, f)
                
                v_rear_boundary = cls.solve_velocity_from_acceleration(v0, a_max, t, f)
                req_p_boundary = cls.solve_kinematic_distance_from_velocity(v_rear_boundary, mu, g_val, f)
                
                # STRATEGIC TARGETS (Applying 10% margin based on directionality)
                # 1. Deceleration Margin
                # If req_a > a_max (e.g. 3.6 > 2.0), we need more room, so ADD 10% -> 3.96
                # If req_a < a_max (e.g. 1.5 < 2.0), we are narrowing, so SUBTRACT 10% -> 1.35
                safe_a = req_a_boundary * 1.10 if req_a_boundary > a_max else req_a_boundary * 0.90
                
                # 2. Perception Margin
                # If req_p > p_limit (e.g. 69 > 30), we need more room, so ADD 10% -> 76
                # If req_p < p_limit (e.g. 20 < 30), we are narrowing, so SUBTRACT 10% -> 18
                safe_p = req_p_boundary * 1.10 if req_p_boundary > p_limit else req_p_boundary * 0.90

                return f"""[UAI TOOL: DETERMINISTIC MATH SOLVER]
- FORWARD COLLISION: To stop within {p_limit}m, velocity MUST be <= {v_fwd_boundary:.4f} km/h.
- REAR COLLISION: To deceleration at {a_max} m/s^2, velocity MUST be >= {v_rear_boundary:.4f} km/h.
- STRATEGIC BOUNDARIES:
    - [RELAX_DECELERATION]: To prioritize forward safety speed ({v_fwd_boundary:.2f} km/h), the REQUIRED max_deceleration_limit boundary is {req_a_boundary:.4f} m/s^2. Applying 10% safety margin, RECOMMENDED_TARGET is {safe_a:.4f} m/s^2.
    - [RELAX_PERCEPTION]: To prioritize rear safety deceleration ({a_max} m/s^2), the REQUIRED perception_range_limit boundary is {req_p_boundary:.4f} m. Applying 10% safety margin, RECOMMENDED_TARGET is {safe_p:.4f} m.
"""
            except Exception as e:
                return f"[UAI TOOL ERROR: {str(e)}]"
                
        if domain_id == "cloud_infra_sla":
            try:
                min_replicas = int(state.get('min_replicas_for_sla', 5))
                cost_per = float(state.get('cost_per_replica_usd', 450))
                budget = float(state.get('monthly_budget_usd', 1200))

                max_affordable = int(budget / cost_per)  # floor division
                gap = min_replicas - max_affordable

                return f"""[UAI TOOL: DETERMINISTIC MATH SOLVER]
- AVAILABILITY SLA: To achieve 99.99% uptime, replica_count MUST be >= {min_replicas}.
- COST BUDGET: At ${cost_per:.0f}/replica, budget ${budget:.0f} allows at most {max_affordable} replicas.
- GAP: Need {min_replicas} replicas but can only afford {max_affordable}. Shortfall = {gap} replicas.
- STRATEGIC BOUNDARIES:
    - [RELAX_SLA]: Lower availability target to 99.9% (three-nines), requiring only {max(max_affordable, 1)} replicas. Fits within budget.
    - [INCREASE_BUDGET]: To maintain {min_replicas} replicas, budget must be >= ${min_replicas * cost_per:.0f}/month (+${min_replicas * cost_per - budget:.0f}).
    - [REDUCE_COST]: Use spot/preemptible instances at ~${budget / min_replicas:.0f}/replica to fit {min_replicas} replicas in ${budget:.0f}.
"""
            except Exception as e:
                return f"[UAI TOOL ERROR: {str(e)}]"

        if domain_id == "pharma_flow_reactor":
            try:
                import math as _m
                A = float(state.get('A_factor', 2.5e8))
                Ea = float(state.get('Ea', 72000))
                R = float(state.get('R_gas', 8.314))
                alpha = float(state.get('alpha', 0.35))
                I_max = 0.02   # ICH Q3A impurity limit
                X_min = 0.95   # ICH Q6A conversion minimum
                tau_max = float(state.get('tau_max_s', 120))

                # C1: k*tau >= -ln(1 - X_min)
                kt_min = -_m.log(1 - X_min)
                # C2: alpha * k^2 * tau <= I_max  =>  k^2*tau <= I_max/alpha
                k2t_max = I_max / alpha
                # Combined: k <= k2t_max / kt_min
                k_upper = k2t_max / kt_min
                tau_lower = kt_min / k_upper  # minimum tau from C1+C2

                # Temperature at k boundary
                T_boundary = Ea / (R * _m.log(A / k_upper)) - 273.15

                gap = tau_lower - tau_max

                return f"""[UAI TOOL: DETERMINISTIC MATH SOLVER — Pharma Flow Reactor]
- CONVERSION (C1): k·τ >= {kt_min:.4f}  (for X >= {X_min})
- IMPURITY (C2): α·k²·τ <= {I_max}  (α={alpha})
- Combined C1+C2: k <= {k_upper:.5f} s⁻¹ → τ >= {tau_lower:.1f} s
- RESIDENCE TIME (C4): τ <= {tau_max:.0f} s
- DEADLOCK: τ >= {tau_lower:.1f} AND τ <= {tau_max:.0f} → gap = {gap:.1f} s. NO VALID τ EXISTS.
- BOUNDARY TEMPERATURE: k={k_upper:.5f} corresponds to T = {T_boundary:.1f}°C
- MINIMAL CONFLICT SET: {{CONVERSION_MINIMUM, IMPURITY_LIMIT, RESIDENCE_TIME_LIMIT}}
- STRATEGIC BOUNDARIES:
    - [RELAX_RESIDENCE_TIME]: Increase τ_max to >= {tau_lower:.0f} s. Allows solution at T ≈ {T_boundary:.1f}°C, τ ≈ {tau_lower:.0f} s.
    - [RELAX_IMPURITY]: Raise impurity limit from {I_max*100:.0f}% to >= {alpha * (kt_min/tau_max)**2 * tau_max * 100:.1f}%. Requires downstream purification.
    - [CHANGE_CHEMISTRY]: Reduce side-reaction coefficient α (catalyst optimization, R&D investment).
"""
            except Exception as e:
                return f"[UAI TOOL ERROR: {str(e)}]"

        return "[UAI TOOL: NO DETERMINISTIC SOLVER AVAILABLE FOR THIS DOMAIN]"
