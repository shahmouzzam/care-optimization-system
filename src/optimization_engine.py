"""
Optimization Engine for Care Visit Scheduling
Solves staff assignment and routing problems with multiple constraints
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pulp import *
import itertools


class CareScheduleOptimizer:
    """Optimize staff schedules for care visits using Mixed Integer Linear Programming"""
    
    def __init__(self, visits_df, staff_df, patients_df):
        self.visits = visits_df.copy()
        self.staff = staff_df.copy()
        self.patients = patients_df.copy()
        
        # Optimization parameters
        self.travel_speed_kmh = 40  # Average travel speed
        self.max_shift_hours = 10
        self.min_break_hours = 0.5
        self.overtime_threshold = 8
        
        self.problem = None
        self.assignments = {}
        self.solution = None
        
    def calculate_travel_time(self, lat1, lon1, lat2, lon2):
        """Calculate travel time between two locations (simplified)"""
        # Haversine distance approximation
        distance_km = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # rough km per degree
        travel_time_hours = distance_km / self.travel_speed_kmh
        return travel_time_hours * 60  # Return in minutes
    
    def build_optimization_model(self, target_date):
        """Build the MILP optimization model"""
        
        print(f"\nBuilding optimization model for {target_date}...")
        
        # Filter visits for target date
        day_visits = self.visits[self.visits['date'] == target_date].copy()
        
        if len(day_visits) == 0:
            print("No visits scheduled for this date")
            return None
        
        print(f"Optimizing {len(day_visits)} visits with {len(self.staff)} staff members...")
        
        # Create the problem
        self.problem = LpProblem("Care_Visit_Scheduling", LpMinimize)
        
        # Decision variables: assign[visit_id, staff_id, time_slot]
        visits_list = day_visits['visit_id'].tolist()
        staff_list = self.staff['staff_id'].tolist()
        time_slots = list(range(7, 21))  # 7am to 9pm
        
        # Binary assignment variables
        assign = LpVariable.dicts("assign",
                                 ((v, s, t) for v in visits_list 
                                  for s in staff_list 
                                  for t in time_slots),
                                 cat='Binary')
        
        # Start time variables (continuous)
        start_time = LpVariable.dicts("start_time",
                                     visits_list,
                                     lowBound=7,
                                     upBound=21,
                                     cat='Continuous')
        
        # Travel time variables
        travel_time = LpVariable.dicts("travel",
                                      ((s, v1, v2) for s in staff_list
                                       for v1 in visits_list
                                       for v2 in visits_list if v1 != v2),
                                      lowBound=0,
                                      cat='Continuous')
        
        # === OBJECTIVE FUNCTION ===
        # Minimize: total cost (staff hours) + travel time + unmet demand penalty
        
        total_cost = 0
        
        # Staff cost
        for v in visits_list:
            visit_duration = day_visits[day_visits['visit_id'] == v]['estimated_duration'].values[0]
            for s in staff_list:
                staff_rate = self.staff[self.staff['staff_id'] == s]['hourly_rate'].values[0]
                for t in time_slots:
                    total_cost += assign[(v, s, t)] * staff_rate * (visit_duration / 60)
        
        # Travel cost (time penalty)
        travel_cost = lpSum([travel_time[k] * 0.5 for k in travel_time.keys()])
        
        # Penalty for using overtime
        overtime_cost = 0
        
        # Objective
        self.problem += total_cost + travel_cost + overtime_cost, "Total_Cost"
        
        # === CONSTRAINTS ===
        
        # 1. Each visit must be assigned exactly once
        for v in visits_list:
            self.problem += (
                lpSum([assign[(v, s, t)] for s in staff_list for t in time_slots]) == 1,
                f"visit_{v}_assigned_once"
            )
        
        # 2. Staff skill level requirements
        for v in visits_list:
            required_skill = day_visits[day_visits['visit_id'] == v]['required_skill_level'].values[0]
            skill_hierarchy = {'junior': 1, 'intermediate': 2, 'senior': 3, 'specialist': 4}
            required_level = skill_hierarchy[required_skill]
            
            for s in staff_list:
                staff_skill = self.staff[self.staff['staff_id'] == s]['skill_level'].values[0]
                staff_level = skill_hierarchy[staff_skill]
                
                # If staff is underqualified, cannot assign
                if staff_level < required_level:
                    for t in time_slots:
                        self.problem += assign[(v, s, t)] == 0, f"skill_req_{v}_{s}"
        
        # 3. Time window constraints
        for v in visits_list:
            visit_data = day_visits[day_visits['visit_id'] == v].iloc[0]
            tw_start = visit_data['time_window_start']
            tw_end = visit_data['time_window_end']
            
            for s in staff_list:
                for t in time_slots:
                    # If assigned, must be within time window
                    self.problem += (
                        start_time[v] >= tw_start - (1 - assign[(v, s, t)]) * 100,
                        f"tw_start_{v}_{s}_{t}"
                    )
                    self.problem += (
                        start_time[v] <= tw_end + (1 - assign[(v, s, t)]) * 100,
                        f"tw_end_{v}_{s}_{t}"
                    )
        
        # 4. Staff working hours limit
        for s in staff_list:
            max_hours = self.staff[self.staff['staff_id'] == s]['hours_per_week'].values[0] / 5
            
            total_hours = 0
            for v in visits_list:
                duration = day_visits[day_visits['visit_id'] == v]['estimated_duration'].values[0]
                for t in time_slots:
                    total_hours += assign[(v, s, t)] * (duration / 60)
            
            self.problem += (
                total_hours <= max_hours,
                f"staff_{s}_hours_limit"
            )
        
        # 5. No overlapping visits for same staff
        for s in staff_list:
            for v1, v2 in itertools.combinations(visits_list, 2):
                duration1 = day_visits[day_visits['visit_id'] == v1]['estimated_duration'].values[0] / 60
                duration2 = day_visits[day_visits['visit_id'] == v2]['estimated_duration'].values[0] / 60
                
                # If both assigned to same staff, must not overlap
                for t1 in time_slots:
                    for t2 in time_slots:
                        overlap_indicator = assign[(v1, s, t1)] + assign[(v2, s, t2)]
                        
                        # If both assigned (sum = 2), times must be separated
                        self.problem += (
                            start_time[v2] >= start_time[v1] + duration1 - (2 - overlap_indicator) * 100,
                            f"no_overlap_{s}_{v1}_{v2}_{t1}_{t2}_a"
                        )
                        self.problem += (
                            start_time[v1] >= start_time[v2] + duration2 - (2 - overlap_indicator) * 100,
                            f"no_overlap_{s}_{v1}_{v2}_{t1}_{t2}_b"
                        )
        
        print("Model built successfully!")
        return self.problem
    
    def solve(self, time_limit=120):
        """Solve the optimization problem"""
        
        if self.problem is None:
            print("No optimization problem built!")
            return None
        
        print(f"\nSolving optimization problem (time limit: {time_limit}s)...")
        
        # Use CBC solver
        solver = PULP_CBC_CMD(timeLimit=time_limit, msg=1, gapRel=0.05)
        
        # Solve
        status = self.problem.solve(solver)
        
        print(f"\nOptimization Status: {LpStatus[status]}")
        
        if status == 1:  # Optimal or feasible solution found
            print(f"Objective Value: £{value(self.problem.objective):.2f}")
            return self.extract_solution()
        else:
            print("No feasible solution found!")
            return None
    
    def extract_solution(self):
        """Extract the solution from the optimization model"""
        
        solution = []
        
        for v in self.problem.variables():
            if v.name.startswith("assign_") and v.varValue > 0.5:
                # Parse variable name: assign_(visit_id, staff_id, time_slot)
                parts = v.name.replace("assign_(", "").replace(")", "").split(",_")
                visit_id = parts[0].strip("'")
                staff_id = parts[1].strip("'")
                time_slot = int(parts[2].strip("'"))
                
                # Get visit details
                visit = self.visits[self.visits['visit_id'] == visit_id].iloc[0]
                staff = self.staff[self.staff['staff_id'] == staff_id].iloc[0]
                
                solution.append({
                    'visit_id': visit_id,
                    'patient_id': visit['patient_id'],
                    'staff_id': staff_id,
                    'staff_name': staff['name'],
                    'date': visit['date'],
                    'scheduled_time': time_slot,
                    'duration': visit['estimated_duration'],
                    'visit_type': visit['visit_type'],
                    'staff_rate': staff['hourly_rate'],
                    'cost': staff['hourly_rate'] * visit['estimated_duration'] / 60
                })
        
        self.solution = pd.DataFrame(solution)
        return self.solution
    
    def calculate_metrics(self):
        """Calculate key performance metrics from the solution"""
        
        if self.solution is None:
            return None
        
        metrics = {
            'total_visits': len(self.solution),
            'total_cost': self.solution['cost'].sum(),
            'staff_utilization': {},
            'avg_cost_per_visit': self.solution['cost'].mean(),
            'total_duration_hours': self.solution['duration'].sum() / 60,
        }
        
        # Staff utilization
        for staff_id in self.solution['staff_id'].unique():
            staff_visits = self.solution[self.solution['staff_id'] == staff_id]
            total_time = staff_visits['duration'].sum() / 60
            max_hours = self.staff[self.staff['staff_id'] == staff_id]['hours_per_week'].values[0] / 5
            
            metrics['staff_utilization'][staff_id] = {
                'visits': len(staff_visits),
                'hours': total_time,
                'utilization_pct': (total_time / max_hours * 100) if max_hours > 0 else 0
            }
        
        return metrics


class GreedyScheduler:
    """Fast greedy heuristic for comparison (non-optimal but quick)"""
    
    def __init__(self, visits_df, staff_df, patients_df, matcher):
        self.visits = visits_df.copy()
        self.staff = staff_df.copy()
        self.patients = patients_df.copy()
        self.matcher = matcher
        
    def schedule_greedy(self, target_date):
        """Simple greedy scheduling algorithm"""
        
        day_visits = self.visits[self.visits['date'] == target_date].copy()
        
        # Sort visits by priority (highest first)
        day_visits = day_visits.sort_values('priority', ascending=False)
        
        # Track staff schedules
        staff_schedules = {s: [] for s in self.staff['staff_id']}
        assignments = []
        
        for _, visit in day_visits.iterrows():
            # Get eligible staff
            eligible = self.matcher.get_eligible_staff(visit, self.staff, self.patients)
            
            # Try to assign to best available staff
            assigned = False
            for staff_match in eligible:
                staff_id = staff_match['staff_id']
                
                # Check if staff has capacity
                current_hours = sum([v['duration'] for v in staff_schedules[staff_id]]) / 60
                max_hours = self.staff[self.staff['staff_id'] == staff_id]['hours_per_week'].values[0] / 5
                
                if current_hours + visit['estimated_duration']/60 <= max_hours:
                    # Assign visit
                    assignments.append({
                        'visit_id': visit['visit_id'],
                        'patient_id': visit['patient_id'],
                        'staff_id': staff_id,
                        'date': target_date,
                        'scheduled_time': visit['time_window_start'],
                        'duration': visit['estimated_duration'],
                        'visit_type': visit['visit_type'],
                        'cost': staff_match['hourly_rate'] * visit['estimated_duration'] / 60
                    })
                    
                    staff_schedules[staff_id].append({
                        'visit_id': visit['visit_id'],
                        'duration': visit['estimated_duration']
                    })
                    
                    assigned = True
                    break
            
            if not assigned:
                print(f"Warning: Could not assign visit {visit['visit_id']}")
        
        return pd.DataFrame(assignments)


if __name__ == "__main__":
    print("Optimization Engine Ready")
    print("Contains:")
    print("- CareScheduleOptimizer (MILP-based)")
    print("- GreedyScheduler (Heuristic)")
