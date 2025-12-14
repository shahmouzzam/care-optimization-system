"""
Visualization Module for Care Optimization Results
Creates charts, maps, and dashboards to present results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class OptimizationVisualizer:
    """Create visualizations for optimization results"""
    
    def __init__(self, solution_df=None):
        self.solution = solution_df
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C757D'
        }
    
    def plot_staff_utilization(self, metrics, save_path=None):
        """Visualize staff utilization rates"""
        
        staff_util = metrics['staff_utilization']
        
        staff_ids = list(staff_util.keys())
        utilization = [staff_util[s]['utilization_pct'] for s in staff_ids]
        hours = [staff_util[s]['hours'] for s in staff_ids]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Utilization rates
        colors = [self.colors['success'] if u >= 70 else 
                 self.colors['warning'] if u >= 50 else 
                 self.colors['danger'] for u in utilization]
        
        ax1.barh(staff_ids, utilization, color=colors, alpha=0.7)
        ax1.axvline(70, color='green', linestyle='--', alpha=0.5, label='Target (70%)')
        ax1.set_xlabel('Utilization (%)')
        ax1.set_title('Staff Utilization Rates', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_xlim(0, 100)
        
        # Working hours
        ax2.barh(staff_ids, hours, color=self.colors['primary'], alpha=0.7)
        ax2.set_xlabel('Hours')
        ax2.set_title('Total Working Hours', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Staff utilization chart saved to {save_path}")
        
        return fig
    
    def plot_schedule_timeline(self, solution_df, target_date, save_path=None):
        """Create a Gantt-chart style timeline of the schedule"""
        
        day_schedule = solution_df[solution_df['date'] == target_date].copy()
        
        if len(day_schedule) == 0:
            print("No visits for this date")
            return None
        
        # Sort by staff and time
        day_schedule = day_schedule.sort_values(['staff_id', 'scheduled_time'])
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(day_schedule['staff_id'].unique()) * 0.6)))
        
        # Create timeline
        staff_ids = sorted(day_schedule['staff_id'].unique())
        y_positions = {staff: i for i, staff in enumerate(staff_ids)}
        
        # Plot visits as bars
        for _, visit in day_schedule.iterrows():
            staff = visit['staff_id']
            start = visit['scheduled_time']
            duration = visit['duration'] / 60  # Convert to hours
            
            y_pos = y_positions[staff]
            
            # Color by visit type
            color = self.colors['primary']
            if 'medication' in visit['visit_type']:
                color = self.colors['danger']
            elif 'meal' in visit['visit_type']:
                color = self.colors['success']
            elif 'personal' in visit['visit_type']:
                color = self.colors['info']
            
            ax.barh(y_pos, duration, left=start, height=0.6, 
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add visit label
            if duration > 0.5:
                ax.text(start + duration/2, y_pos, visit['visit_id'], 
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(staff_ids)))
        ax.set_yticklabels(staff_ids)
        ax.set_xlabel('Time of Day', fontsize=12)
        ax.set_ylabel('Staff Member', fontsize=12)
        ax.set_title(f'Care Visit Schedule - {target_date}', fontsize=14, fontweight='bold')
        ax.set_xlim(7, 21)
        ax.grid(axis='x', alpha=0.3)
        
        # Add time labels
        ax.set_xticks(range(7, 22))
        ax.set_xticklabels([f'{h}:00' for h in range(7, 22)], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Schedule timeline saved to {save_path}")
        
        return fig
    
    def plot_cost_breakdown(self, solution_df, save_path=None):
        """Visualize cost breakdown by different dimensions"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cost by visit type
        cost_by_type = solution_df.groupby('visit_type')['cost'].sum().sort_values(ascending=False)
        ax1.bar(range(len(cost_by_type)), cost_by_type.values, color=self.colors['primary'], alpha=0.7)
        ax1.set_xticks(range(len(cost_by_type)))
        ax1.set_xticklabels(cost_by_type.index, rotation=45, ha='right')
        ax1.set_ylabel('Total Cost (£)')
        ax1.set_title('Cost by Visit Type', fontweight='bold')
        
        # 2. Cost by staff member
        cost_by_staff = solution_df.groupby('staff_id')['cost'].sum().sort_values(ascending=False).head(10)
        ax2.barh(range(len(cost_by_staff)), cost_by_staff.values, color=self.colors['secondary'], alpha=0.7)
        ax2.set_yticks(range(len(cost_by_staff)))
        ax2.set_yticklabels(cost_by_staff.index)
        ax2.set_xlabel('Total Cost (£)')
        ax2.set_title('Cost by Staff (Top 10)', fontweight='bold')
        
        # 3. Daily cost trend
        daily_cost = solution_df.groupby('date')['cost'].sum().sort_index()
        ax3.plot(range(len(daily_cost)), daily_cost.values, marker='o', 
                color=self.colors['success'], linewidth=2)
        ax3.set_xticks(range(len(daily_cost)))
        ax3.set_xticklabels([d.strftime('%m/%d') for d in daily_cost.index], rotation=45)
        ax3.set_ylabel('Daily Cost (£)')
        ax3.set_title('Daily Cost Trend', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Cost distribution histogram
        ax4.hist(solution_df['cost'], bins=20, color=self.colors['warning'], alpha=0.7, edgecolor='black')
        ax4.axvline(solution_df['cost'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: £{solution_df["cost"].mean():.2f}')
        ax4.set_xlabel('Visit Cost (£)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Visit Cost Distribution', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost breakdown chart saved to {save_path}")
        
        return fig
    
    def plot_patient_coverage(self, solution_df, patients_df, save_path=None):
        """Visualize patient coverage and visit distribution"""
        
        visits_per_patient = solution_df.groupby('patient_id').size()
        
        # Merge with patient acuity
        patient_stats = patients_df[['patient_id', 'acuity']].copy()
        patient_stats['visits_scheduled'] = patient_stats['patient_id'].map(visits_per_patient).fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Visits by patient acuity
        acuity_visits = patient_stats.groupby('acuity')['visits_scheduled'].sum()
        acuity_order = ['low', 'medium', 'high', 'critical']
        acuity_visits = acuity_visits.reindex(acuity_order)
        
        colors_acuity = [self.colors['success'], self.colors['info'], 
                        self.colors['warning'], self.colors['danger']]
        
        ax1.bar(range(len(acuity_visits)), acuity_visits.values, 
               color=colors_acuity, alpha=0.7)
        ax1.set_xticks(range(len(acuity_visits)))
        ax1.set_xticklabels(acuity_order, rotation=45)
        ax1.set_ylabel('Number of Visits')
        ax1.set_title('Visits by Patient Acuity Level', fontweight='bold')
        
        # 2. Visit distribution
        ax2.hist(patient_stats['visits_scheduled'], bins=15, 
                color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax2.axvline(patient_stats['visits_scheduled'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Mean: {patient_stats["visits_scheduled"].mean():.1f}')
        ax2.set_xlabel('Visits per Patient')
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Visit Distribution Across Patients', fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Patient coverage chart saved to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, metrics, solution_df, patients_df, save_path=None):
        """Create comprehensive summary dashboard"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Care Optimization Results Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Key metrics (top left)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metric_text = f"""
        OPTIMIZATION SUMMARY
        
        Total Visits: {metrics['total_visits']}
        Total Cost: £{metrics['total_cost']:.2f}
        Average Cost per Visit: £{metrics['avg_cost_per_visit']:.2f}
        Total Service Hours: {metrics['total_duration_hours']:.1f}
        
        Staff Utilization: {np.mean([s['utilization_pct'] for s in metrics['staff_utilization'].values()]):.1f}%
        """
        
        ax1.text(0.1, 0.5, metric_text, fontsize=12, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 2. Staff utilization
        ax2 = fig.add_subplot(gs[1, 0])
        staff_util = metrics['staff_utilization']
        staff_ids = list(staff_util.keys())[:10]
        utilization = [staff_util[s]['utilization_pct'] for s in staff_ids]
        
        ax2.barh(staff_ids, utilization, color=self.colors['primary'], alpha=0.7)
        ax2.set_xlabel('Utilization (%)')
        ax2.set_title('Staff Utilization (Top 10)', fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # 3. Daily cost
        ax3 = fig.add_subplot(gs[1, 1])
        daily_cost = solution_df.groupby('date')['cost'].sum()
        ax3.plot(range(len(daily_cost)), daily_cost.values, 
                marker='o', color=self.colors['success'], linewidth=2)
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Cost (£)')
        ax3.set_title('Daily Cost Trend', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Visit types
        ax4 = fig.add_subplot(gs[1, 2])
        visit_counts = solution_df['visit_type'].value_counts()
        ax4.pie(visit_counts.values, labels=visit_counts.index, autopct='%1.1f%%',
               colors=[self.colors[c] for c in ['primary', 'secondary', 'success', 'warning', 'danger']])
        ax4.set_title('Visit Type Distribution', fontweight='bold')
        
        # 5. Cost by acuity
        ax5 = fig.add_subplot(gs[2, 0])
        patient_costs = solution_df.merge(patients_df[['patient_id', 'acuity']], on='patient_id')
        acuity_cost = patient_costs.groupby('acuity')['cost'].sum()
        ax5.bar(range(len(acuity_cost)), acuity_cost.values, 
               color=self.colors['warning'], alpha=0.7)
        ax5.set_xticks(range(len(acuity_cost)))
        ax5.set_xticklabels(acuity_cost.index, rotation=45)
        ax5.set_ylabel('Total Cost (£)')
        ax5.set_title('Cost by Patient Acuity', fontweight='bold')
        
        # 6. Time distribution
        ax6 = fig.add_subplot(gs[2, 1])
        time_dist = solution_df['scheduled_time'].value_counts().sort_index()
        ax6.bar(time_dist.index, time_dist.values, 
               color=self.colors['info'], alpha=0.7)
        ax6.set_xlabel('Hour of Day')
        ax6.set_ylabel('Number of Visits')
        ax6.set_title('Visit Time Distribution', fontweight='bold')
        
        # 7. Coverage stats
        ax7 = fig.add_subplot(gs[2, 2])
        patients_served = solution_df['patient_id'].nunique()
        total_patients = len(patients_df)
        coverage = (patients_served / total_patients) * 100
        
        sizes = [patients_served, total_patients - patients_served]
        colors_pie = [self.colors['success'], self.colors['danger']]
        ax7.pie(sizes, labels=['Served', 'Not Served'], autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax7.set_title(f'Patient Coverage ({coverage:.1f}%)', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to {save_path}")
        
        return fig
    
    def export_results_summary(self, metrics, solution_df, filepath):
        """Export results summary as JSON"""
        
        # Convert date column to string for JSON serialization
        daily_breakdown = solution_df.groupby('date').agg({
            'visit_id': 'count',
            'cost': 'sum'
        })
        daily_breakdown.index = daily_breakdown.index.astype(str)
        
        summary = {
            'optimization_metrics': {
                'total_visits': metrics['total_visits'],
                'total_cost': float(metrics['total_cost']),
                'avg_cost_per_visit': float(metrics['avg_cost_per_visit']),
                'total_hours': float(metrics['total_duration_hours'])
            },
            'staff_utilization': {
                'average': float(np.mean([s['utilization_pct'] 
                                         for s in metrics['staff_utilization'].values()])),
                'by_staff': {k: float(v['utilization_pct']) 
                           for k, v in metrics['staff_utilization'].items()}
            },
            'daily_breakdown': daily_breakdown.to_dict(),
            'visit_type_distribution': solution_df['visit_type'].value_counts().to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results summary exported to {filepath}")


if __name__ == "__main__":
    print("Visualization Module Ready")
