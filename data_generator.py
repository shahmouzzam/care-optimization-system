"""
Data Generator for Care Resource Optimization
Generates realistic synthetic data for care facilities, patients, and staff
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class CareDataGenerator:
    """Generate realistic care facility data for optimization modeling"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Care visit types and durations
        self.visit_types = {
            'personal_care': {'base_duration': 30, 'variance': 10, 'frequency': 0.35},
            'medication': {'base_duration': 20, 'variance': 5, 'frequency': 0.25},
            'meal_prep': {'base_duration': 45, 'variance': 15, 'frequency': 0.20},
            'mobility_assistance': {'base_duration': 40, 'variance': 12, 'frequency': 0.15},
            'companionship': {'base_duration': 60, 'variance': 20, 'frequency': 0.05}
        }
        
        # Staff skill levels
        self.skill_levels = ['junior', 'intermediate', 'senior', 'specialist']
        
        # Patient acuity levels (complexity of care)
        self.acuity_levels = ['low', 'medium', 'high', 'critical']
        
    def generate_patients(self, n_patients=50):
        """Generate patient profiles with care requirements"""
        
        patients = []
        for i in range(n_patients):
            # Basic demographics
            age = np.random.randint(65, 95)
            gender = random.choice(['M', 'F'])
            
            # Care needs (increases with age)
            age_factor = (age - 65) / 30.0
            acuity = np.random.choice(
                self.acuity_levels,
                p=[0.3 - 0.15*age_factor, 0.35 - 0.1*age_factor, 
                   0.25 + 0.15*age_factor, 0.1 + 0.1*age_factor]
            )
            
            # Health conditions
            n_conditions = np.random.poisson(2) + 1
            conditions = random.sample([
                'diabetes', 'hypertension', 'arthritis', 'dementia',
                'heart_disease', 'copd', 'osteoporosis', 'parkinsons'
            ], min(n_conditions, 8))
            
            # Location (for routing)
            location = {
                'lat': 50.7192 + np.random.uniform(-0.05, 0.05),  # Bournemouth area
                'lon': -1.8808 + np.random.uniform(-0.05, 0.05)
            }
            
            # Visit requirements (visits per week)
            if acuity == 'critical':
                visits_per_week = np.random.randint(14, 21)
            elif acuity == 'high':
                visits_per_week = np.random.randint(7, 14)
            elif acuity == 'medium':
                visits_per_week = np.random.randint(4, 7)
            else:
                visits_per_week = np.random.randint(2, 4)
            
            patients.append({
                'patient_id': f'P{i+1:03d}',
                'age': age,
                'gender': gender,
                'acuity': acuity,
                'conditions': conditions,
                'lat': location['lat'],
                'lon': location['lon'],
                'visits_per_week': visits_per_week,
                'preferred_time': random.choice(['morning', 'afternoon', 'evening', 'flexible']),
                'mobility_issues': acuity in ['high', 'critical'] and random.random() > 0.3
            })
        
        return pd.DataFrame(patients)
    
    def generate_staff(self, n_staff=15):
        """Generate care staff profiles with skills and availability"""
        
        staff = []
        for i in range(n_staff):
            # Skill level distribution
            skill_probs = [0.25, 0.35, 0.30, 0.10]
            skill_level = np.random.choice(self.skill_levels, p=skill_probs)
            
            # Certifications based on skill level
            base_certs = ['basic_care', 'first_aid']
            
            if skill_level in ['intermediate', 'senior', 'specialist']:
                base_certs.extend(['medication_admin', 'manual_handling'])
            
            if skill_level in ['senior', 'specialist']:
                base_certs.extend(['dementia_care', 'palliative_care'])
            
            if skill_level == 'specialist':
                base_certs.extend(['nursing_procedures', 'complex_care'])
            
            # Working hours and availability
            full_time = random.random() > 0.3
            hours_per_week = 37.5 if full_time else np.random.randint(16, 30)
            
            # Home location for routing
            location = {
                'lat': 50.7192 + np.random.uniform(-0.08, 0.08),
                'lon': -1.8808 + np.random.uniform(-0.08, 0.08)
            }
            
            # Experience and performance metrics
            years_experience = {
                'junior': np.random.randint(0, 2),
                'intermediate': np.random.randint(2, 5),
                'senior': np.random.randint(5, 10),
                'specialist': np.random.randint(8, 20)
            }[skill_level]
            
            staff.append({
                'staff_id': f'S{i+1:03d}',
                'name': f'Carer_{i+1}',
                'skill_level': skill_level,
                'certifications': base_certs,
                'full_time': full_time,
                'hours_per_week': hours_per_week,
                'lat': location['lat'],
                'lon': location['lon'],
                'years_experience': years_experience,
                'hourly_rate': self._calculate_hourly_rate(skill_level, years_experience),
                'performance_score': np.random.uniform(0.7, 1.0)
            })
        
        return pd.DataFrame(staff)
    
    def generate_visit_requests(self, patients_df, date_range=7):
        """Generate visit requests for patients over a date range"""
        
        visits = []
        start_date = datetime.now().date()
        visit_id = 1
        
        for _, patient in patients_df.iterrows():
            visits_per_week = patient['visits_per_week']
            
            # Generate visits for the week
            for day in range(date_range):
                date = start_date + timedelta(days=day)
                
                # Number of visits for this day
                daily_visits = int(visits_per_week / 7) + (1 if random.random() < (visits_per_week % 7) / 7 else 0)
                
                for _ in range(daily_visits):
                    # Select visit type
                    visit_type = np.random.choice(
                        list(self.visit_types.keys()),
                        p=[v['frequency'] for v in self.visit_types.values()]
                    )
                    
                    # Calculate duration with patient-specific factors
                    base_duration = self.visit_types[visit_type]['base_duration']
                    variance = self.visit_types[visit_type]['variance']
                    
                    # Acuity affects duration
                    acuity_multiplier = {
                        'low': 0.9,
                        'medium': 1.0,
                        'high': 1.2,
                        'critical': 1.4
                    }[patient['acuity']]
                    
                    duration = int(base_duration * acuity_multiplier + np.random.uniform(-variance, variance))
                    duration = max(15, duration)  # Minimum 15 minutes
                    
                    # Time preferences
                    if patient['preferred_time'] == 'morning':
                        time_window_start = 7
                        time_window_end = 12
                    elif patient['preferred_time'] == 'afternoon':
                        time_window_start = 12
                        time_window_end = 17
                    elif patient['preferred_time'] == 'evening':
                        time_window_start = 17
                        time_window_end = 21
                    else:
                        time_window_start = 7
                        time_window_end = 21
                    
                    # Required staff level
                    if patient['acuity'] == 'critical':
                        required_level = random.choice(['senior', 'specialist'])
                    elif patient['acuity'] == 'high':
                        required_level = random.choice(['intermediate', 'senior'])
                    else:
                        required_level = random.choice(['junior', 'intermediate'])
                    
                    visits.append({
                        'visit_id': f'V{visit_id:04d}',
                        'patient_id': patient['patient_id'],
                        'date': date,
                        'visit_type': visit_type,
                        'estimated_duration': duration,
                        'acuity': patient['acuity'],
                        'time_window_start': time_window_start,
                        'time_window_end': time_window_end,
                        'required_skill_level': required_level,
                        'patient_lat': patient['lat'],
                        'patient_lon': patient['lon'],
                        'priority': self._calculate_priority(patient['acuity'], visit_type)
                    })
                    
                    visit_id += 1
        
        return pd.DataFrame(visits)
    
    def _calculate_hourly_rate(self, skill_level, years_experience):
        """Calculate hourly rate based on skill level and experience"""
        base_rates = {
            'junior': 12.50,
            'intermediate': 15.00,
            'senior': 18.50,
            'specialist': 22.00
        }
        
        base = base_rates[skill_level]
        experience_bonus = years_experience * 0.25
        return round(base + experience_bonus, 2)
    
    def _calculate_priority(self, acuity, visit_type):
        """Calculate visit priority (1-5, 5 being highest)"""
        acuity_score = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }[acuity]
        
        # Medication visits are always higher priority
        if visit_type == 'medication':
            return min(5, acuity_score + 1)
        
        return acuity_score


def generate_sample_data():
    """Generate a complete dataset for demonstration"""
    generator = CareDataGenerator(seed=42)
    
    print("Generating patients...")
    patients = generator.generate_patients(n_patients=50)
    
    print("Generating staff...")
    staff = generator.generate_staff(n_staff=15)
    
    print("Generating visit requests...")
    visits = generator.generate_visit_requests(patients, date_range=7)
    
    return patients, staff, visits


if __name__ == "__main__":
    patients, staff, visits = generate_sample_data()
    
    print(f"\nDataset Summary:")
    print(f"Patients: {len(patients)}")
    print(f"Staff: {len(staff)}")
    print(f"Visits (7 days): {len(visits)}")
    print(f"\nAcuity Distribution:")
    print(patients['acuity'].value_counts())
    print(f"\nStaff Skill Distribution:")
    print(staff['skill_level'].value_counts())
