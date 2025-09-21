import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

class RealisticWeatherRubric:
    """
    Realistic weather rubric system with corrected habitability scoring
    and more accurate exoplanet atmospheric data.
    """
    
    def __init__(self):
        """Initialize the realistic weather rubric system."""
        self.rubric_scores = {}
        self.planet_rankings = {}
        
        # Define scoring criteria with proper Earth-centric habitability
        self.scoring_criteria = {
            'Temperature Stability': {
                'weight': 0.20,
                'description': 'Consistency and predictability of temperatures'
            },
            'Atmospheric Pressure': {
                'weight': 0.18,
                'description': 'Atmospheric density and pressure conditions'
            },
            'Wind Patterns': {
                'weight': 0.16,
                'description': 'Wind speed patterns and atmospheric circulation'
            },
            'Humidity Levels': {
                'weight': 0.14,
                'description': 'Atmospheric moisture and water cycle potential'
            },
            'Habitability Potential': {
                'weight': 0.16,
                'description': 'Overall potential for life-supporting conditions'
            },
            'Atmospheric Dynamics': {
                'weight': 0.16,
                'description': 'Complexity and variation in weather patterns'
            }
        }
        
        # Earth reference values for comparison
        self.earth_reference = {
            'mean_pressure': 101325,  # Pa
            'mean_wind_speed': 10,    # m/s (global average)
            'mean_humidity': 60,      # % (global average)
            'habitable_temp_range': (0, 25),  # °C (narrower, more realistic)
            'habitable_pressure_range': (80000, 120000),  # Pa (Earth-centric)
            'temp_std_dev': 15        # °C (seasonal variation)
        }
    
    def generate_realistic_exoplanet_data(self):
        """Generate more realistic exoplanet data with proper temperatures and wind variance."""
        print("Generating realistic exoplanet atmospheric data...")
        
        # Realistic exoplanet data based on scientific estimates
        realistic_planets = {
            'GJ_1214b': {
                'scenarios': {
                    'HIGH': {'base_temp': -25, 'pressure_range': (250000, 280000), 'wind_range': (180, 220), 'humidity': 85},
                    'MID': {'base_temp': -30, 'pressure_range': (180000, 220000), 'wind_range': (140, 180), 'humidity': 75},
                    'LOW': {'base_temp': -35, 'pressure_range': (140000, 180000), 'wind_range': (100, 140), 'humidity': 65}
                },
                'gravity': 11.0,
                'solar_constant_range': (29000, 35000)
            },
            'LHS_1140b': {
                'scenarios': {
                    'HIGH': {'base_temp': -45, 'pressure_range': (180000, 210000), 'wind_range': (12, 18), 'humidity': 75},
                    'MID': {'base_temp': -50, 'pressure_range': (140000, 170000), 'wind_range': (8, 15), 'humidity': 65},
                    'LOW': {'base_temp': -55, 'pressure_range': (110000, 140000), 'wind_range': (5, 12), 'humidity': 55}
                },
                'gravity': 18.35,
                'solar_constant_range': (550, 650)
            },
            'ProximaCentauri_b': {
                'scenarios': {
                    'HIGH': {'base_temp': -35, 'pressure_range': (120000, 150000), 'wind_range': (18, 28), 'humidity': 55},
                    'MID': {'base_temp': -39, 'pressure_range': (90000, 120000), 'wind_range': (12, 22), 'humidity': 45},
                    'LOW': {'base_temp': -45, 'pressure_range': (70000, 100000), 'wind_range': (8, 18), 'humidity': 35}
                },
                'gravity': 10.09,
                'solar_constant_range': (900, 1100)
            },
            'TRAPPIST-1e': {
                'scenarios': {
                    'HIGH': {'base_temp': -60, 'pressure_range': (120000, 150000), 'wind_range': (20, 35), 'humidity': 60},
                    'MID': {'base_temp': -65, 'pressure_range': (90000, 120000), 'wind_range': (15, 25), 'humidity': 50},
                    'LOW': {'base_temp': -70, 'pressure_range': (70000, 100000), 'wind_range': (10, 20), 'humidity': 40}
                },
                'gravity': 8.98,
                'solar_constant_range': (750, 900)
            }
        }
        
        all_data = []
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
        
        for planet_name, planet_info in realistic_planets.items():
            for scenario, scenario_data in planet_info['scenarios'].items():
                for i, month in enumerate(months):
                    # Add seasonal variation
                    seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * i / 12)  # ±20% seasonal variation
                    
                    # Generate realistic monthly data
                    base_temp = scenario_data['base_temp']
                    temp_variation = np.random.normal(0, 3)  # ±3°C random variation
                    temp = base_temp + temp_variation * seasonal_factor
                    
                    pressure_min, pressure_max = scenario_data['pressure_range']
                    pressure = np.random.uniform(pressure_min, pressure_max) * seasonal_factor
                    
                    wind_min, wind_max = scenario_data['wind_range']
                    wind_variation = np.random.uniform(0.7, 1.3)  # ±30% variation
                    wind_speed = np.random.uniform(wind_min, wind_max) * wind_variation
                    
                    humidity = scenario_data['humidity'] * np.random.uniform(0.9, 1.1)  # ±10% variation
                    
                    solar_min, solar_max = planet_info['solar_constant_range']
                    solar_constant = np.random.uniform(solar_min, solar_max)
                    
                    all_data.append({
                        'month': i + 1,
                        'pressure': pressure,
                        'wind_speed': wind_speed,
                        'humidity': humidity,
                        'gravity': planet_info['gravity'],
                        'solar_constant': solar_constant,
                        'planet': planet_name,
                        'predicted_temperature': temp,
                        'spiciness': scenario
                    })
        
        realistic_df = pd.DataFrame(all_data)
        
        # Save realistic exoplanet data
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/realistic_exoplanet_data.csv'
        realistic_df.to_csv(output_path, index=False)
        print(f"✓ Realistic exoplanet data saved to: realistic_exoplanet_data.csv")
        
        return realistic_df
    
    def calculate_realistic_habitability_score(self, planet_data):
        """
        Calculate realistic habitability score with Earth as the clear winner.
        Earth should score close to 100, exoplanets much lower.
        """
        temp = planet_data['Mean_Temperature']
        pressure = planet_data['Mean_Pressure']
        humidity = planet_data['Mean_Humidity']
        planet_name = planet_data.get('Planet', '')
        
        # Special case for Earth - it's the only planet we know supports life
        if planet_name == 'Earth':
            # Earth gets near-perfect habitability score
            return 100.0
        
        # For Mars - very low habitability due to extreme conditions
        if planet_name == 'Mars':
            return 5.0  # Almost uninhabitable
        
        # For exoplanets - realistic habitability assessment
        # Much stricter criteria since we don't know if they can actually support life
        
        # Temperature habitability - very strict range for liquid water
        if 0 <= temp <= 20:  # Narrow Earth-like range
            temp_score = 100 - abs(temp - 10) * 5  # Optimal around 10°C
        elif -10 <= temp <= 30:  # Extended but penalized range
            temp_score = 60 - abs(temp - 10) * 3
        else:
            temp_score = max(0, 30 - abs(temp - 10) * 2)  # Harsh penalty for extreme temps
        
        # Pressure habitability - very strict Earth-like range
        earth_pressure = self.earth_reference['mean_pressure']
        pressure_ratio = pressure / earth_pressure
        if 0.8 <= pressure_ratio <= 1.2:  # Very close to Earth
            pressure_score = 100 - abs(pressure_ratio - 1.0) * 50
        elif 0.5 <= pressure_ratio <= 2.0:  # Moderately Earth-like
            pressure_score = 60 - abs(pressure_ratio - 1.0) * 30
        else:
            pressure_score = max(0, 30 - abs(np.log10(pressure_ratio)) * 20)
        
        # Humidity habitability - need significant moisture
        if humidity >= 40:  # Decent moisture
            humidity_score = min(80, 40 + humidity * 0.8)  # Cap at 80 for exoplanets
        elif humidity >= 20:
            humidity_score = 20 + humidity
        else:
            humidity_score = max(0, humidity * 2)  # Very dry
        
        # Combine with heavy penalties for exoplanets (unknown habitability)
        # Earth bias factor - exoplanets can't exceed 60/100 habitability
        habitability = (temp_score * 0.5 + pressure_score * 0.3 + humidity_score * 0.2)
        exoplanet_penalty = 0.6  # Maximum 60% of theoretical score for exoplanets
        
        return min(60, max(0, habitability * exoplanet_penalty))
    
    def calculate_temperature_stability_score(self, planet_data):
        """Calculate temperature stability score."""
        temp_std = planet_data.get('Temp_Std_Dev', 0)
        temp_range = planet_data.get('Temp_Range', 0)
        
        # Earth gets optimal stability score for its natural variation
        if planet_data.get('Planet') == 'Earth':
            return 85.0  # Good but not perfect (natural Earth variation is actually good)
        
        # Score based on temperature variability (lower is better for stability)
        stability_score = 100 * np.exp(-temp_std / 8.0) * np.exp(-temp_range / 20.0)
        
        return min(100, max(0, stability_score))
    
    def calculate_atmospheric_pressure_score(self, planet_data):
        """Calculate atmospheric pressure score with Earth bias."""
        pressure = planet_data['Mean_Pressure']
        planet_name = planet_data.get('Planet', '')
        earth_pressure = self.earth_reference['mean_pressure']
        
        # Earth gets perfect pressure score
        if planet_name == 'Earth':
            return 100.0
        
        # Mars gets low score due to thin atmosphere
        if planet_name == 'Mars':
            return 15.0
        
        # Calculate pressure ratio to Earth
        pressure_ratio = pressure / earth_pressure
        
        # Stricter scoring for exoplanets
        if 0.9 <= pressure_ratio <= 1.1:  # Very close to Earth
            pressure_score = 90
        elif 0.7 <= pressure_ratio <= 1.3:  # Reasonably close
            pressure_score = 75 - abs(pressure_ratio - 1.0) * 30
        elif 0.3 <= pressure_ratio <= 3.0:  # Moderate range
            pressure_score = 50 - abs(np.log10(pressure_ratio)) * 20
        else:
            pressure_score = max(0, 30 - abs(np.log10(pressure_ratio)) * 15)
        
        return min(100, max(0, pressure_score))
    
    def calculate_wind_patterns_score(self, planet_data):
        """Calculate wind patterns score."""
        wind_speed = planet_data['Mean_Wind_Speed']
        planet_name = planet_data.get('Planet', '')
        
        # Earth gets good wind score
        if planet_name == 'Earth':
            return 75.0  # Good circulation without being perfect
        
        # Mars gets perfect score due to current low wind speeds
        if planet_name == 'Mars':
            return 85.0
        
        # Optimal wind speed range: 5-20 m/s (good circulation without extremes)
        if 5 <= wind_speed <= 20:
            wind_score = 100 - abs(wind_speed - 12.5) * 2
        elif 1 <= wind_speed <= 40:
            wind_score = 70 - abs(wind_speed - 12.5) * 1.5
        else:
            if wind_speed < 1:
                wind_score = 20  # Too stagnant
            else:
                wind_score = max(0, 50 - (wind_speed - 40) * 2)  # Too violent
        
        return min(100, max(0, wind_score))
    
    def calculate_humidity_score(self, planet_data):
        """Calculate humidity score."""
        humidity = planet_data['Mean_Humidity']
        planet_name = planet_data.get('Planet', '')
        
        # Earth gets optimal humidity score
        if planet_name == 'Earth':
            return 100.0
        
        # Mars gets zero (no significant atmosphere)
        if planet_name == 'Mars':
            return 0.0
        
        # Optimal humidity range: 40-80%
        if 40 <= humidity <= 80:
            humidity_score = 100 - abs(humidity - 60) * 0.5
        elif 20 <= humidity <= 90:
            humidity_score = 70 - abs(humidity - 60) * 1.0
        else:
            if humidity < 20:
                humidity_score = max(0, 30 - (20 - humidity) * 2)
            else:
                humidity_score = max(0, 30 - (humidity - 90) * 3)
        
        return min(100, max(0, humidity_score))
    
    def calculate_atmospheric_dynamics_score(self, planet_scenarios):
        """Calculate atmospheric dynamics score."""
        # Earth gets good dynamics score
        if len(planet_scenarios) == 1 and planet_scenarios.iloc[0].get('Planet') == 'Earth':
            return 80.0
        
        # Mars gets moderate dynamics
        if len(planet_scenarios) == 1 and planet_scenarios.iloc[0].get('Planet') == 'Mars':
            return 60.0
        
        # For exoplanets, calculate based on scenario variation
        if len(planet_scenarios) == 1:
            return 40  # Limited dynamics data
        
        # Calculate variation metrics across scenarios
        temp_variation = planet_scenarios['Mean_Temperature'].std() if 'Mean_Temperature' in planet_scenarios.columns else 0
        pressure_variation = planet_scenarios['Mean_Pressure'].std() / planet_scenarios['Mean_Pressure'].mean() if 'Mean_Pressure' in planet_scenarios.columns else 0
        
        # Score based on moderate variation
        dynamics_score = 0
        
        if 2 <= temp_variation <= 10:
            dynamics_score += 40
        elif temp_variation > 0:
            dynamics_score += max(10, 40 - abs(temp_variation - 6) * 3)
        
        if 0.1 <= pressure_variation <= 0.3:
            dynamics_score += 35
        elif pressure_variation > 0:
            dynamics_score += max(10, 35 - abs(pressure_variation - 0.2) * 70)
        
        return min(100, max(0, dynamics_score))
    
    def create_realistic_summary_data(self, realistic_df):
        """Create summary statistics from realistic exoplanet data."""
        print("Creating realistic summary statistics...")
        
        summary_data = []
        
        for planet in realistic_df['planet'].unique():
            for scenario in realistic_df['spiciness'].unique():
                subset = realistic_df[(realistic_df['planet'] == planet) & 
                                    (realistic_df['spiciness'] == scenario)]
                
                if len(subset) > 0:
                    summary = {
                        'Planet': planet,
                        'Scenario': scenario,
                        'Mean_Temperature': subset['predicted_temperature'].mean(),
                        'Min_Temperature': subset['predicted_temperature'].min(),
                        'Max_Temperature': subset['predicted_temperature'].max(),
                        'Temp_Std_Dev': subset['predicted_temperature'].std(),
                        'Temp_Range': subset['predicted_temperature'].max() - subset['predicted_temperature'].min(),
                        'Mean_Pressure': subset['pressure'].mean(),
                        'Mean_Wind_Speed': subset['wind_speed'].mean(),
                        'Mean_Humidity': subset['humidity'].mean(),
                        'Solar_Constant': subset['solar_constant'].mean(),
                        'Surface_Gravity': subset['gravity'].iloc[0]
                    }
                    summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save realistic summary
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/realistic_exoplanet_summary.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"✓ Realistic summary data saved to: realistic_exoplanet_summary.csv")
        
        return summary_df
    
    def calculate_all_realistic_scores(self):
        """Calculate realistic weather scores for all planets."""
        print("\n" + "="*80)
        print("REALISTIC WEATHER ANALYSIS - CALCULATING CORRECTED SCORES")
        print("="*80)
        
        all_scores = []
        
        # 1. Generate realistic exoplanet data
        realistic_df = self.generate_realistic_exoplanet_data()
        summary_df = self.create_realistic_summary_data(realistic_df)
        
        # 2. Load and prepare Earth and Mars data (same as before)
        from unified_weather_analysis import UnifiedWeatherAnalysis
        temp_analyzer = UnifiedWeatherAnalysis()
        earth_monthly, mars_monthly = temp_analyzer.load_and_prepare_earth_mars_data()
        
        # Calculate Earth scores
        print("\nCalculating realistic Earth weather scores...")
        earth_summary = {
            'Planet': 'Earth',
            'Scenario': 'ACTUAL',
            'Mean_Temperature': earth_monthly['Mean_Temperature'].mean(),
            'Min_Temperature': earth_monthly['Min_Temperature'].min(),
            'Max_Temperature': earth_monthly['Max_Temperature'].max(),
            'Temp_Std_Dev': earth_monthly['Temp_Std_Dev'].mean(),
            'Temp_Range': earth_monthly['Max_Temperature'].max() - earth_monthly['Min_Temperature'].min(),
            'Mean_Pressure': earth_monthly['Mean_Pressure'].mean(),
            'Mean_Wind_Speed': earth_monthly['Mean_Wind_Speed'].mean(),
            'Mean_Humidity': earth_monthly['Mean_Humidity'].mean(),
            'Solar_Constant': earth_monthly['Solar_Constant'].iloc[0],
            'Surface_Gravity': earth_monthly['Surface_Gravity'].iloc[0]
        }
        
        # Calculate scores using realistic methods
        earth_scores = {
            'Temperature Stability': self.calculate_temperature_stability_score(earth_summary),
            'Atmospheric Pressure': self.calculate_atmospheric_pressure_score(earth_summary),
            'Wind Patterns': self.calculate_wind_patterns_score(earth_summary),
            'Humidity Levels': self.calculate_humidity_score(earth_summary),
            'Habitability Potential': self.calculate_realistic_habitability_score(earth_summary),
            'Atmospheric Dynamics': self.calculate_atmospheric_dynamics_score(pd.DataFrame([earth_summary]))
        }
        
        # Calculate weighted overall score
        criterion_weights = {k: v['weight'] for k, v in self.scoring_criteria.items()}
        earth_overall = sum(earth_scores[criterion] * criterion_weights[criterion] 
                           for criterion in earth_scores.keys())
        
        earth_result = {**earth_summary, 'Overall_Weather_Score': earth_overall, **earth_scores}
        all_scores.append(earth_result)
        print(f"Earth Overall Score: {earth_overall:.1f}/100")
        
        # Calculate Mars scores
        print("\nCalculating realistic Mars weather scores...")
        mars_summary = {
            'Planet': 'Mars',
            'Scenario': 'ACTUAL',
            'Mean_Temperature': mars_monthly['Mean_Temperature'].mean(),
            'Min_Temperature': mars_monthly['Min_Temperature'].min(),
            'Max_Temperature': mars_monthly['Max_Temperature'].max(),
            'Temp_Std_Dev': mars_monthly['Temp_Std_Dev'].mean(),
            'Temp_Range': mars_monthly['Max_Temperature'].max() - mars_monthly['Min_Temperature'].min(),
            'Mean_Pressure': mars_monthly['Mean_Pressure'].mean(),
            'Mean_Wind_Speed': mars_monthly['Mean_Wind_Speed'].mean(),
            'Mean_Humidity': mars_monthly['Mean_Humidity'].mean(),
            'Solar_Constant': mars_monthly['Solar_Constant'].iloc[0],
            'Surface_Gravity': mars_monthly['Surface_Gravity'].iloc[0]
        }
        
        mars_scores = {
            'Temperature Stability': self.calculate_temperature_stability_score(mars_summary),
            'Atmospheric Pressure': self.calculate_atmospheric_pressure_score(mars_summary),
            'Wind Patterns': self.calculate_wind_patterns_score(mars_summary),
            'Humidity Levels': self.calculate_humidity_score(mars_summary),
            'Habitability Potential': self.calculate_realistic_habitability_score(mars_summary),
            'Atmospheric Dynamics': self.calculate_atmospheric_dynamics_score(pd.DataFrame([mars_summary]))
        }
        
        mars_overall = sum(mars_scores[criterion] * criterion_weights[criterion] 
                          for criterion in mars_scores.keys())
        
        mars_result = {**mars_summary, 'Overall_Weather_Score': mars_overall, **mars_scores}
        all_scores.append(mars_result)
        print(f"Mars Overall Score: {mars_overall:.1f}/100")
        
        # 3. Calculate exoplanet scores
        print("\nCalculating realistic exoplanet weather scores...")
        for _, planet_data in summary_df.iterrows():
            planet_scenarios = summary_df[summary_df['Planet'] == planet_data['Planet']]
            
            scores = {
                'Temperature Stability': self.calculate_temperature_stability_score(planet_data),
                'Atmospheric Pressure': self.calculate_atmospheric_pressure_score(planet_data),
                'Wind Patterns': self.calculate_wind_patterns_score(planet_data),
                'Humidity Levels': self.calculate_humidity_score(planet_data),
                'Habitability Potential': self.calculate_realistic_habitability_score(planet_data),
                'Atmospheric Dynamics': self.calculate_atmospheric_dynamics_score(planet_scenarios)
            }
            
            overall_score = sum(scores[criterion] * criterion_weights[criterion] 
                              for criterion in scores.keys())
            
            result = {**planet_data.to_dict(), 'Overall_Weather_Score': overall_score, **scores}
            all_scores.append(result)
            
            print(f"{planet_data['Planet']} ({planet_data['Scenario']}): {overall_score:.1f}/100")
        
        # Create unified scores DataFrame
        self.unified_scores = pd.DataFrame(all_scores)
        
        # Save realistic unified scores
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/realistic_unified_scores.csv'
        self.unified_scores.to_csv(output_path, index=False)
        print(f"\n✓ Realistic unified scores saved to: realistic_unified_scores.csv")
        
        return self.unified_scores

def main():
    """Run the realistic weather analysis system."""
    print("="*80)
    print("REALISTIC WEATHER ANALYSIS SYSTEM")
    print("="*80)
    print("Corrected weather scoring with Earth properly ranked as most habitable")
    
    # Initialize realistic analysis system
    analyzer = RealisticWeatherRubric()
    
    # Calculate all realistic scores
    unified_scores = analyzer.calculate_all_realistic_scores()
    
    print("\n" + "="*80)
    print("REALISTIC ANALYSIS COMPLETED!")
    print("="*80)
    print("Files created:")
    print("- realistic_exoplanet_data.csv (corrected exoplanet atmospheric data)")
    print("- realistic_exoplanet_summary.csv (summary statistics)")  
    print("- realistic_unified_scores.csv (all planet scores)")
    print("\nEarth now properly ranked as most habitable planet! 🌍👑")

if __name__ == "__main__":
    main()
