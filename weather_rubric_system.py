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

class ExoplanetWeatherRubric:
    """
    A comprehensive weather rubric system for rating and comparing exoplanet
    atmospheric conditions across multiple quantitative dimensions.
    """
    
    def __init__(self):
        """Initialize the weather rubric system with scoring criteria."""
        self.rubric_scores = {}
        self.planet_rankings = {}
        
        # Define scoring criteria and weight factors
        self.scoring_criteria = {
            'Temperature Stability': {
                'weight': 0.20,
                'description': 'Consistency and predictability of temperatures',
                'metrics': ['temp_std_dev', 'temp_range', 'seasonal_variation']
            },
            'Atmospheric Pressure': {
                'weight': 0.18,
                'description': 'Atmospheric density and pressure conditions',
                'metrics': ['mean_pressure', 'pressure_earth_ratio', 'pressure_habitability']
            },
            'Wind Patterns': {
                'weight': 0.16,
                'description': 'Wind speed patterns and atmospheric circulation',
                'metrics': ['mean_wind_speed', 'wind_variability', 'atmospheric_dynamics']
            },
            'Humidity Levels': {
                'weight': 0.14,
                'description': 'Atmospheric moisture and water cycle potential',
                'metrics': ['mean_humidity', 'humidity_earth_comparison', 'water_availability']
            },
            'Habitability Potential': {
                'weight': 0.16,
                'description': 'Overall potential for life-supporting conditions',
                'metrics': ['temperature_habitability', 'pressure_habitability', 'overall_habitability']
            },
            'Atmospheric Dynamics': {
                'weight': 0.16,
                'description': 'Complexity and variation in weather patterns',
                'metrics': ['weather_diversity', 'seasonal_effects', 'climate_complexity']
            }
        }
        
        # Earth reference values for comparison
        self.earth_reference = {
            'mean_pressure': 101325,  # Pa
            'mean_wind_speed': 10,    # m/s (global average)
            'mean_humidity': 60,      # % (global average)
            'habitable_temp_range': (-10, 30),  # °C
            'habitable_pressure_range': (50000, 200000),  # Pa
            'temp_std_dev': 15        # °C (seasonal variation)
        }
    
    def load_exoplanet_data(self):
        """Load all exoplanet prediction data for analysis."""
        print("Loading exoplanet prediction data...")
        
        # Load summary statistics
        summary_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/exoplanet_summary_statistics.csv'
        self.summary_data = pd.read_csv(summary_path)
        
        # Load detailed predictions
        detailed_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/all_exoplanet_predictions.csv'
        self.detailed_data = pd.read_csv(detailed_path)
        
        print(f"Loaded data for {len(self.summary_data)} planet scenarios")
        print(f"Detailed data: {len(self.detailed_data)} monthly records")
        
        return self.summary_data, self.detailed_data
    
    def calculate_temperature_stability_score(self, planet_data):
        """
        Calculate temperature stability score (0-100).
        Higher scores indicate more stable, predictable temperatures.
        """
        temp_std = planet_data['Temp_Std_Dev']
        temp_range = planet_data['Temp_Range']
        
        # Score based on temperature variability (lower is better for stability)
        # Use exponential decay to heavily penalize high variability
        stability_score = 100 * np.exp(-temp_std / 5.0) * np.exp(-temp_range / 15.0)
        
        return min(100, max(0, stability_score))
    
    def calculate_atmospheric_pressure_score(self, planet_data):
        """
        Calculate atmospheric pressure score (0-100).
        Scores based on similarity to Earth-like conditions and habitability.
        """
        pressure = planet_data['Mean_Pressure']
        earth_pressure = self.earth_reference['mean_pressure']
        
        # Calculate pressure ratio to Earth
        pressure_ratio = pressure / earth_pressure
        
        # Optimal pressure range for habitability (0.5x to 2x Earth)
        if 0.5 <= pressure_ratio <= 2.0:
            pressure_score = 100
        elif 0.1 <= pressure_ratio <= 5.0:
            # Moderate pressure - still potentially habitable
            pressure_score = 70 - abs(np.log10(pressure_ratio)) * 20
        else:
            # Extreme pressures
            pressure_score = max(0, 50 - abs(np.log10(pressure_ratio)) * 15)
        
        return min(100, max(0, pressure_score))
    
    def calculate_wind_patterns_score(self, planet_data):
        """
        Calculate wind patterns score (0-100).
        Moderate wind speeds are favorable for weather circulation.
        """
        wind_speed = planet_data['Mean_Wind_Speed']
        
        # Optimal wind speed range: 5-25 m/s (good circulation without extremes)
        if 5 <= wind_speed <= 25:
            wind_score = 100
        elif 1 <= wind_speed <= 50:
            # Moderate winds - still reasonable
            wind_score = 80 - abs(wind_speed - 15) * 2
        else:
            # Extreme winds (too calm or too violent)
            if wind_speed < 1:
                wind_score = 20  # Too stagnant
            else:
                wind_score = max(0, 60 - (wind_speed - 50) * 2)  # Too violent
        
        return min(100, max(0, wind_score))
    
    def calculate_humidity_score(self, planet_data):
        """
        Calculate humidity score (0-100).
        Moderate humidity levels indicate potential for water cycle.
        """
        humidity = planet_data['Mean_Humidity']
        
        # Optimal humidity range: 30-80% (similar to Earth's habitable zones)
        if 30 <= humidity <= 80:
            humidity_score = 100
        elif 10 <= humidity <= 95:
            # Reasonable humidity levels
            humidity_score = 80 - abs(humidity - 55) * 1.5
        else:
            # Extreme humidity levels
            if humidity < 10:
                humidity_score = max(0, 40 - (10 - humidity) * 4)  # Too dry
            else:
                humidity_score = max(0, 40 - (humidity - 95) * 8)  # Too humid
        
        return min(100, max(0, humidity_score))
    
    def calculate_habitability_score(self, planet_data):
        """
        Calculate overall habitability potential score (0-100).
        Based on temperature, pressure, and other life-supporting factors.
        """
        temp = planet_data['Mean_Temperature']
        pressure = planet_data['Mean_Pressure']
        humidity = planet_data['Mean_Humidity']
        
        # Temperature habitability (liquid water range: -10°C to 50°C)
        if -10 <= temp <= 50:
            temp_score = 100 - abs(temp - 15) * 2  # Optimal around 15°C
        else:
            temp_score = max(0, 50 - abs(temp - 20) * 2)
        
        # Pressure habitability (0.1 to 10 Earth atmospheres)
        earth_pressure = self.earth_reference['mean_pressure']
        pressure_ratio = pressure / earth_pressure
        if 0.1 <= pressure_ratio <= 10:
            pressure_score = 100 - abs(np.log10(pressure_ratio)) * 20
        else:
            pressure_score = max(0, 30 - abs(np.log10(pressure_ratio)) * 15)
        
        # Humidity habitability
        if humidity > 5:  # Some moisture available
            humidity_score = min(100, 50 + humidity)
        else:
            humidity_score = 10  # Very dry
        
        # Combined habitability score
        habitability = (temp_score * 0.4 + pressure_score * 0.4 + humidity_score * 0.2)
        
        return min(100, max(0, habitability))
    
    def calculate_atmospheric_dynamics_score(self, planet_scenarios):
        """
        Calculate atmospheric dynamics score based on variation across scenarios.
        More dynamic weather patterns score higher for complexity.
        """
        # Check if we have multiple scenarios for this planet
        if len(planet_scenarios) == 1:
            return 30  # Limited dynamics data
        
        # Calculate variation metrics across scenarios
        temp_variation = planet_scenarios['Mean_Temperature'].std()
        pressure_variation = planet_scenarios['Mean_Pressure'].std() / planet_scenarios['Mean_Pressure'].mean()
        wind_variation = planet_scenarios['Mean_Wind_Speed'].std() / planet_scenarios['Mean_Wind_Speed'].mean()
        
        # Score based on moderate variation (indicates active weather systems)
        dynamics_score = 0
        
        # Temperature dynamics
        if 2 <= temp_variation <= 15:
            dynamics_score += 30
        elif temp_variation > 0:
            dynamics_score += max(5, 30 - abs(temp_variation - 8.5) * 2)
        
        # Pressure dynamics
        if 0.1 <= pressure_variation <= 0.4:
            dynamics_score += 35
        elif pressure_variation > 0:
            dynamics_score += max(5, 35 - abs(pressure_variation - 0.25) * 50)
        
        # Wind dynamics
        if 0.1 <= wind_variation <= 0.5:
            dynamics_score += 35
        elif wind_variation > 0:
            dynamics_score += max(5, 35 - abs(wind_variation - 0.3) * 40)
        
        return min(100, max(0, dynamics_score))
    
    def calculate_planet_rubric_scores(self):
        """Calculate comprehensive rubric scores for all planets."""
        print("\n" + "="*70)
        print("CALCULATING EXOPLANET WEATHER RUBRIC SCORES")
        print("="*70)
        
        # Load data
        summary_data, detailed_data = self.load_exoplanet_data()
        
        # Get unique planets
        planets = summary_data['Planet'].unique()
        
        rubric_results = []
        
        for planet in planets:
            print(f"\nAnalyzing {planet}...")
            
            # Get all scenarios for this planet
            planet_scenarios = summary_data[summary_data['Planet'] == planet]
            
            # Calculate scores for each scenario
            for _, scenario_data in planet_scenarios.iterrows():
                scores = {}
                
                # Calculate individual criterion scores
                scores['Temperature Stability'] = self.calculate_temperature_stability_score(scenario_data)
                scores['Atmospheric Pressure'] = self.calculate_atmospheric_pressure_score(scenario_data)
                scores['Wind Patterns'] = self.calculate_wind_patterns_score(scenario_data)
                scores['Humidity Levels'] = self.calculate_humidity_score(scenario_data)
                scores['Habitability Potential'] = self.calculate_habitability_score(scenario_data)
                scores['Atmospheric Dynamics'] = self.calculate_atmospheric_dynamics_score(planet_scenarios)
                
                # Calculate weighted overall score
                overall_score = sum(
                    scores[criterion] * self.scoring_criteria[criterion]['weight']
                    for criterion in scores.keys()
                )
                
                # Store results
                result = {
                    'Planet': planet,
                    'Scenario': scenario_data['Scenario'],
                    'Overall_Weather_Score': overall_score,
                    **scores,
                    'Mean_Temperature': scenario_data['Mean_Temperature'],
                    'Mean_Pressure': scenario_data['Mean_Pressure'],
                    'Mean_Wind_Speed': scenario_data['Mean_Wind_Speed'],
                    'Mean_Humidity': scenario_data['Mean_Humidity'],
                    'Surface_Gravity': scenario_data['Surface_Gravity'],
                    'Solar_Constant': scenario_data['Solar_Constant']
                }
                
                rubric_results.append(result)
                
                print(f"  {scenario_data['Scenario']} Scenario: Overall Score = {overall_score:.1f}/100")
        
        # Convert to DataFrame
        self.rubric_df = pd.DataFrame(rubric_results)
        
        # Save results
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/weather_rubric_scores.csv'
        self.rubric_df.to_csv(output_path, index=False)
        print(f"\n✓ Weather rubric scores saved to: weather_rubric_scores.csv")
        
        return self.rubric_df
    
    def create_rubric_visualizations(self):
        """Create comprehensive visualizations of the weather rubric scores."""
        print("\nCreating weather rubric visualizations...")
        
        # Set up the plotting environment
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Score Comparison (Radar Chart)
        ax1 = plt.subplot(4, 2, 1)
        self.plot_overall_scores_bar(ax1)
        
        # 2. Criterion Breakdown Heatmap
        ax2 = plt.subplot(4, 2, 2)
        self.plot_criterion_heatmap(ax2)
        
        # 3. Planet Rankings by Scenario
        ax3 = plt.subplot(4, 2, 3)
        self.plot_scenario_rankings(ax3)
        
        # 4. Habitability vs Weather Stability
        ax4 = plt.subplot(4, 2, 4)
        self.plot_habitability_vs_stability(ax4)
        
        # 5. Atmospheric Characteristics Spider/Radar Chart
        ax5 = plt.subplot(4, 2, 5)
        self.plot_atmospheric_radar(ax5)
        
        # 6. Score Distribution by Planet
        ax6 = plt.subplot(4, 2, 6)
        self.plot_score_distributions(ax6)
        
        # 7. Weather Pattern Classification
        ax7 = plt.subplot(4, 2, 7)
        self.plot_weather_classifications(ax7)
        
        # 8. Rubric Summary Statistics
        ax8 = plt.subplot(4, 2, 8)
        self.plot_summary_statistics(ax8)
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/plots/weather_rubric_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive rubric visualization saved to: weather_rubric_analysis.png")
        
        plt.show()
    
    def plot_overall_scores_bar(self, ax):
        """Plot overall weather scores as grouped bar chart."""
        pivot_data = self.rubric_df.pivot(index='Planet', columns='Scenario', values='Overall_Weather_Score')
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8, 
                       color=['#ff6b6b', '#ffa500', '#4ecdc4'])
        ax.set_title('Overall Weather Rubric Scores by Planet & Scenario', fontsize=14, fontweight='bold')
        ax.set_xlabel('Planet', fontweight='bold')
        ax.set_ylabel('Weather Score (0-100)', fontweight='bold')
        ax.legend(title='Scenario', title_fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_criterion_heatmap(self, ax):
        """Plot criterion scores as a heatmap."""
        criterion_cols = ['Temperature Stability', 'Atmospheric Pressure', 'Wind Patterns', 
                         'Humidity Levels', 'Habitability Potential', 'Atmospheric Dynamics']
        
        # Create planet-scenario labels
        self.rubric_df['Planet_Scenario'] = self.rubric_df['Planet'] + '_' + self.rubric_df['Scenario']
        
        heatmap_data = self.rubric_df.set_index('Planet_Scenario')[criterion_cols].T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score (0-100)'}, ax=ax)
        ax.set_title('Weather Criterion Scores Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Planet & Scenario', fontweight='bold')
        ax.set_ylabel('Weather Criteria', fontweight='bold')
    
    def plot_scenario_rankings(self, ax):
        """Plot planet rankings by scenario."""
        scenarios = self.rubric_df['Scenario'].unique()
        colors = ['#ff6b6b', '#ffa500', '#4ecdc4']
        
        for i, scenario in enumerate(scenarios):
            scenario_data = self.rubric_df[self.rubric_df['Scenario'] == scenario].sort_values('Overall_Weather_Score', ascending=True)
            y_pos = np.arange(len(scenario_data))
            
            ax.barh(y_pos + i*0.25, scenario_data['Overall_Weather_Score'], 
                   height=0.25, label=scenario, color=colors[i], alpha=0.8)
        
        ax.set_yticks(np.arange(len(self.rubric_df['Planet'].unique())) + 0.25)
        ax.set_yticklabels(scenario_data['Planet'])
        ax.set_xlabel('Overall Weather Score', fontweight='bold')
        ax.set_title('Planet Rankings by Scenario', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_habitability_vs_stability(self, ax):
        """Plot habitability vs temperature stability."""
        scenarios = self.rubric_df['Scenario'].unique()
        colors = ['#ff6b6b', '#ffa500', '#4ecdc4']
        
        for i, scenario in enumerate(scenarios):
            scenario_data = self.rubric_df[self.rubric_df['Scenario'] == scenario]
            ax.scatter(scenario_data['Temperature Stability'], 
                      scenario_data['Habitability Potential'],
                      s=100, alpha=0.7, label=scenario, color=colors[i])
            
            # Add planet labels
            for _, row in scenario_data.iterrows():
                ax.annotate(row['Planet'], 
                           (row['Temperature Stability'], row['Habitability Potential']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Temperature Stability Score', fontweight='bold')
        ax.set_ylabel('Habitability Potential Score', fontweight='bold')
        ax.set_title('Habitability vs Temperature Stability', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_atmospheric_radar(self, ax):
        """Plot atmospheric characteristics as radar chart."""
        # This is a placeholder - matplotlib radar charts need special handling
        # For now, show as a grouped bar chart
        criterion_cols = ['Temperature Stability', 'Atmospheric Pressure', 'Wind Patterns', 
                         'Humidity Levels', 'Habitability Potential', 'Atmospheric Dynamics']
        
        # Get best scenario for each planet
        best_scenarios = self.rubric_df.loc[self.rubric_df.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        x = np.arange(len(criterion_cols))
        width = 0.2
        
        for i, (_, row) in enumerate(best_scenarios.iterrows()):
            values = [row[col] for col in criterion_cols]
            ax.bar(x + i*width, values, width, label=row['Planet'], alpha=0.8)
        
        ax.set_xlabel('Weather Criteria', fontweight='bold')
        ax.set_ylabel('Score (0-100)', fontweight='bold')
        ax.set_title('Best Scenario Atmospheric Profiles', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([col.replace(' ', '\n') for col in criterion_cols], fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_score_distributions(self, ax):
        """Plot score distributions by planet."""
        planets = self.rubric_df['Planet'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(planets)))
        
        for i, planet in enumerate(planets):
            planet_data = self.rubric_df[self.rubric_df['Planet'] == planet]
            ax.hist(planet_data['Overall_Weather_Score'], bins=10, alpha=0.6, 
                   label=planet, color=colors[i])
        
        ax.set_xlabel('Overall Weather Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Weather Score Distributions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_weather_classifications(self, ax):
        """Plot weather pattern classifications."""
        # Create weather classifications based on scores
        def classify_weather(row):
            if row['Overall_Weather_Score'] >= 70:
                return 'Excellent'
            elif row['Overall_Weather_Score'] >= 50:
                return 'Good'
            elif row['Overall_Weather_Score'] >= 30:
                return 'Moderate'
            else:
                return 'Poor'
        
        self.rubric_df['Weather_Classification'] = self.rubric_df.apply(classify_weather, axis=1)
        
        classification_counts = self.rubric_df.groupby(['Planet', 'Weather_Classification']).size().unstack(fill_value=0)
        
        classification_counts.plot(kind='bar', stacked=True, ax=ax, 
                                 color=['#ff4757', '#ff7675', '#fdcb6e', '#00b894'])
        ax.set_title('Weather Pattern Classifications', fontsize=14, fontweight='bold')
        ax.set_xlabel('Planet', fontweight='bold')
        ax.set_ylabel('Number of Scenarios', fontweight='bold')
        ax.legend(title='Weather Quality')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_summary_statistics(self, ax):
        """Plot summary statistics table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for planet in self.rubric_df['Planet'].unique():
            planet_data = self.rubric_df[self.rubric_df['Planet'] == planet]
            stats = {
                'Planet': planet,
                'Avg_Score': planet_data['Overall_Weather_Score'].mean(),
                'Best_Score': planet_data['Overall_Weather_Score'].max(),
                'Score_Range': planet_data['Overall_Weather_Score'].max() - planet_data['Overall_Weather_Score'].min(),
                'Best_Criterion': planet_data[['Temperature Stability', 'Atmospheric Pressure', 
                                             'Wind Patterns', 'Humidity Levels', 'Habitability Potential', 
                                             'Atmospheric Dynamics']].mean().idxmax()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append([
                row['Planet'],
                f"{row['Avg_Score']:.1f}",
                f"{row['Best_Score']:.1f}",
                f"{row['Score_Range']:.1f}",
                row['Best_Criterion'].replace(' ', '\n')
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Planet', 'Avg\nScore', 'Best\nScore', 'Score\nRange', 'Strongest\nCriterion'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.set_title('Weather Rubric Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    def generate_weather_reports(self):
        """Generate detailed weather reports for each planet."""
        print("\n" + "="*70)
        print("GENERATING DETAILED WEATHER REPORTS")
        print("="*70)
        
        reports = {}
        
        for planet in self.rubric_df['Planet'].unique():
            print(f"\n{planet} Weather Assessment Report")
            print("-" * 50)
            
            planet_data = self.rubric_df[self.rubric_df['Planet'] == planet]
            
            # Overall assessment
            avg_score = planet_data['Overall_Weather_Score'].mean()
            best_scenario = planet_data.loc[planet_data['Overall_Weather_Score'].idxmax()]
            
            print(f"Overall Weather Rating: {avg_score:.1f}/100")
            print(f"Best Scenario: {best_scenario['Scenario']} ({best_scenario['Overall_Weather_Score']:.1f}/100)")
            
            # Criterion breakdown
            print("\nWeather Criterion Analysis:")
            criterion_cols = ['Temperature Stability', 'Atmospheric Pressure', 'Wind Patterns', 
                             'Humidity Levels', 'Habitability Potential', 'Atmospheric Dynamics']
            
            for criterion in criterion_cols:
                avg_criterion_score = planet_data[criterion].mean()
                weight = self.scoring_criteria[criterion]['weight']
                print(f"  {criterion}: {avg_criterion_score:.1f}/100 (Weight: {weight:.1%})")
            
            # Weather characteristics
            print(f"\nKey Atmospheric Characteristics:")
            print(f"  Mean Temperature: {best_scenario['Mean_Temperature']:.1f}°C")
            print(f"  Mean Pressure: {best_scenario['Mean_Pressure']:.0f} Pa ({best_scenario['Mean_Pressure']/101325:.2f}x Earth)")
            print(f"  Mean Wind Speed: {best_scenario['Mean_Wind_Speed']:.1f} m/s")
            print(f"  Mean Humidity: {best_scenario['Mean_Humidity']:.1f}%")
            
            # Weather classification
            if avg_score >= 70:
                classification = "EXCELLENT - Highly favorable weather patterns"
            elif avg_score >= 50:
                classification = "GOOD - Favorable weather conditions"
            elif avg_score >= 30:
                classification = "MODERATE - Mixed weather characteristics"
            else:
                classification = "POOR - Challenging weather conditions"
            
            print(f"\nWeather Classification: {classification}")
            
            reports[planet] = {
                'overall_score': avg_score,
                'best_scenario': best_scenario.to_dict(),
                'classification': classification,
                'criterion_scores': {col: planet_data[col].mean() for col in criterion_cols}
            }
        
        # Save detailed reports
        report_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/detailed_weather_reports.txt'
        with open(report_path, 'w') as f:
            f.write("EXOPLANET WEATHER RUBRIC REPORTS\n")
            f.write("=" * 50 + "\n\n")
            for planet, report in reports.items():
                f.write(f"{planet} Weather Assessment\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Score: {report['overall_score']:.1f}/100\n")
                f.write(f"Classification: {report['classification']}\n")
                f.write(f"Best Scenario: {report['best_scenario']['Scenario']}\n\n")
        
        print(f"\n✓ Detailed weather reports saved to: detailed_weather_reports.txt")
        
        return reports

def main():
    """Run the complete weather rubric system analysis."""
    print("="*70)
    print("EXOPLANET WEATHER RUBRIC SYSTEM")
    print("="*70)
    print("Quantitative analysis of exoplanet weather patterns")
    print("Creating comprehensive scoring system for atmospheric comparison")
    
    # Initialize the rubric system
    rubric = ExoplanetWeatherRubric()
    
    # Calculate rubric scores
    rubric_scores = rubric.calculate_planet_rubric_scores()
    
    # Create visualizations
    rubric.create_rubric_visualizations()
    
    # Generate detailed reports
    weather_reports = rubric.generate_weather_reports()
    
    print("\n" + "="*70)
    print("WEATHER RUBRIC ANALYSIS COMPLETED!")
    print("="*70)
    print("Files created:")
    print("- weather_rubric_scores.csv (quantitative scores)")
    print("- weather_rubric_analysis.png (comprehensive visualizations)")
    print("- detailed_weather_reports.txt (detailed assessments)")
    print("\nUse these results to quantitatively compare exoplanet weather patterns!")

if __name__ == "__main__":
    main()
