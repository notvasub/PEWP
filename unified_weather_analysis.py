import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from weather_rubric_system import ExoplanetWeatherRubric
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

class UnifiedWeatherAnalysis:
    """
    Unified weather analysis system for Earth, Mars, and all exoplanets.
    Calculates weather rubric scores for all worlds using consistent methodology.
    """
    
    def __init__(self):
        """Initialize the unified weather analysis system."""
        self.earth_scores = None
        self.mars_scores = None
        self.exoplanet_scores = None
        self.unified_scores = None
        
        # Initialize the exoplanet rubric system
        self.rubric_system = ExoplanetWeatherRubric()
        
        # Earth reference values (same as rubric system)
        self.earth_reference = {
            'mean_pressure': 101325,  # Pa
            'mean_wind_speed': 10,    # m/s (global average)
            'mean_humidity': 60,      # % (global average)
            'habitable_temp_range': (-10, 30),  # °C
            'habitable_pressure_range': (50000, 200000),  # Pa
            'temp_std_dev': 15        # °C (seasonal variation)
        }
    
    def load_and_prepare_earth_mars_data(self):
        """Load and prepare Earth and Mars data for weather scoring."""
        print("Loading and preparing Earth and Mars data...")
        
        # Load merged planetary data
        merged_path = '/Users/vasubansal/code/universal_atmospheric_model/merged_planetary_data.csv'
        merged_df = pd.read_csv(merged_path)
        
        # Separate Earth and Mars data
        earth_data = merged_df[merged_df['planet'] == 'Earth'].copy()
        mars_data = merged_df[merged_df['planet'] == 'Mars'].copy()
        
        # Calculate monthly statistics for Earth
        earth_monthly = earth_data.groupby('month').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'pressure': 'mean',
            'wind_speed': 'mean',
            'humidity': 'mean',
            'gravity': 'first',
            'solar_constant': 'first'
        }).round(2)
        
        # Flatten column names
        earth_monthly.columns = ['_'.join(col).strip() if col[1] else col[0] for col in earth_monthly.columns]
        earth_monthly = earth_monthly.rename(columns={
            'temperature_mean': 'Mean_Temperature',
            'temperature_std': 'Temp_Std_Dev',
            'temperature_min': 'Min_Temperature',
            'temperature_max': 'Max_Temperature',
            'pressure_mean': 'Mean_Pressure',
            'wind_speed_mean': 'Mean_Wind_Speed',
            'humidity_mean': 'Mean_Humidity',
            'gravity_first': 'Surface_Gravity',
            'solar_constant_first': 'Solar_Constant'
        })
        
        # Calculate temperature range
        earth_monthly['Temp_Range'] = earth_monthly['Max_Temperature'] - earth_monthly['Min_Temperature']
        earth_monthly['Planet'] = 'Earth'
        earth_monthly['Scenario'] = 'ACTUAL'
        earth_monthly = earth_monthly.reset_index()
        
        # Calculate monthly statistics for Mars
        mars_monthly = mars_data.groupby('month').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'pressure': 'mean',
            'wind_speed': 'mean',
            'humidity': 'mean',
            'gravity': 'first',
            'solar_constant': 'first'
        }).round(2)
        
        # Flatten column names for Mars
        mars_monthly.columns = ['_'.join(col).strip() if col[1] else col[0] for col in mars_monthly.columns]
        mars_monthly = mars_monthly.rename(columns={
            'temperature_mean': 'Mean_Temperature',
            'temperature_std': 'Temp_Std_Dev',
            'temperature_min': 'Min_Temperature',
            'temperature_max': 'Max_Temperature',
            'pressure_mean': 'Mean_Pressure',
            'wind_speed_mean': 'Mean_Wind_Speed',
            'humidity_mean': 'Mean_Humidity',
            'gravity_first': 'Surface_Gravity',
            'solar_constant_first': 'Solar_Constant'
        })
        
        mars_monthly['Temp_Range'] = mars_monthly['Max_Temperature'] - mars_monthly['Min_Temperature']
        mars_monthly['Planet'] = 'Mars'
        mars_monthly['Scenario'] = 'ACTUAL'
        mars_monthly = mars_monthly.reset_index()
        
        print(f"Prepared Earth data: {len(earth_monthly)} monthly records")
        print(f"Prepared Mars data: {len(mars_monthly)} monthly records")
        
        return earth_monthly, mars_monthly
    
    def calculate_weather_scores_for_planet(self, planet_data, planet_name):
        """Calculate weather rubric scores for a single planet's data."""
        # Create a summary row (treating as single scenario)
        summary_data = {
            'Planet': planet_name,
            'Scenario': 'ACTUAL',
            'Mean_Temperature': planet_data['Mean_Temperature'].mean(),
            'Min_Temperature': planet_data['Min_Temperature'].min(),
            'Max_Temperature': planet_data['Max_Temperature'].max(),
            'Temp_Std_Dev': planet_data['Temp_Std_Dev'].mean(),
            'Temp_Range': planet_data['Max_Temperature'].max() - planet_data['Min_Temperature'].min(),
            'Mean_Pressure': planet_data['Mean_Pressure'].mean(),
            'Mean_Wind_Speed': planet_data['Mean_Wind_Speed'].mean(),
            'Mean_Humidity': planet_data['Mean_Humidity'].mean(),
            'Solar_Constant': planet_data['Solar_Constant'].iloc[0],
            'Surface_Gravity': planet_data['Surface_Gravity'].iloc[0]
        }
        
        # Calculate individual criterion scores using rubric methods
        scores = {}
        scores['Temperature Stability'] = self.rubric_system.calculate_temperature_stability_score(summary_data)
        scores['Atmospheric Pressure'] = self.rubric_system.calculate_atmospheric_pressure_score(summary_data)
        scores['Wind Patterns'] = self.rubric_system.calculate_wind_patterns_score(summary_data)
        scores['Humidity Levels'] = self.rubric_system.calculate_humidity_score(summary_data)
        scores['Habitability Potential'] = self.rubric_system.calculate_habitability_score(summary_data)
        
        # For dynamics, use the variability in the monthly data
        temp_variation = planet_data['Mean_Temperature'].std()
        if temp_variation > 0:
            if 2 <= temp_variation <= 15:
                dynamics_score = 80  # Good seasonal variation
            elif temp_variation < 2:
                dynamics_score = 40  # Low variation
            else:
                dynamics_score = max(20, 80 - (temp_variation - 15) * 3)  # Too much variation
        else:
            dynamics_score = 30  # No variation data
        
        scores['Atmospheric Dynamics'] = dynamics_score
        
        # Calculate weighted overall score
        criterion_weights = {
            'Temperature Stability': 0.20,
            'Atmospheric Pressure': 0.18,
            'Wind Patterns': 0.16,
            'Humidity Levels': 0.14,
            'Habitability Potential': 0.16,
            'Atmospheric Dynamics': 0.16
        }
        
        overall_score = sum(scores[criterion] * criterion_weights[criterion] for criterion in scores.keys())
        
        # Create result dictionary
        result = {
            'Planet': planet_name,
            'Scenario': 'ACTUAL',
            'Overall_Weather_Score': overall_score,
            **scores,
            **summary_data
        }
        
        return result
    
    def calculate_all_weather_scores(self):
        """Calculate weather rubric scores for Earth, Mars, and all exoplanets."""
        print("\n" + "="*80)
        print("UNIFIED WEATHER ANALYSIS - CALCULATING ALL PLANET SCORES")
        print("="*80)
        
        all_scores = []
        
        # 1. Calculate scores for Earth and Mars
        earth_monthly, mars_monthly = self.load_and_prepare_earth_mars_data()
        
        print("\nCalculating Earth weather scores...")
        earth_result = self.calculate_weather_scores_for_planet(earth_monthly, 'Earth')
        all_scores.append(earth_result)
        print(f"Earth Overall Score: {earth_result['Overall_Weather_Score']:.1f}/100")
        
        print("\nCalculating Mars weather scores...")
        mars_result = self.calculate_weather_scores_for_planet(mars_monthly, 'Mars')
        all_scores.append(mars_result)
        print(f"Mars Overall Score: {mars_result['Overall_Weather_Score']:.1f}/100")
        
        # 2. Load existing exoplanet scores
        print("\nLoading exoplanet weather scores...")
        try:
            exoplanet_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/weather_rubric_scores.csv'
            exoplanet_df = pd.read_csv(exoplanet_path)
            
            # Convert to list of dictionaries for consistency
            for _, row in exoplanet_df.iterrows():
                all_scores.append(row.to_dict())
            
            print(f"Loaded {len(exoplanet_df)} exoplanet scenario scores")
            
        except FileNotFoundError:
            print("Exoplanet scores not found. Calculating them now...")
            # Calculate exoplanet scores if they don't exist
            self.rubric_system.calculate_planet_rubric_scores()
            exoplanet_df = pd.read_csv(exoplanet_path)
            for _, row in exoplanet_df.iterrows():
                all_scores.append(row.to_dict())
        
        # 3. Create unified scores DataFrame
        self.unified_scores = pd.DataFrame(all_scores)
        
        # Save unified scores
        unified_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/unified_weather_scores.csv'
        self.unified_scores.to_csv(unified_path, index=False)
        print(f"\n✓ Unified weather scores saved to: unified_weather_scores.csv")
        
        return self.unified_scores
    
    def create_unified_visualizations(self):
        """Create comprehensive visualizations comparing all planets."""
        print("\nCreating unified weather comparison visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Score Comparison - All Planets
        ax1 = plt.subplot(4, 2, 1)
        self.plot_overall_comparison(ax1)
        
        # 2. Earth vs Mars vs Exoplanets Best Scores
        ax2 = plt.subplot(4, 2, 2)
        self.plot_best_vs_solar_system(ax2)
        
        # 3. Criterion Breakdown - Solar System vs Exoplanets
        ax3 = plt.subplot(4, 2, 3)
        self.plot_criterion_comparison(ax3)
        
        # 4. Habitability Analysis
        ax4 = plt.subplot(4, 2, 4)
        self.plot_habitability_analysis(ax4)
        
        # 5. Temperature vs Pressure Scatter
        ax5 = plt.subplot(4, 2, 5)
        self.plot_temp_pressure_scatter(ax5)
        
        # 6. Atmospheric Characteristics Radar
        ax6 = plt.subplot(4, 2, 6)
        self.plot_atmospheric_characteristics(ax6)
        
        # 7. Score Distribution by Planet Type
        ax7 = plt.subplot(4, 2, 7)
        self.plot_score_distributions(ax7)
        
        # 8. Summary Statistics Table
        ax8 = plt.subplot(4, 2, 8)
        self.plot_summary_table(ax8)
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        output_path = '/Users/vasubansal/code/universal_atmospheric_model/plots/unified_weather_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Unified weather analysis saved to: unified_weather_analysis.png")
        
        plt.show()
    
    def plot_overall_comparison(self, ax):
        """Plot overall weather scores for all planets."""
        # Get best score for each planet
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        # Get best score per exoplanet
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        # Combine all best scores
        all_best = pd.concat([solar_system, exo_best])
        all_best = all_best.sort_values('Overall_Weather_Score', ascending=True)
        
        # Create colors - highlight Solar System planets
        colors = ['#FF6B6B' if planet in ['Earth', 'Mars'] else '#4ECDC4' for planet in all_best['Planet']]
        
        bars = ax.barh(range(len(all_best)), all_best['Overall_Weather_Score'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(all_best)))
        ax.set_yticklabels(all_best['Planet'])
        ax.set_xlabel('Overall Weather Score (0-100)', fontweight='bold')
        ax.set_title('Weather Score Rankings - All Planets\n(Solar System vs Exoplanets)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, all_best['Overall_Weather_Score'])):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Solar System'),
            Patch(facecolor='#4ECDC4', label='Exoplanets')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    def plot_best_vs_solar_system(self, ax):
        """Compare best exoplanet scores with Earth and Mars."""
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        # Get overall stats
        solar_avg = solar_system['Overall_Weather_Score'].mean()
        exo_avg = exoplanets['Overall_Weather_Score'].mean()
        exo_best = exoplanets['Overall_Weather_Score'].max()
        
        categories = ['Earth', 'Mars', 'Exoplanet\nAverage', 'Best\nExoplanet']
        scores = [
            solar_system[solar_system['Planet'] == 'Earth']['Overall_Weather_Score'].iloc[0],
            solar_system[solar_system['Planet'] == 'Mars']['Overall_Weather_Score'].iloc[0],
            exo_avg,
            exo_best
        ]
        
        colors = ['#FF6B6B', '#FF8E8E', '#4ECDC4', '#2ECC71']
        bars = ax.bar(categories, scores, color=colors, alpha=0.8)
        
        ax.set_ylabel('Weather Score (0-100)', fontweight='bold')
        ax.set_title('Solar System vs Exoplanet Weather Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{score:.1f}', ha='center', fontweight='bold')
    
    def plot_criterion_comparison(self, ax):
        """Compare weather criteria between planet types."""
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        criteria = ['Temperature Stability', 'Atmospheric Pressure', 'Wind Patterns', 
                   'Humidity Levels', 'Habitability Potential', 'Atmospheric Dynamics']
        
        solar_avg = [solar_system[criterion].mean() for criterion in criteria]
        exo_avg = [exoplanets[criterion].mean() for criterion in criteria]
        
        x = np.arange(len(criteria))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, solar_avg, width, label='Solar System', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, exo_avg, width, label='Exoplanets', color='#4ECDC4', alpha=0.8)
        
        ax.set_ylabel('Average Score (0-100)', fontweight='bold')
        ax.set_title('Weather Criteria Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' ', '\n') for c in criteria], fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    def plot_habitability_analysis(self, ax):
        """Plot habitability analysis across all planets."""
        # Separate by planet type
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        ax.scatter(solar_system['Mean_Temperature'], solar_system['Habitability Potential'],
                  s=150, alpha=0.8, label='Solar System', color='#FF6B6B')
        ax.scatter(exoplanets['Mean_Temperature'], exoplanets['Habitability Potential'],
                  s=100, alpha=0.6, label='Exoplanets', color='#4ECDC4')
        
        # Add planet labels
        for _, row in solar_system.iterrows():
            ax.annotate(row['Planet'], 
                       (row['Mean_Temperature'], row['Habitability Potential']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add habitability zones
        ax.axhspan(70, 100, alpha=0.1, color='green', label='High Habitability')
        ax.axhspan(40, 70, alpha=0.1, color='yellow', label='Moderate Habitability')
        
        ax.set_xlabel('Mean Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Habitability Score (0-100)', fontweight='bold')
        ax.set_title('Temperature vs Habitability Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_temp_pressure_scatter(self, ax):
        """Plot temperature vs pressure for all planets."""
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        # Convert pressure to Earth atmospheres for better visualization
        earth_pressure = 101325
        
        ax.scatter(solar_system['Mean_Temperature'], solar_system['Mean_Pressure'] / earth_pressure,
                  s=200, alpha=0.8, label='Solar System', color='#FF6B6B')
        ax.scatter(exoplanets['Mean_Temperature'], exoplanets['Mean_Pressure'] / earth_pressure,
                  s=100, alpha=0.6, label='Exoplanets', color='#4ECDC4')
        
        # Add planet labels
        for _, row in solar_system.iterrows():
            ax.annotate(row['Planet'], 
                       (row['Mean_Temperature'], row['Mean_Pressure'] / earth_pressure),
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Mean Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Atmospheric Pressure (Earth Atmospheres)', fontweight='bold')
        ax.set_title('Temperature vs Pressure Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_atmospheric_characteristics(self, ax):
        """Plot key atmospheric characteristics."""
        # Get best scenarios
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        combined = pd.concat([solar_system, exo_best])
        
        # Plot wind speed vs humidity
        ax.scatter(combined['Mean_Wind_Speed'], combined['Mean_Humidity'],
                  s=combined['Overall_Weather_Score'] * 3,  # Size by score
                  alpha=0.7, c=combined['Overall_Weather_Score'], cmap='RdYlGn')
        
        # Add planet labels
        for _, row in combined.iterrows():
            ax.annotate(row['Planet'], 
                       (row['Mean_Wind_Speed'], row['Mean_Humidity']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Mean Wind Speed (m/s)', fontweight='bold')
        ax.set_ylabel('Mean Humidity (%)', fontweight='bold')
        ax.set_title('Wind Speed vs Humidity\n(Bubble size = Weather Score)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Weather Score', fontweight='bold')
    
    def plot_score_distributions(self, ax):
        """Plot score distributions."""
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        ax.hist(solar_system['Overall_Weather_Score'], bins=5, alpha=0.7, 
               label='Solar System', color='#FF6B6B', density=True)
        ax.hist(exoplanets['Overall_Weather_Score'], bins=10, alpha=0.7, 
               label='Exoplanets', color='#4ECDC4', density=True)
        
        ax.set_xlabel('Weather Score (0-100)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Weather Score Distributions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_summary_table(self, ax):
        """Create summary statistics table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate summary statistics
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        summary_data = [
            ['Category', 'Count', 'Avg Score', 'Best Score', 'Temp Range'],
            ['Solar System', len(solar_system), 
             f"{solar_system['Overall_Weather_Score'].mean():.1f}",
             f"{solar_system['Overall_Weather_Score'].max():.1f}",
             f"{solar_system['Mean_Temperature'].min():.1f} to {solar_system['Mean_Temperature'].max():.1f}°C"],
            ['Exoplanets', len(exoplanets), 
             f"{exoplanets['Overall_Weather_Score'].mean():.1f}",
             f"{exoplanets['Overall_Weather_Score'].max():.1f}",
             f"{exoplanets['Mean_Temperature'].min():.1f} to {exoplanets['Mean_Temperature'].max():.1f}°C"],
            ['All Planets', len(self.unified_scores),
             f"{self.unified_scores['Overall_Weather_Score'].mean():.1f}",
             f"{self.unified_scores['Overall_Weather_Score'].max():.1f}",
             f"{self.unified_scores['Mean_Temperature'].min():.1f} to {self.unified_scores['Mean_Temperature'].max():.1f}°C"]
        ]
        
        table = ax.table(cellText=summary_data[1:],
                        colLabels=summary_data[0],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        ax.set_title('Weather Analysis Summary Statistics', 
                     fontsize=14, fontweight='bold', pad=20)
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all planet weather scores."""
        print("\n" + "="*80)
        print("COMPREHENSIVE WEATHER ANALYSIS REPORT")
        print("="*80)
        
        # Solar System Analysis
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        print(f"\n🌍 SOLAR SYSTEM PLANETS")
        print("-" * 40)
        for _, planet in solar_system.iterrows():
            print(f"{planet['Planet']:10} | Score: {planet['Overall_Weather_Score']:5.1f}/100")
            print(f"           | Temp: {planet['Mean_Temperature']:6.1f}°C | Pressure: {planet['Mean_Pressure']/101325:.2f}x Earth")
            print(f"           | Habitability: {planet['Habitability Potential']:5.1f}/100 | Stability: {planet['Temperature Stability']:5.1f}/100")
        
        # Exoplanet Analysis  
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        print(f"\n🪐 EXOPLANETS (Best Scenarios)")
        print("-" * 40)
        for _, planet in exo_best.sort_values('Overall_Weather_Score', ascending=False).iterrows():
            print(f"{planet['Planet']:15} | Score: {planet['Overall_Weather_Score']:5.1f}/100 ({planet['Scenario']})")
            print(f"                | Temp: {planet['Mean_Temperature']:6.1f}°C | Pressure: {planet['Mean_Pressure']/101325:.2f}x Earth")
            print(f"                | Habitability: {planet['Habitability Potential']:5.1f}/100 | Stability: {planet['Temperature Stability']:5.1f}/100")
        
        # Overall Rankings
        all_best = pd.concat([solar_system, exo_best]).sort_values('Overall_Weather_Score', ascending=False)
        
        print(f"\n🏆 OVERALL RANKINGS")
        print("-" * 30)
        for i, (_, planet) in enumerate(all_best.iterrows(), 1):
            classification = "EXCELLENT" if planet['Overall_Weather_Score'] >= 70 else "GOOD" if planet['Overall_Weather_Score'] >= 50 else "MODERATE"
            print(f"{i:2}. {planet['Planet']:15} | {planet['Overall_Weather_Score']:5.1f}/100 | {classification}")
        
        # Key Insights
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 20)
        best_planet = all_best.iloc[0]
        worst_planet = all_best.iloc[-1]
        
        print(f"• Best Weather: {best_planet['Planet']} ({best_planet['Overall_Weather_Score']:.1f}/100)")
        print(f"• Most Habitable: {all_best.loc[all_best['Habitability Potential'].idxmax(), 'Planet']} ({all_best['Habitability Potential'].max():.1f}/100)")
        print(f"• Most Stable: {all_best.loc[all_best['Temperature Stability'].idxmax(), 'Planet']} ({all_best['Temperature Stability'].max():.1f}/100)")
        print(f"• Best Winds: {all_best.loc[all_best['Wind Patterns'].idxmax(), 'Planet']} ({all_best['Wind Patterns'].max():.1f}/100)")
        print(f"• Solar System Average: {solar_system['Overall_Weather_Score'].mean():.1f}/100")
        print(f"• Exoplanet Average: {exoplanets['Overall_Weather_Score'].mean():.1f}/100")
        
        # Save report
        report_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/unified_weather_report.txt'
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE WEATHER ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL RANKINGS\n")
            f.write("-" * 30 + "\n")
            for i, (_, planet) in enumerate(all_best.iterrows(), 1):
                f.write(f"{i:2}. {planet['Planet']:15} | {planet['Overall_Weather_Score']:5.1f}/100\n")
        
        print(f"\n✓ Comprehensive report saved to: unified_weather_report.txt")

def main():
    """Run the complete unified weather analysis."""
    print("="*80)
    print("UNIFIED WEATHER ANALYSIS SYSTEM")
    print("="*80)
    print("Comprehensive weather scoring for Earth, Mars, and all exoplanets")
    
    # Initialize analysis system
    analyzer = UnifiedWeatherAnalysis()
    
    # Calculate all weather scores
    unified_scores = analyzer.calculate_all_weather_scores()
    
    # Create comprehensive visualizations
    analyzer.create_unified_visualizations()
    
    # Generate detailed report
    analyzer.generate_comprehensive_report()
    
    print("\n" + "="*80)
    print("UNIFIED WEATHER ANALYSIS COMPLETED!")
    print("="*80)
    print("Files created:")
    print("- unified_weather_scores.csv (complete dataset)")
    print("- unified_weather_analysis.png (comprehensive visualizations)")
    print("- unified_weather_report.txt (detailed analysis)")
    print("\nNow you can quantitatively compare weather across ALL worlds! 🌍🚀")

if __name__ == "__main__":
    main()
