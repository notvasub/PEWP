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

class SeparateVisualizationSystem:
    """
    Creates individual PNG files for each visualization chart.
    """
    
    def __init__(self):
        """Initialize the visualization system."""
        self.unified_scores = None
        self.plots_dir = '/Users/vasubansal/code/universal_atmospheric_model/plots'
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def load_weather_data(self):
        """Load the unified weather scores."""
        try:
            scores_path = '/Users/vasubansal/code/universal_atmospheric_model/data/predictions/unified_weather_scores.csv'
            self.unified_scores = pd.read_csv(scores_path)
            print(f"Loaded {len(self.unified_scores)} planet scenario scores")
            return True
        except FileNotFoundError:
            print("❌ Weather scores not found. Run weather_analysis_system.py first.")
            return False
    
    def create_overall_rankings_chart(self):
        """Create overall planet rankings chart - SEPARATE PNG."""
        print("Creating overall rankings chart...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
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
        ax.set_yticklabels(all_best['Planet'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Overall Weather Score (0-100)', fontsize=16, fontweight='bold')
        ax.set_title('Weather Score Rankings - All Planets\n(Realistic Earth-Centric Analysis)', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, all_best['Overall_Weather_Score'])):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold', fontsize=12)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Solar System'),
            Patch(facecolor='#4ECDC4', label='Exoplanets')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/01_overall_rankings.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_habitability_comparison_chart(self):
        """Create habitability comparison chart - SEPARATE PNG."""
        print("Creating habitability comparison chart...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get best scenarios
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        combined = pd.concat([solar_system, exo_best])
        combined = combined.sort_values('Habitability Potential', ascending=True)
        
        # Create bars with color based on habitability score
        bars = ax.barh(range(len(combined)), combined['Habitability Potential'],
                      color=plt.cm.RdYlGn([score/100 for score in combined['Habitability Potential']]),
                      alpha=0.8)
        
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined['Planet'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Habitability Score (0-100)', fontsize=16, fontweight='bold')
        ax.set_title('Planetary Habitability Comparison\n(Earth-Centric Realistic Assessment)', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, combined['Habitability Potential'])):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold', fontsize=12)
        
        # Add habitability zones
        ax.axvspan(80, 100, alpha=0.1, color='green', label='High Habitability')
        ax.axvspan(50, 80, alpha=0.1, color='yellow', label='Moderate Habitability')
        ax.axvspan(0, 50, alpha=0.1, color='red', label='Low Habitability')
        ax.legend(loc='lower right', fontsize=12)
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/02_habitability_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_temperature_analysis_chart(self):
        """Create temperature analysis chart - SEPARATE PNG."""
        print("Creating temperature analysis chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get best scenarios for each planet
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        combined = pd.concat([solar_system, exo_best])
        
        # Chart 1: Temperature vs Habitability
        ax1.scatter(combined['Mean_Temperature'], combined['Habitability Potential'],
                   s=200, alpha=0.7, c=combined['Overall_Weather_Score'], cmap='RdYlGn')
        
        # Add planet labels
        for _, row in combined.iterrows():
            ax1.annotate(row['Planet'], 
                        (row['Mean_Temperature'], row['Habitability Potential']),
                        xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
        
        # Add habitability zones
        ax1.axhspan(80, 100, alpha=0.1, color='green', label='High Habitability')
        ax1.axhspan(50, 80, alpha=0.1, color='yellow', label='Moderate Habitability')
        ax1.axvspan(-10, 30, alpha=0.1, color='blue', label='Liquid Water Range')
        
        ax1.set_xlabel('Mean Temperature (°C)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Habitability Score', fontweight='bold', fontsize=14)
        ax1.set_title('Temperature vs Habitability Analysis', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Temperature ranges
        planets = combined['Planet']
        mean_temps = combined['Mean_Temperature']
        temp_stds = combined.get('Temp_Std_Dev', [3] * len(combined))  # Default if not available
        
        bars = ax2.bar(range(len(planets)), mean_temps, 
                      yerr=temp_stds, capsize=5,
                      color=['#FF6B6B' if p in ['Earth', 'Mars'] else '#4ECDC4' for p in planets],
                      alpha=0.7)
        
        ax2.set_xticks(range(len(planets)))
        ax2.set_xticklabels(planets, rotation=45, ha='right', fontweight='bold')
        ax2.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=14)
        ax2.set_title('Mean Temperature with Variability', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Freezing Point')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/03_temperature_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_atmospheric_conditions_chart(self):
        """Create atmospheric conditions chart - SEPARATE PNG."""
        print("Creating atmospheric conditions chart...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get best scenarios
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        combined = pd.concat([solar_system, exo_best])
        
        # Chart 1: Pressure comparison (in Earth atmospheres)
        earth_pressure = 101325
        pressure_ratios = combined['Mean_Pressure'] / earth_pressure
        
        bars1 = ax1.bar(range(len(combined)), pressure_ratios,
                       color=['#FF6B6B' if p in ['Earth', 'Mars'] else '#4ECDC4' for p in combined['Planet']],
                       alpha=0.8)
        ax1.set_xticks(range(len(combined)))
        ax1.set_xticklabels(combined['Planet'], rotation=45, ha='right', fontweight='bold')
        ax1.set_ylabel('Pressure (Earth Atmospheres)', fontweight='bold')
        ax1.set_title('Atmospheric Pressure Comparison', fontweight='bold')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Earth Standard')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars1, pressure_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}', ha='center', fontweight='bold')
        
        # Chart 2: Wind speed comparison
        bars2 = ax2.bar(range(len(combined)), combined['Mean_Wind_Speed'],
                       color=['#FF6B6B' if p in ['Earth', 'Mars'] else '#4ECDC4' for p in combined['Planet']],
                       alpha=0.8)
        ax2.set_xticks(range(len(combined)))
        ax2.set_xticklabels(combined['Planet'], rotation=45, ha='right', fontweight='bold')
        ax2.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax2.set_title('Average Wind Speed Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, speed in zip(bars2, combined['Mean_Wind_Speed']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{speed:.1f}', ha='center', fontweight='bold')
        
        # Chart 3: Humidity comparison
        bars3 = ax3.bar(range(len(combined)), combined['Mean_Humidity'],
                       color=['#FF6B6B' if p in ['Earth', 'Mars'] else '#4ECDC4' for p in combined['Planet']],
                       alpha=0.8)
        ax3.set_xticks(range(len(combined)))
        ax3.set_xticklabels(combined['Planet'], rotation=45, ha='right', fontweight='bold')
        ax3.set_ylabel('Humidity (%)', fontweight='bold')
        ax3.set_title('Atmospheric Humidity Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, humidity in zip(bars3, combined['Mean_Humidity']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{humidity:.1f}', ha='center', fontweight='bold')
        
        # Chart 4: Solar constant comparison (in Earth units)
        earth_solar = 1361
        solar_ratios = combined['Solar_Constant'] / earth_solar
        
        bars4 = ax4.bar(range(len(combined)), solar_ratios,
                       color=['#FF6B6B' if p in ['Earth', 'Mars'] else '#4ECDC4' for p in combined['Planet']],
                       alpha=0.8)
        ax4.set_xticks(range(len(combined)))
        ax4.set_xticklabels(combined['Planet'], rotation=45, ha='right', fontweight='bold')
        ax4.set_ylabel('Solar Constant (Earth Units)', fontweight='bold')
        ax4.set_title('Solar Irradiation Comparison', fontweight='bold')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Earth Standard')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars4, solar_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{ratio:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/04_atmospheric_conditions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_weather_criteria_breakdown_chart(self):
        """Create weather criteria breakdown chart - SEPARATE PNG."""
        print("Creating weather criteria breakdown chart...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get best scenarios
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exo_best = exoplanets.loc[exoplanets.groupby('Planet')['Overall_Weather_Score'].idxmax()]
        
        combined = pd.concat([solar_system, exo_best])
        
        # Create heatmap of all criteria
        criteria_cols = ['Temperature Stability', 'Atmospheric Pressure', 'Wind Patterns', 
                        'Humidity Levels', 'Habitability Potential', 'Atmospheric Dynamics']
        
        heatmap_data = combined.set_index('Planet')[criteria_cols].T
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, cbar_kws={'label': 'Score (0-100)'}, ax=ax,
                   linewidths=0.5)
        
        ax.set_title('Weather Criterion Scores - All Planets\n(Realistic Earth-Centric Assessment)', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Planet', fontweight='bold', fontsize=14)
        ax.set_ylabel('Weather Criteria', fontweight='bold', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/05_weather_criteria_breakdown.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_scenario_comparison_chart(self):
        """Create scenario comparison chart for exoplanets - SEPARATE PNG."""
        print("Creating scenario comparison chart...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get only exoplanets
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        # Create grouped bar chart
        planets = exoplanets['Planet'].unique()
        scenarios = ['HIGH', 'MID', 'LOW']
        
        x = np.arange(len(planets))
        width = 0.25
        
        for i, scenario in enumerate(scenarios):
            scenario_data = []
            for planet in planets:
                planet_scenario = exoplanets[(exoplanets['Planet'] == planet) & 
                                           (exoplanets['Scenario'] == scenario)]
                if len(planet_scenario) > 0:
                    scenario_data.append(planet_scenario['Overall_Weather_Score'].iloc[0])
                else:
                    scenario_data.append(0)
            
            bars = ax.bar(x + i*width, scenario_data, width, 
                         label=scenario, alpha=0.8)
            
            # Add value labels
            for bar, score in zip(bars, scenario_data):
                if score > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{score:.1f}', ha='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Exoplanets', fontweight='bold', fontsize=14)
        ax.set_ylabel('Overall Weather Score (0-100)', fontweight='bold', fontsize=14)
        ax.set_title('Exoplanet Weather Scores by Atmospheric Scenario\n(HIGH/MID/LOW Intensity)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(planets, fontweight='bold')
        ax.legend(title='Atmospheric Scenario', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/06_scenario_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_summary_statistics_chart(self):
        """Create summary statistics chart - SEPARATE PNG."""
        print("Creating summary statistics chart...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate comprehensive summary statistics
        solar_system = self.unified_scores[self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        exoplanets = self.unified_scores[~self.unified_scores['Planet'].isin(['Earth', 'Mars'])]
        
        # Create detailed summary table
        summary_data = [
            ['Category', 'Count', 'Avg Score', 'Best Score', 'Temp Range (°C)', 'Best Planet'],
            ['Solar System', len(solar_system), 
             f"{solar_system['Overall_Weather_Score'].mean():.1f}",
             f"{solar_system['Overall_Weather_Score'].max():.1f}",
             f"{solar_system['Mean_Temperature'].min():.0f} to {solar_system['Mean_Temperature'].max():.0f}",
             solar_system.loc[solar_system['Overall_Weather_Score'].idxmax(), 'Planet']],
            ['Exoplanets', len(exoplanets), 
             f"{exoplanets['Overall_Weather_Score'].mean():.1f}",
             f"{exoplanets['Overall_Weather_Score'].max():.1f}",
             f"{exoplanets['Mean_Temperature'].min():.0f} to {exoplanets['Mean_Temperature'].max():.0f}",
             exoplanets.loc[exoplanets['Overall_Weather_Score'].idxmax(), 'Planet']],
            ['All Planets', len(self.unified_scores),
             f"{self.unified_scores['Overall_Weather_Score'].mean():.1f}",
             f"{self.unified_scores['Overall_Weather_Score'].max():.1f}",
             f"{self.unified_scores['Mean_Temperature'].min():.0f} to {self.unified_scores['Mean_Temperature'].max():.0f}",
             self.unified_scores.loc[self.unified_scores['Overall_Weather_Score'].idxmax(), 'Planet']]
        ]
        
        # Create table
        table = ax.table(cellText=summary_data[1:],
                        colLabels=summary_data[0],
                        cellLoc='center',
                        loc='upper center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        # Color coding for the table
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold')
        
        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if i == 1:  # Solar System row
                    table[(i, j)].set_facecolor('#FFE5E5')
                elif i == 2:  # Exoplanets row  
                    table[(i, j)].set_facecolor('#E5F7F5')
                else:  # All planets row
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        ax.set_title('Weather Analysis Summary Statistics\n(Realistic Earth-Centric Assessment)', 
                     fontsize=18, fontweight='bold', pad=40)
        
        # Add analysis date
        ax.text(0.5, 0.1, f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', transform=ax.transAxes, fontsize=10, style='italic')
        
        plt.tight_layout()
        output_path = f'{self.plots_dir}/07_summary_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def create_all_visualizations(self):
        """Create all visualization charts as separate PNG files."""
        print("\n" + "="*80)
        print("CREATING SEPARATE VISUALIZATION FILES")
        print("="*80)
        
        if not self.load_weather_data():
            return False
        
        # Create all individual charts
        self.create_overall_rankings_chart()
        self.create_habitability_comparison_chart()
        self.create_temperature_analysis_chart()
        self.create_atmospheric_conditions_chart()
        self.create_weather_criteria_breakdown_chart()
        self.create_scenario_comparison_chart()
        self.create_summary_statistics_chart()
        
        print(f"\n✅ ALL VISUALIZATIONS COMPLETED!")
        print(f"📁 All charts saved to: {self.plots_dir}/")
        print(f"🎨 7 separate PNG files created for detailed analysis")
        
        return True

def main():
    """Run the separate visualization system."""
    print("="*80)
    print("SEPARATE VISUALIZATION SYSTEM")
    print("="*80)
    print("Creating individual PNG files for each chart")
    
    # Initialize visualization system
    viz = SeparateVisualizationSystem()
    
    # Create all visualizations
    success = viz.create_all_visualizations()
    
    if success:
        print("\n" + "="*80)
        print("VISUALIZATION SYSTEM COMPLETED!")
        print("="*80)
        print("All charts are now available as separate PNG files in the plots/ directory.")
    else:
        print("\n❌ Visualization creation failed. Check error messages above.")

if __name__ == "__main__":
    main()
