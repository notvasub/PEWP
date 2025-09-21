import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from planetary_weather_predictor import PlanetaryWeatherPredictor
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality matplotlib parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
sns.set_style("whitegrid", {'axes.spines.right': False, 'axes.spines.top': False})

class AcademicExoplanetVisualizer:
    """
    Creates academic-quality visualizations for exoplanet climate analysis.
    
    This class loads moderate atmospheric scenario data for four well-studied
    exoplanets and creates focused visualizations for comparative planetology
    research.
    """
    
    def __init__(self):
        self.predictor = PlanetaryWeatherPredictor()
        self.exoplanets = ['GJ_1214b', 'LHS_1140b', 'ProximaCentauri_b', 'TRAPPIST-1e']
        self.predictions_data = None
        self.output_dir = '/Users/vasubansal/code/universal_atmospheric_model/plots/academic'
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color palette for consistent planet representation
        self.planet_colors = {
            'GJ_1214b': '#2E86AB',      # Blue - Sub-Neptune
            'LHS_1140b': '#A23B72',     # Purple - Super-Earth  
            'ProximaCentauri_b': '#F18F01',  # Orange - Terrestrial
            'TRAPPIST-1e': '#C73E1D'    # Red - Terrestrial
        }
        
    def load_model_and_data(self):
        """Load the trained model and prediction data for moderate scenarios."""
        print("Loading planetary climate prediction model...")
        self.predictor.load_model('/Users/vasubansal/code/universal_atmospheric_model/data/models/planetary_temp_predictor.pkl')
        
        print("Loading exoplanet prediction data...")
        self.predictions_data = []
        
        for planet in self.exoplanets:
            # Load MID scenario predictions
            file_path = f'/Users/vasubansal/code/universal_atmospheric_model/data/predictions/{planet}_MID_spicy_predictions.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['planet'] = planet
                self.predictions_data.append(df)
                print(f"  Loaded {planet} data: {len(df)} monthly predictions")
        
        if self.predictions_data:
            self.predictions_data = pd.concat(self.predictions_data, ignore_index=True)
            print(f"\nTotal dataset: {len(self.predictions_data)} predictions across {len(self.exoplanets)} exoplanets")
        else:
            raise FileNotFoundError("No prediction data found. Please run predictions first.")
    
    def create_seasonal_temperature_patterns(self):
        """
        Figure 1: Seasonal Temperature Variability in Exoplanetary Atmospheres
        
        Purpose: Examine how different exoplanets exhibit seasonal temperature 
        variations under moderate atmospheric conditions. This addresses questions
        about climate stability and seasonal dynamics on potentially habitable worlds.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                ax.plot(planet_data['month'], planet_data['predicted_temperature'], 
                       marker='o', linewidth=3, markersize=8, 
                       color=self.planet_colors[planet], label=planet, alpha=0.9)
        
        # Add reference zones for habitability assessment
        ax.axhspan(273.15-273.15, 303.15-273.15, alpha=0.15, color='green', 
                  label='Liquid Water Stable (0-30°C)', zorder=0)
        ax.axhspan(-20, 0, alpha=0.1, color='blue', 
                  label='Potential Subsurface Liquid Water', zorder=0)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Surface Temperature (°C)')
        ax.set_title('Seasonal Temperature Patterns in Exoplanetary Atmospheres\nunder Moderate Atmospheric Conditions')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 12.5)
        
        # Add statistical annotations
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            if not planet_data.empty:
                temp_range = planet_data['predicted_temperature'].max() - planet_data['predicted_temperature'].min()
                if temp_range > 1:  # Only annotate if there's meaningful variation
                    ax.annotate(f'ΔT = {temp_range:.1f}°C', 
                              xy=(6, planet_data['predicted_temperature'].mean()),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=10, alpha=0.7,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=self.planet_colors[planet], alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/seasonal_temperature_patterns.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/seasonal_temperature_patterns.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Seasonal Temperature Patterns visualization")
    
    def create_atmospheric_pressure_temperature_relationship(self):
        """
        Figure 2: Atmospheric Pressure-Temperature Relationships
        
        Purpose: Investigate the relationship between atmospheric pressure and
        surface temperature across different exoplanetary environments. This
        addresses fundamental questions about atmospheric greenhouse effects
        and pressure-temperature scaling laws.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                # Convert pressure from Pa to kPa for readability
                pressure_kPa = planet_data['pressure'] / 1000
                
                scatter = ax.scatter(pressure_kPa, planet_data['predicted_temperature'], 
                                   s=100, alpha=0.7, color=self.planet_colors[planet], 
                                   label=planet, edgecolors='black', linewidth=0.5)
                
                # Add trend line if there's variation in pressure
                if pressure_kPa.std() > 1:
                    z = np.polyfit(pressure_kPa, planet_data['predicted_temperature'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(pressure_kPa.min(), pressure_kPa.max(), 100)
                    ax.plot(x_trend, p(x_trend), '--', color=self.planet_colors[planet], 
                           alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Atmospheric Pressure (kPa)')
        ax.set_ylabel('Surface Temperature (°C)')
        ax.set_title('Atmospheric Pressure-Temperature Relationships in Exoplanetary Systems')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add theoretical reference lines
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Water Freezing Point')
        ax.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='Water Boiling Point (1 atm)')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pressure_temperature_relationship.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/pressure_temperature_relationship.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Pressure-Temperature Relationship visualization")
    
    def create_solar_irradiation_climate_analysis(self):
        """
        Figure 3: Solar Irradiation and Climate Response Analysis
        
        Purpose: Examine how variations in stellar irradiation affect surface
        temperatures across different exoplanetary systems. This addresses
        questions about habitable zone boundaries and stellar-planetary 
        energy balance.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with solar constant vs temperature
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                scatter = ax.scatter(planet_data['solar_constant'], planet_data['predicted_temperature'],
                                   s=120, alpha=0.7, color=self.planet_colors[planet], 
                                   label=planet, edgecolors='black', linewidth=0.8)
        
        # Add Earth's solar constant reference
        earth_solar_constant = 1361  # W/m²
        ax.axvline(x=earth_solar_constant, color='green', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Earth Solar Constant (1361 W/m²)')
        
        # Add habitable zone estimates (very rough)
        ax.axvspan(0.8 * earth_solar_constant, 1.4 * earth_solar_constant, alpha=0.1, 
                  color='green', label='Approximate Habitable Zone')
        
        ax.set_xlabel('Stellar Irradiation (W/m²)')
        ax.set_ylabel('Surface Temperature (°C)')
        ax.set_title('Solar Irradiation and Surface Temperature in Exoplanetary Systems')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Calculate and display correlation coefficients
        correlations = {}
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            if not planet_data.empty and len(planet_data) > 2:
                corr = np.corrcoef(planet_data['solar_constant'], planet_data['predicted_temperature'])[0,1]
                correlations[planet] = corr
        
        # Add correlation text box
        corr_text = "Solar-Temperature Correlations:\n"
        for planet, corr in correlations.items():
            corr_text += f"{planet}: r = {corr:.3f}\n"
        
        ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/solar_irradiation_climate.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/solar_irradiation_climate.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Solar Irradiation-Climate Analysis visualization")
    
    def create_atmospheric_dynamics_comparison(self):
        """
        Figure 4: Comparative Atmospheric Dynamics
        
        Purpose: Compare wind speed patterns and atmospheric dynamics across
        different exoplanetary systems. This addresses questions about 
        atmospheric circulation, weather patterns, and climate stability.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left panel: Seasonal wind speed patterns
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                ax1.plot(planet_data['month'], planet_data['wind_speed'], 
                        marker='s', linewidth=3, markersize=6,
                        color=self.planet_colors[planet], label=planet, alpha=0.9)
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Wind Speed (m/s)')
        ax1.set_title('Seasonal Wind Speed Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, 12.5)
        
        # Right panel: Wind speed distribution
        wind_data = []
        labels = []
        colors = []
        
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            if not planet_data.empty:
                wind_data.append(planet_data['wind_speed'].values)
                labels.append(planet)
                colors.append(self.planet_colors[planet])
        
        bp = ax2.boxplot(wind_data, labels=labels, patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='black', linewidth=2))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Wind Speed (m/s)')
        ax2.set_title('Wind Speed Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistical annotations
        for i, planet in enumerate(labels):
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            if not planet_data.empty:
                mean_wind = planet_data['wind_speed'].mean()
                std_wind = planet_data['wind_speed'].std()
                ax2.text(i+1, mean_wind + std_wind + 1, f'μ={mean_wind:.1f}\nσ={std_wind:.1f}', 
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.suptitle('Atmospheric Dynamics in Exoplanetary Systems', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/atmospheric_dynamics.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/atmospheric_dynamics.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Atmospheric Dynamics Comparison visualization")
    
    def create_habitability_assessment(self):
        """
        Figure 5: Exoplanet Habitability Assessment
        
        Purpose: Provide a comprehensive assessment of potential habitability
        based on temperature ranges, atmospheric conditions, and stellar
        irradiation. This addresses fundamental astrobiology questions.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate habitability metrics for each planet
        habitability_data = []
        
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                # Temperature-based habitability score
                temps = planet_data['predicted_temperature']
                
                # Score based on proximity to Earth-like temperatures (0-30°C optimal)
                temp_scores = []
                for temp in temps:
                    if 0 <= temp <= 30:
                        score = 100
                    elif -20 <= temp < 0:
                        score = 80 - abs(temp) * 2  # Gradual decrease for cold temperatures
                    elif 30 < temp <= 50:
                        score = 80 - (temp - 30) * 2  # Gradual decrease for hot temperatures
                    else:
                        score = max(0, 40 - abs(temp - 15) * 2)  # Further from optimal
                    temp_scores.append(max(0, score))
                
                avg_temp_score = np.mean(temp_scores)
                temp_stability = max(0, 100 - temps.std() * 10)  # Penalty for high variability
                
                # Pressure-based score (Earth-like pressure gets higher score)
                pressure = planet_data['pressure'].mean() / 1000  # Convert to kPa
                if 50 <= pressure <= 150:  # Earth-like range
                    pressure_score = 100
                elif 10 <= pressure < 50 or 150 < pressure <= 300:
                    pressure_score = 70
                else:
                    pressure_score = max(0, 50 - abs(pressure - 100))
                
                # Solar irradiation score
                solar = planet_data['solar_constant'].mean()
                earth_solar = 1361
                solar_ratio = solar / earth_solar
                if 0.8 <= solar_ratio <= 1.4:
                    solar_score = 100
                else:
                    solar_score = max(0, 100 - abs(solar_ratio - 1.1) * 100)
                
                # Overall habitability index
                overall_score = (avg_temp_score * 0.4 + temp_stability * 0.2 + 
                               pressure_score * 0.2 + solar_score * 0.2)
                
                habitability_data.append({
                    'planet': planet,
                    'temperature_score': avg_temp_score,
                    'stability_score': temp_stability,
                    'pressure_score': pressure_score,
                    'irradiation_score': solar_score,
                    'overall_score': overall_score,
                    'mean_temp': temps.mean(),
                    'temp_range': temps.max() - temps.min()
                })
        
        # Create bubble chart
        for data in habitability_data:
            planet = data['planet']
            x = data['mean_temp']
            y = data['overall_score']
            size = data['irradiation_score'] * 3  # Size represents solar irradiation suitability
            
            ax.scatter(x, y, s=size, alpha=0.7, color=self.planet_colors[planet], 
                      label=planet, edgecolors='black', linewidth=1)
            
            # Add planet labels
            ax.annotate(planet.replace('_', ' '), (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=11, fontweight='bold')
        
        # Add reference zones
        ax.axvspan(0, 30, alpha=0.15, color='green', label='Optimal Temperature Range')
        ax.axhspan(80, 100, alpha=0.1, color='green', label='High Habitability')
        ax.axhspan(60, 80, alpha=0.1, color='yellow', label='Moderate Habitability')
        ax.axhspan(0, 60, alpha=0.1, color='red', label='Low Habitability')
        
        ax.set_xlabel('Mean Surface Temperature (°C)')
        ax.set_ylabel('Habitability Index (0-100)')
        ax.set_title('Exoplanet Habitability Assessment\n(Bubble size represents solar irradiation suitability)')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add Earth reference point
        ax.scatter(15, 95, s=200, marker='*', color='green', 
                  edgecolors='black', linewidth=1, label='Earth Reference', zorder=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/habitability_assessment.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/habitability_assessment.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Habitability Assessment visualization")
    
    def create_atmospheric_correlation_matrix(self):
        """
        Figure 6: Atmospheric Parameter Correlation Analysis
        
        Purpose: Examine correlations between atmospheric parameters and
        predicted temperatures to understand dominant climate drivers.
        This addresses questions about climate sensitivity and feedback
        mechanisms in exoplanetary atmospheres.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select numerical columns for correlation analysis
        correlation_data = self.predictions_data[['month', 'pressure', 'wind_speed', 
                                                'humidity', 'gravity', 'solar_constant', 
                                                'predicted_temperature']].copy()
        
        # Rename columns for better display
        correlation_data.columns = ['Month', 'Pressure (Pa)', 'Wind Speed (m/s)', 
                                  'Humidity (%)', 'Surface Gravity (m/s²)', 
                                  'Solar Constant (W/m²)', 'Temperature (°C)']
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title('Atmospheric Parameter Correlation Matrix\nfor Exoplanetary Climate Predictions')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/correlation_matrix.pdf', bbox_inches='tight')
        plt.show()
        print("✓ Created: Correlation Matrix visualization")
    
    def create_summary_statistics_table(self):
        """Create and save comprehensive summary statistics for the paper."""
        
        summary_stats = []
        
        for planet in self.exoplanets:
            planet_data = self.predictions_data[self.predictions_data['planet'] == planet]
            
            if not planet_data.empty:
                stats = {
                    'Exoplanet': planet.replace('_', ' '),
                    'Mean_Temperature_C': round(planet_data['predicted_temperature'].mean(), 2),
                    'Temperature_Range_C': round(planet_data['predicted_temperature'].max() - planet_data['predicted_temperature'].min(), 2),
                    'Temperature_StdDev_C': round(planet_data['predicted_temperature'].std(), 3),
                    'Mean_Pressure_kPa': round(planet_data['pressure'].mean() / 1000, 1),
                    'Mean_WindSpeed_ms': round(planet_data['wind_speed'].mean(), 1),
                    'Mean_Humidity_percent': round(planet_data['humidity'].mean(), 1),
                    'Solar_Constant_Wm2': round(planet_data['solar_constant'].mean(), 1),
                    'Surface_Gravity_ms2': round(planet_data['gravity'].iloc[0], 2)
                }
                summary_stats.append(stats)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_stats)
        
        # Save as CSV for data sharing
        summary_df.to_csv(f'{self.output_dir}/exoplanet_climate_summary.csv', index=False)
        
        # Create publication-ready table
        print("\n" + "="*100)
        print("EXOPLANET CLIMATE ANALYSIS - SUMMARY STATISTICS")
        print("="*100)
        print(summary_df.to_string(index=False))
        print("="*100)
        
        return summary_df
    
    def generate_all_visualizations(self):
        """Generate all academic visualizations for the exoplanet climate analysis."""
        
        print("ACADEMIC EXOPLANET CLIMATE VISUALIZATION SUITE")
        print("=" * 80)
        
        # Load data
        self.load_model_and_data()
        
        print(f"\nGenerating individual visualizations...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)
        
        # Generate each visualization
        self.create_seasonal_temperature_patterns()
        self.create_atmospheric_pressure_temperature_relationship()
        self.create_solar_irradiation_climate_analysis()
        self.create_atmospheric_dynamics_comparison()
        self.create_habitability_assessment()
        self.create_atmospheric_correlation_matrix()
        
        # Generate summary statistics
        summary_df = self.create_summary_statistics_table()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION SUITE COMPLETED!")
        print("=" * 80)
        print("\nGenerated files:")
        print("📊 Individual Visualizations (PNG + PDF):")
        print("   • seasonal_temperature_patterns")
        print("   • pressure_temperature_relationship")
        print("   • solar_irradiation_climate")
        print("   • atmospheric_dynamics")
        print("   • habitability_assessment")
        print("   • correlation_matrix")
        print("\n📈 Data Files:")
        print("   • exoplanet_climate_summary.csv")
        print(f"\n📁 All files saved to: {self.output_dir}")
        
        return summary_df

if __name__ == "__main__":
    visualizer = AcademicExoplanetVisualizer()
    summary = visualizer.generate_all_visualizations()
