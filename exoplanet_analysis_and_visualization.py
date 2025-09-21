import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from planetary_weather_predictor import PlanetaryWeatherPredictor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_style("whitegrid")

class ExoplanetAnalyzer:
    """Comprehensive analysis and visualization of exoplanet predictions."""
    
    def __init__(self):
        self.predictor = PlanetaryWeatherPredictor()
        self.all_predictions = None
        self.exoplanet_files = [
            'GJ_1214b_HIGH_spicy.csv', 'GJ_1214b_MID_spicy.csv', 'GJ_1214b_LOW_spicy.csv',
            'LHS_1140b_HIGH_spicy.csv', 'LHS_1140b_MID_spicy.csv', 'LHS_1140b_LOW_spicy.csv',
            'ProximaCentauri_b_HIGH_spicy.csv', 'ProximaCentauri_b_MID_spicy.csv', 'ProximaCentauri_b_LOW_spicy.csv',
            'TRAPPIST-1e_HIGH_spicy.csv', 'TRAPPIST-1e_MID_spicy.csv', 'TRAPPIST-1e_LOW_spicy.csv'
        ]
        self.planets = ['GJ_1214b', 'LHS_1140b', 'ProximaCentauri_b', 'TRAPPIST-1e']
        self.spiciness_levels = ['HIGH', 'MID', 'LOW']
        
    def load_model(self):
        """Load the trained model."""
        print("Loading trained planetary weather prediction model...")
        self.predictor.load_model()
        print("Model loaded successfully!")
        
    def run_all_predictions(self):
        """Run predictions on all exoplanet scenario files."""
        print("\nRunning predictions on all exoplanet scenarios...")
        all_results = []
        
        for filename in self.exoplanet_files:
            filepath = f'/Users/vasubansal/code/universal_atmospheric_model/exoplanet_data/{filename}'
            print(f"  Processing {filename}...")
            
            # Extract planet name and spiciness level from filename
            parts = filename.replace('.csv', '').split('_')
            planet_name = '_'.join(parts[:-2])  # Everything except last 2 parts
            spiciness = parts[-2]  # Second to last part
            
            # Load the scenario data
            scenario_df = pd.read_csv(filepath)
            
            # Add planet column
            scenario_df['planet'] = planet_name
            
            # Run predictions for each month
            predictions = []
            for _, row in scenario_df.iterrows():
                features = {
                    'month': row['month'],
                    'pressure': row['pressure'],
                    'wind_speed': row['wind_speed'],
                    'humidity': row['humidity'],
                    'gravity': row['gravity'],
                    'solar_constant': row['solar_constant'],
                    'planet': planet_name
                }
                
                pred_temp = self.predictor.predict_planet_temp(features)
                predictions.append(pred_temp)
            
            # Add predictions to the dataframe
            scenario_df['predicted_temperature'] = predictions
            scenario_df['spiciness'] = spiciness
            
            # Save individual predictions
            output_path = f'/Users/vasubansal/code/universal_atmospheric_model/{planet_name}_{spiciness}_spicy_predictions.csv'
            scenario_df.to_csv(output_path, index=False)
            
            all_results.append(scenario_df)
        
        # Combine all results
        self.all_predictions = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        self.all_predictions.to_csv('/Users/vasubansal/code/universal_atmospheric_model/all_exoplanet_predictions.csv', index=False)
        print(f"\nCompleted! Processed {len(self.all_predictions)} total predictions across {len(self.exoplanet_files)} scenarios.")
        
        return self.all_predictions
    
    def create_multi_planet_temperature_profiles(self):
        """Create line plots showing temperature profiles for all planets and scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, spiciness in enumerate(['HIGH', 'LOW']):  # Focus on HIGH and LOW for clarity
            ax = axes[i]
            
            for planet in self.planets:
                data = self.all_predictions[
                    (self.all_predictions['planet'] == planet) & 
                    (self.all_predictions['spiciness'] == spiciness)
                ]
                
                if not data.empty:
                    ax.plot(data['month'], data['predicted_temperature'], 
                           marker='o', linewidth=2, label=planet, markersize=6)
            
            ax.set_title(f'Temperature Profiles - {spiciness} Atmospheric Conditions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Temperature (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add reference lines for habitability
            ax.axhspan(0, 30, alpha=0.1, color='green', label='Earth-like range' if i == 0 else '')
            
        # All scenarios comparison
        ax = axes[2]
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            # Plot all spiciness levels for this planet
            for spiciness in self.spiciness_levels:
                data = planet_data[planet_data['spiciness'] == spiciness]
                if not data.empty:
                    alpha = 1.0 if spiciness == 'HIGH' else 0.6
                    linestyle = '-' if spiciness == 'HIGH' else '--' if spiciness == 'MID' else ':'
                    ax.plot(data['month'], data['predicted_temperature'], 
                           label=f'{planet} ({spiciness})', alpha=alpha, linestyle=linestyle)
        
        ax.set_title('All Planets and Atmospheric Scenarios', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Temperature range comparison
        ax = axes[3]
        temp_ranges = []
        labels = []
        
        for planet in self.planets:
            for spiciness in self.spiciness_levels:
                data = self.all_predictions[
                    (self.all_predictions['planet'] == planet) & 
                    (self.all_predictions['spiciness'] == spiciness)
                ]
                if not data.empty:
                    temp_ranges.append(data['predicted_temperature'].values)
                    labels.append(f'{planet}\n({spiciness})')
        
        bp = ax.boxplot(temp_ranges, labels=labels, patch_artist=True)
        
        # Color by planet
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        planet_colors = {}
        for i, planet in enumerate(self.planets):
            planet_colors[planet] = colors[i]
        
        for i, label in enumerate(labels):
            planet = label.split('\n')[0]
            bp['boxes'][i].set_facecolor(planet_colors[planet])
        
        ax.set_title('Temperature Distribution by Planet and Scenario', fontsize=14, fontweight='bold')
        ax.set_ylabel('Temperature (°C)')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/vasubansal/code/universal_atmospheric_model/plots/exoplanet_temperature_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_atmospheric_conditions_heatmap(self):
        """Create correlation heatmaps showing relationships between variables and temperature."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Overall correlation for all planets
        numeric_cols = ['month', 'pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'predicted_temperature']
        corr_matrix = self.all_predictions[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0,0], cbar_kws={'label': 'Correlation'})
        axes[0,0].set_title('Overall Variable Correlations', fontsize=14, fontweight='bold')
        
        # Correlation by planet
        for i, planet in enumerate(['GJ_1214b', 'LHS_1140b'], 1):
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            planet_corr = planet_data[numeric_cols].corr()
            
            ax = axes[0, i] if i == 1 else axes[1, 0]
            sns.heatmap(planet_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title(f'{planet} Variable Correlations', fontsize=14, fontweight='bold')
        
        # Average conditions by planet heatmap
        avg_conditions = self.all_predictions.groupby(['planet', 'spiciness']).agg({
            'pressure': 'mean',
            'wind_speed': 'mean', 
            'humidity': 'mean',
            'solar_constant': 'mean',
            'predicted_temperature': 'mean'
        }).reset_index()
        
        # Pivot for heatmap
        pivot_data = avg_conditions.pivot_table(
            index=['planet'], 
            columns='spiciness', 
            values='predicted_temperature'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', 
                   square=True, ax=axes[1,1], cbar_kws={'label': 'Temperature (°C)'})
        axes[1,1].set_title('Average Temperature by Planet and Scenario', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/vasubansal/code/universal_atmospheric_model/plots/atmospheric_conditions_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_spiciness_level_comparisons(self):
        """Create visualizations comparing different atmospheric scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Temperature comparison by spiciness level
        for i, planet in enumerate(self.planets):
            ax = axes[i//2, i%2]
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            for spiciness in self.spiciness_levels:
                data = planet_data[planet_data['spiciness'] == spiciness]
                if not data.empty:
                    ax.plot(data['month'], data['predicted_temperature'], 
                           marker='o', linewidth=2, label=f'{spiciness} spicy', markersize=6)
            
            ax.set_title(f'{planet} - Atmospheric Scenario Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Temperature (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add temperature statistics text
            temp_stats = planet_data.groupby('spiciness')['predicted_temperature'].agg(['mean', 'std']).round(2)
            stats_text = '\n'.join([f'{spice}: μ={row["mean"]:.1f}°C, σ={row["std"]:.1f}°C' 
                                   for spice, row in temp_stats.iterrows()])
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/Users/vasubansal/code/universal_atmospheric_model/plots/spiciness_level_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_habitability_analysis(self):
        """Create habitability zone and environmental factor analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(24, 15))
        
        # Habitability zone analysis
        ax = axes[0, 0]
        
        # Create temperature ranges for each planet-scenario combination
        temp_data = []
        labels = []
        colors = []
        
        color_map = {'GJ_1214b': 'lightblue', 'LHS_1140b': 'lightgreen', 
                    'ProximaCentauri_b': 'lightcoral', 'TRAPPIST-1e': 'lightyellow'}
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            for spiciness in self.spiciness_levels:
                data = planet_data[planet_data['spiciness'] == spiciness]
                if not data.empty:
                    temp_data.append(data['predicted_temperature'].values)
                    labels.append(f'{planet}\n{spiciness}')
                    colors.append(color_map[planet])
        
        bp = ax.boxplot(temp_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add habitability reference zones
        ax.axhspan(0, 30, alpha=0.2, color='green', label='Earth-like habitability')
        ax.axhspan(-20, 0, alpha=0.2, color='blue', label='Cold but potentially habitable')
        ax.axhspan(30, 50, alpha=0.2, color='orange', label='Hot but potentially habitable')
        
        ax.set_title('Exoplanet Temperature Ranges vs Habitability Zones', fontsize=14, fontweight='bold')
        ax.set_ylabel('Temperature (°C)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right')
        
        # Environmental factor impact - Solar Constant vs Temperature
        ax = axes[0, 1]
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            ax.scatter(planet_data['solar_constant'], planet_data['predicted_temperature'], 
                      label=planet, alpha=0.7, s=60)
        
        ax.set_title('Solar Constant vs Temperature', fontsize=14, fontweight='bold')
        ax.set_xlabel('Solar Constant (W/m²)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pressure vs Temperature
        ax = axes[0, 2]
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            ax.scatter(planet_data['pressure'], planet_data['predicted_temperature'], 
                      label=planet, alpha=0.7, s=60)
        
        ax.set_title('Atmospheric Pressure vs Temperature', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pressure (Pa)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gravity vs Climate Stability
        ax = axes[1, 0]
        climate_stability = []
        gravity_values = []
        planet_names = []
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            if not planet_data.empty:
                # Use temperature standard deviation as a measure of climate variability
                temp_std = planet_data['predicted_temperature'].std()
                gravity = planet_data['gravity'].iloc[0]  # Gravity is constant per planet
                
                climate_stability.append(temp_std)
                gravity_values.append(gravity)
                planet_names.append(planet)
        
        colors = [color_map[planet] for planet in planet_names]
        scatter = ax.scatter(gravity_values, climate_stability, c=colors, s=120, alpha=0.8)
        
        # Add planet labels
        for i, planet in enumerate(planet_names):
            ax.annotate(planet, (gravity_values[i], climate_stability[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_title('Gravity vs Climate Variability', fontsize=14, fontweight='bold')
        ax.set_xlabel('Surface Gravity (m/s²)')
        ax.set_ylabel('Temperature Variability (°C std dev)')
        ax.grid(True, alpha=0.3)
        
        # Wind Speed Distribution by Planet
        ax = axes[1, 1]
        wind_data = []
        wind_labels = []
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            if not planet_data.empty:
                wind_data.append(planet_data['wind_speed'].values)
                wind_labels.append(planet)
        
        bp = ax.boxplot(wind_data, labels=wind_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(list(color_map.values())[i])
        
        ax.set_title('Wind Speed Distribution by Planet', fontsize=14, fontweight='bold')
        ax.set_ylabel('Wind Speed (m/s)')
        ax.grid(True, alpha=0.3)
        
        # Humidity vs Temperature relationship
        ax = axes[1, 2]
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            ax.scatter(planet_data['humidity'], planet_data['predicted_temperature'], 
                      label=planet, alpha=0.7, s=60)
        
        ax.set_title('Humidity vs Temperature', fontsize=14, fontweight='bold')
        ax.set_xlabel('Humidity (%)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/vasubansal/code/universal_atmospheric_model/plots/habitability_and_environmental_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_advanced_atmospheric_modeling_visualizations(self):
        """Create advanced visualizations including radar charts and 3D plots."""
        fig = plt.figure(figsize=(24, 18))
        
        # Create a comprehensive planetary comparison radar chart
        from math import pi
        
        # Calculate average values for each planet across all scenarios
        planet_stats = []
        categories = ['Temperature', 'Pressure', 'Wind Speed', 'Humidity', 'Solar Constant', 'Gravity']
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            stats = [
                planet_data['predicted_temperature'].mean(),
                planet_data['pressure'].mean() / 1000,  # Scale down for visibility
                planet_data['wind_speed'].mean(),
                planet_data['humidity'].mean(),
                planet_data['solar_constant'].mean() / 100,  # Scale down
                planet_data['gravity'].mean()
            ]
            planet_stats.append(stats)
        
        # Normalize values to 0-1 scale for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_stats = scaler.fit_transform(planet_stats)
        
        # Create radar chart
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]  # Complete the circle
        
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, planet in enumerate(self.planets):
            values = normalized_stats[i].tolist()
            values += values[:1]  # Complete the circle
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=planet, color=colors[i])
            ax1.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Planetary Atmospheric Characteristics\n(Normalized Scale)', fontsize=14, fontweight='bold', y=1.08)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Temperature stability comparison
        ax2 = plt.subplot(2, 3, 2)
        stability_data = []
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            # Calculate stability metrics
            temp_range = planet_data['predicted_temperature'].max() - planet_data['predicted_temperature'].min()
            temp_std = planet_data['predicted_temperature'].std()
            
            stability_data.append([temp_range, temp_std])
        
        x_pos = np.arange(len(self.planets))
        width = 0.35
        
        ranges = [data[0] for data in stability_data]
        stds = [data[1] for data in stability_data]
        
        ax2.bar(x_pos - width/2, ranges, width, label='Temperature Range', alpha=0.7)
        ax2.bar(x_pos + width/2, stds, width, label='Temperature Std Dev', alpha=0.7)
        
        ax2.set_xlabel('Planet')
        ax2.set_ylabel('Temperature Variation (°C)')
        ax2.set_title('Climate Stability Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.planets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Seasonal pattern analysis
        ax3 = plt.subplot(2, 3, 3)
        
        # Calculate seasonal amplitude for each planet-scenario
        seasonal_patterns = {}
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            # Group by scenario and calculate seasonal patterns
            for spiciness in self.spiciness_levels:
                scenario_data = planet_data[planet_data['spiciness'] == spiciness]
                if not scenario_data.empty:
                    # Simple seasonal analysis: difference between max and min monthly temps
                    seasonal_amplitude = scenario_data['predicted_temperature'].max() - scenario_data['predicted_temperature'].min()
                    
                    key = f"{planet}_{spiciness}"
                    seasonal_patterns[key] = seasonal_amplitude
        
        # Plot seasonal patterns
        keys = list(seasonal_patterns.keys())
        values = list(seasonal_patterns.values())
        
        # Group by planet for coloring
        colors = []
        for key in keys:
            planet = key.split('_')[0] + '_' + key.split('_')[1] if 'Proxima' in key or 'TRAPPIST' in key else key.split('_')[0]
            if 'GJ_1214b' in planet:
                colors.append('lightblue')
            elif 'LHS_1140b' in planet:
                colors.append('lightgreen')
            elif 'ProximaCentauri' in planet:
                colors.append('lightcoral')
            else:
                colors.append('lightyellow')
        
        bars = ax3.bar(range(len(keys)), values, color=colors, alpha=0.7)
        ax3.set_xlabel('Planet-Scenario')
        ax3.set_ylabel('Seasonal Temperature Range (°C)')
        ax3.set_title('Seasonal Variability by Scenario', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(keys)))
        ax3.set_xticklabels([key.replace('_', '\n') for key in keys], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Atmospheric pressure vs temperature relationship with trend lines
        ax4 = plt.subplot(2, 3, 4)
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            x = planet_data['pressure']
            y = planet_data['predicted_temperature']
            
            ax4.scatter(x, y, label=planet, alpha=0.7, s=50)
            
            # Add trend line
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x.min(), x.max(), 100)
                ax4.plot(x_trend, p(x_trend), linestyle='--', alpha=0.8)
        
        ax4.set_xlabel('Atmospheric Pressure (Pa)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Pressure-Temperature Relationships\nwith Trend Lines', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Wind speed patterns
        ax5 = plt.subplot(2, 3, 5)
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            # Plot wind speed vs month for HIGH scenario
            high_data = planet_data[planet_data['spiciness'] == 'HIGH']
            if not high_data.empty:
                ax5.plot(high_data['month'], high_data['wind_speed'], 
                        marker='o', linewidth=2, label=planet, markersize=6)
        
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Wind Speed (m/s)')
        ax5.set_title('Seasonal Wind Patterns\n(HIGH Atmospheric Conditions)', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Habitability index calculation and visualization
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate simple habitability index based on temperature proximity to Earth-like conditions
        habitability_scores = []
        scenario_labels = []
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            for spiciness in self.spiciness_levels:
                scenario_data = planet_data[planet_data['spiciness'] == spiciness]
                if not scenario_data.empty:
                    # Simple habitability: inverse of distance from Earth-like temperature range (5-25°C)
                    temps = scenario_data['predicted_temperature']
                    
                    # Calculate distance from habitable range
                    habitable_distances = []
                    for temp in temps:
                        if 5 <= temp <= 25:
                            distance = 0  # Perfect score
                        elif temp < 5:
                            distance = 5 - temp
                        else:
                            distance = temp - 25
                        habitable_distances.append(distance)
                    
                    # Habitability score: higher is better
                    avg_distance = np.mean(habitable_distances)
                    habitability_score = max(0, 100 - avg_distance * 2)  # Scale to 0-100
                    
                    habitability_scores.append(habitability_score)
                    scenario_labels.append(f"{planet}\n{spiciness}")
        
        # Create habitability index chart
        bars = ax6.bar(range(len(scenario_labels)), habitability_scores, 
                      color=plt.cm.RdYlGn([score/100 for score in habitability_scores]))
        
        ax6.set_xlabel('Planet-Scenario')
        ax6.set_ylabel('Habitability Index (0-100)')
        ax6.set_title('Exoplanet Habitability Assessment\n(Based on Temperature)', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(scenario_labels)))
        ax6.set_xticklabels(scenario_labels, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # Add color bar for habitability index
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax6)
        cbar.set_label('Habitability Score')
        
        plt.tight_layout()
        plt.savefig('/Users/vasubansal/code/universal_atmospheric_model/plots/advanced_atmospheric_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_statistics_table(self):
        """Create and save a comprehensive summary statistics table."""
        
        # Calculate comprehensive statistics
        summary_stats = []
        
        for planet in self.planets:
            planet_data = self.all_predictions[self.all_predictions['planet'] == planet]
            
            for spiciness in self.spiciness_levels:
                scenario_data = planet_data[planet_data['spiciness'] == spiciness]
                
                if not scenario_data.empty:
                    stats = {
                        'Planet': planet,
                        'Scenario': spiciness,
                        'Mean_Temperature': scenario_data['predicted_temperature'].mean(),
                        'Min_Temperature': scenario_data['predicted_temperature'].min(),
                        'Max_Temperature': scenario_data['predicted_temperature'].max(),
                        'Temp_Std_Dev': scenario_data['predicted_temperature'].std(),
                        'Temp_Range': scenario_data['predicted_temperature'].max() - scenario_data['predicted_temperature'].min(),
                        'Mean_Pressure': scenario_data['pressure'].mean(),
                        'Mean_Wind_Speed': scenario_data['wind_speed'].mean(),
                        'Mean_Humidity': scenario_data['humidity'].mean(),
                        'Solar_Constant': scenario_data['solar_constant'].mean(),
                        'Surface_Gravity': scenario_data['gravity'].iloc[0]
                    }
                    summary_stats.append(stats)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.round(3)
        
        summary_df.to_csv('/Users/vasubansal/code/universal_atmospheric_model/exoplanet_summary_statistics.csv', index=False)
        
        print("\nExoplanet Analysis Summary:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("EXOPLANET ATMOSPHERIC ANALYSIS")
        print("=" * 80)
        
        # Load model and run predictions
        self.load_model()
        predictions = self.run_all_predictions()
        
        # Mark predictions as completed
        
        # Create all visualizations
        print("\nCreating comprehensive visualizations...")
        
        print("  1. Multi-planet temperature profiles...")
        self.create_multi_planet_temperature_profiles()
        
        print("  2. Atmospheric conditions heatmaps...")
        self.create_atmospheric_conditions_heatmap()
        
        print("  3. Spiciness level comparisons...")
        self.create_spiciness_level_comparisons()
        
        print("  4. Habitability analysis...")
        self.create_habitability_analysis()
        
        print("  5. Advanced atmospheric modeling...")
        self.create_advanced_atmospheric_modeling_visualizations()
        
        print("  6. Summary statistics...")
        summary_df = self.create_summary_statistics_table()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED!")
        print("=" * 80)
        print("\nGenerated files:")
        print("- all_exoplanet_predictions.csv (combined predictions)")
        print("- exoplanet_summary_statistics.csv (summary table)")
        print("- plots/exoplanet_temperature_profiles.png")
        print("- plots/atmospheric_conditions_heatmaps.png")
        print("- plots/spiciness_level_comparisons.png")
        print("- plots/habitability_and_environmental_analysis.png")
        print("- plots/advanced_atmospheric_modeling.png")
        print("- Individual prediction files for each planet-scenario combination")
        
        return predictions, summary_df

if __name__ == "__main__":
    analyzer = ExoplanetAnalyzer()
    predictions, summary = analyzer.run_complete_analysis()
