import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

class PlanetaryWeatherPredictor:
    """
    A machine learning model for predicting planetary temperatures.
    
    This class handles data loading, preprocessing, model training, evaluation,
    and prediction. Trains on Earth and Mars data, then predicts exoplanet temperatures
    using HIGH spicy atmospheric conditions for optimal diversity.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.preprocessor = None
        self.feature_names = None
        self.month_mapping = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
    def load_earth_data(self, filepath):
        """
        Load and preprocess Earth weather data (preserving daily granularity).
        
        Args:
            filepath (str): Path to Earth CSV file
            
        Returns:
            pd.DataFrame: Processed Earth data (daily records)
        """
        print("Loading Earth data...")
        
        # Load the data
        earth_df = pd.read_csv(filepath)
        
        # Convert date column to datetime and extract temporal information
        earth_df['valid_time'] = pd.to_datetime(earth_df['valid_time'])
        earth_df['month_num'] = earth_df['valid_time'].dt.month
        earth_df['month'] = earth_df['month_num'].map(self.month_mapping)
        earth_df['day_of_year'] = earth_df['valid_time'].dt.dayofyear
        earth_df['date'] = earth_df['valid_time']
        
        # Rename columns to standard names
        earth_df = earth_df.rename(columns={
            'sp': 'pressure',
            'windspeed': 'wind_speed',
            't2m_f': 'temperature'
        })
        
        # Convert temperature from Fahrenheit to Celsius for consistency with Mars
        earth_df['temperature'] = (earth_df['temperature'] - 32) * 5/9
        
        # Add missing humidity column (Earth data doesn't have humidity in the provided file)
        # Using approximate values for NYC based on climate data with daily variation
        humidity_by_month = {
            'January': 60, 'February': 58, 'March': 56, 'April': 55, 'May': 62,
            'June': 65, 'July': 66, 'August': 67, 'September': 65, 'October': 61,
            'November': 61, 'December': 62
        }
        
        # Add base humidity by month, then add daily variation (±5% random variation)
        np.random.seed(42)  # For reproducible results
        earth_df['humidity_base'] = earth_df['month'].map(humidity_by_month)
        earth_df['humidity'] = earth_df['humidity_base'] + np.random.normal(0, 3, len(earth_df))
        earth_df['humidity'] = earth_df['humidity'].clip(0, 100)  # Keep within 0-100% range
        earth_df = earth_df.drop('humidity_base', axis=1)
        
        # Add planetary constants
        earth_df['gravity'] = 9.807
        earth_df['solar_constant'] = 1361
        earth_df['planet'] = 'Earth'
        
        # Clean up temporary columns but keep date info
        earth_df = earth_df.drop(['valid_time', 'month_num'], axis=1)
        
        print(f"Earth data loaded: {len(earth_df)} daily records")
        return earth_df
    
    def load_mars_data(self, filepath):
        """
        Load and preprocess Mars weather data (preserving daily granularity).
        
        Args:
            filepath (str): Path to Mars CSV file
            
        Returns:
            pd.DataFrame: Processed Mars data (daily records)
        """
        print("Loading Mars data...")
        
        # Load the data
        mars_df = pd.read_csv(filepath)
        
        # Extract month number from season column (e.g., "Month 1" -> 1)
        mars_df['month_num'] = mars_df['season'].str.extract(r'(\d+)').astype(int)
        mars_df['month'] = mars_df['month_num'].map(self.month_mapping)
        
        # Convert terrestrial date and extract temporal information
        mars_df['terrestrial_date'] = pd.to_datetime(mars_df['terrestrial_date'])
        mars_df['day_of_year'] = mars_df['terrestrial_date'].dt.dayofyear
        mars_df['date'] = mars_df['terrestrial_date']
        
        # Calculate average temperature from min and max
        mars_df['temperature'] = (mars_df['min_temp'] + mars_df['max_temp']) / 2
        
        # Add missing humidity column (Mars has no significant humidity)
        mars_df['humidity'] = 0
        
        # Add planetary constants
        mars_df['gravity'] = 3.711
        mars_df['solar_constant'] = 590
        mars_df['planet'] = 'Mars'
        
        # Clean up temporary and unused columns
        columns_to_drop = ['index', 'id', 'terrestrial_date', 'sol', 'ls', 'season', 'month_num', 
                          'min_temp', 'max_temp', 'atmo_opacity']
        mars_df = mars_df.drop([col for col in columns_to_drop if col in mars_df.columns], axis=1)
        
        print(f"Mars data loaded: {len(mars_df)} daily records")
        return mars_df
        
    
    def merge_data(self, earth_df, mars_df):
        """
        Merge Earth and Mars daily datasets.
        
        Args:
            earth_df (pd.DataFrame): Processed Earth daily data
            mars_df (pd.DataFrame): Processed Mars daily data
            
        Returns:
            pd.DataFrame: Combined daily dataset
        """
        print("Merging daily datasets...")
        
        # Concatenate the datasets
        combined_df = pd.concat([earth_df, mars_df], ignore_index=True)
        
        # Reorder columns as specified (including date information for visualizations)
        column_order = ['date', 'day_of_year', 'month', 'pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'planet', 'temperature']
        combined_df = combined_df[column_order]
        
        # Handle any missing values by filling with mean for each parameter
        numeric_columns = ['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'temperature']
        for col in numeric_columns:
            combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
        
        print(f"Combined daily dataset created: {len(combined_df)} records")
        print(f"  - Earth records: {len(earth_df)}")
        print(f"  - Mars records: {len(mars_df)}")
        print("\nDataset summary:")
        print(combined_df.describe())
        
        return combined_df
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            tuple: (X, y, feature_names)
        """
        print("Preparing features...")
        
        # Separate features and target (excluding date columns from features)
        feature_columns = ['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'planet']
        X = df[feature_columns].copy()
        y = df['temperature'].values
        
        # Create preprocessor for features
        numeric_features = ['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant']
        categorical_features = ['planet']
        
        # Standardize numeric features (except month and planet as specified)
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform the features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names for later use
        numeric_names = numeric_features
        categorical_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        # Rename categorical features to be more intuitive
        # OneHotEncoder creates 'planet_Mars' but 'is_mars' is clearer (1=Mars, 0=Earth)
        categorical_names_clean = []
        for name in categorical_names:
            if name == 'planet_Mars':
                categorical_names_clean.append('is_mars')
            else:
                categorical_names_clean.append(name)
        
        self.feature_names = numeric_names + categorical_names_clean
        
        print(f"Features prepared: {X_processed.shape[1]} features")
        print(f"Feature names: {self.feature_names}")
        
        return X_processed, y, self.feature_names
    
    def train_model(self, X, y):
        """
        Train the gradient boosting model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            
        Returns:
            tuple: (trained_model, X_train, X_test, y_train, y_test)
        """
        print("Training model...")
        
        # Split the data (80/20 as specified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Initialize the model with specified hyperparameters
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Perform 5-fold cross-validation
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='r2')
        
        print(f"Training completed!")
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return self.model, X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test, save_predictions=True):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target values
            save_predictions (bool): Whether to save predictions to CSV
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        # Save predictions to CSV if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'actual_temperature': y_test,
                'predicted_temperature': y_pred
            })
            predictions_df.to_csv('/Users/vasubansal/code/universal_atmospheric_model/temperature_predictions.csv', index=False)
            print("Predictions saved to temperature_predictions.csv")
        
        return metrics
    
    def create_visualizations(self, df, X_test, y_test):
        """
        Create all required visualizations.
        
        Args:
            df (pd.DataFrame): Original combined dataset
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target values
        """
        print("Creating visualizations...")
        
        # Create plots directory if it doesn't exist
        plots_dir = '/Users/vasubansal/code/universal_atmospheric_model/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Feature Importances (Horizontal Bar Chart)
        plt.figure(figsize=(12, 8))
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importances - Planetary Temperature Prediction')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predicted vs Actual Temperatures (Line Plot by Month, Colored by Planet)
        y_pred = self.model.predict(X_test)
        
        # Get month and planet info for test data (this is a simplified approach)
        plt.figure(figsize=(14, 8))
        
        # For this plot, we'll use the full dataset predictions
        X_full = self.preprocessor.transform(df[['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'planet']])
        y_full_pred = self.model.predict(X_full)
        
        df_with_pred = df.copy()
        df_with_pred['predicted_temperature'] = y_full_pred
        
        # Create month order for plotting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        plt.figure(figsize=(14, 8))
        for planet in ['Earth', 'Mars']:
            planet_data = df_with_pred[df_with_pred['planet'] == planet]
            
            # Sort by month order
            planet_data['month'] = pd.Categorical(planet_data['month'], categories=month_order, ordered=True)
            planet_data = planet_data.sort_values('month')
            
            plt.plot(planet_data['month'], planet_data['temperature'], 
                    marker='o', linewidth=2, markersize=8, label=f'{planet} (Actual)')
            plt.plot(planet_data['month'], planet_data['predicted_temperature'], 
                    marker='s', linewidth=2, markersize=6, linestyle='--', label=f'{planet} (Predicted)')
        
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.title('Predicted vs Actual Monthly Temperatures by Planet')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/predicted_vs_actual_by_month.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Daily Trends Overlay
        plt.figure(figsize=(16, 10))
        
        colors = {'Earth': 'blue', 'Mars': 'red'}
        
        for planet in ['Earth', 'Mars']:
            planet_data = df_with_pred[df_with_pred['planet'] == planet].copy()
            planet_data = planet_data.sort_values('day_of_year')
            
            # Use a subset of points to avoid overcrowding (every 7th day for better visibility)
            step_size = max(1, len(planet_data) // 100)  # Show ~100 points max per planet
            planet_subset = planet_data.iloc[::step_size]
            
            color = colors[planet]
            
            # Plot actual temperatures
            plt.plot(planet_data['day_of_year'], planet_data['temperature'], 
                    color=color, alpha=0.6, linewidth=1, label=f'{planet} Actual (Daily)')
            
            # Plot predicted temperatures 
            plt.plot(planet_data['day_of_year'], planet_data['predicted_temperature'], 
                    color=color, linestyle='--', alpha=0.8, linewidth=2, label=f'{planet} Predicted (Daily)')
            
            # Add markers at subset points for clarity
            plt.scatter(planet_subset['day_of_year'], planet_subset['temperature'], 
                       color=color, alpha=0.7, s=20, marker='o')
            plt.scatter(planet_subset['day_of_year'], planet_subset['predicted_temperature'], 
                       color=color, alpha=0.7, s=20, marker='s')
            
            # Fill between actual and predicted to show prediction accuracy
            plt.fill_between(planet_data['day_of_year'], 
                           planet_data['temperature'], 
                           planet_data['predicted_temperature'], 
                           color=color, alpha=0.2, label=f'{planet} Prediction Error')
        
        plt.xlabel('Day of Year')
        plt.ylabel('Temperature (°C)')
        plt.title('Daily Temperature Trends - Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, 365)
        
        # Add month labels on x-axis
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(month_starts, month_labels)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/daily_trends_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All visualizations saved to {plots_dir}/")
    
    def save_model(self, filepath=None):
        """
        Save the trained model and preprocessor.
        
        Args:
            filepath (str): Path to save the model
        """
        if filepath is None:
            filepath = '/Users/vasubansal/code/universal_atmospheric_model/planetary_temp_predictor.pkl'
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'month_mapping': self.month_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model and preprocessor.
        
        Args:
            filepath (str): Path to load the model from
        """
        if filepath is None:
            filepath = '/Users/vasubansal/code/universal_atmospheric_model/planetary_temp_predictor.pkl'
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_names = model_data['feature_names']
        self.month_mapping = model_data['month_mapping']
        
        print(f"Model loaded from {filepath}")
    
    def predict_planet_temp(self, features_dict):
        """
        Predict temperature for given features.
        
        Args:
            features_dict (dict): Dictionary containing feature values
                Required keys: pressure, wind_speed, humidity, gravity, solar_constant, planet
                
        Returns:
            float: Predicted temperature in Celsius
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train a model first.")
        
        # Convert dict to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Handle unknown planet categories by mapping to Mars (closest analog)
        if features_dict['planet'] not in ['Earth', 'Mars']:
            features_df['planet'] = 'Mars'
            print(f"Note: Unknown planet '{features_dict['planet']}' mapped to 'Mars' for prediction")
        
        # Preprocess the features
        X = self.preprocessor.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction
    
    def predict_planet_file(self, filepath):
        """
        Predict temperatures for a hypothetical planet from input file.
        
        Args:
            filepath (str): Path to input CSV or JSON file
            
        Returns:
            pd.DataFrame: DataFrame with input features and predictions
        """
        print(f"Loading hypothetical planet data from {filepath}")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train a model first.")
        
        # Load the file
        if filepath.endswith('.csv'):
            planet_df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            planet_df = pd.read_json(filepath)
        else:
            raise ValueError("File format not supported. Use CSV or JSON.")
        
        # Handle missing humidity (fill with zero if not provided)
        if 'humidity' not in planet_df.columns:
            planet_df['humidity'] = 0
        
        # Add planet column if not present (map to Mars as closest analog)
        if 'planet' not in planet_df.columns:
            planet_df['planet'] = 'Mars'
            print("Note: No planet column found, using 'Mars' as analog for predictions")
        
        # Preprocess features
        feature_columns = ['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'planet']
        X = self.preprocessor.transform(planet_df[feature_columns])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Add predictions to the dataframe
        result_df = planet_df.copy()
        result_df['predicted_monthly_temperature'] = predictions
        
        # Save results
        output_path = filepath.replace('.csv', '_predictions.csv').replace('.json', '_predictions.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Create visualization
        self.plot_hypothetical_planet(result_df, filepath)
        
        return result_df
    
    def plot_hypothetical_planet(self, df, original_filepath):
        """
        Create temperature trend plot for hypothetical planet.
        
        Args:
            df (pd.DataFrame): DataFrame with predictions
            original_filepath (str): Original input file path
        """
        plots_dir = '/Users/vasubansal/code/universal_atmospheric_model/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create month order for plotting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        plt.figure(figsize=(12, 8))
        
        # Sort by month order
        df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
        df_sorted = df.sort_values('month')
        
        plt.plot(df_sorted['month'], df_sorted['predicted_monthly_temperature'], 
                marker='o', linewidth=3, markersize=8, color='purple')
        plt.xlabel('Month')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Predicted Monthly Temperature Trends - Hypothetical Planet')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = os.path.basename(original_filepath).split('.')[0]
        plt.savefig(f'{plots_dir}/hypothetical_planet_{filename}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Hypothetical planet visualization saved to plots/")
    
    def predict_exoplanets_high_spicy(self):
        """
        Predict temperatures for all 4 exoplanets using HIGH spicy atmospheric conditions.
        Uses the model trained on Earth/Mars data.
        
        Returns:
            dict: Dictionary containing predictions for each exoplanet
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train a model first.")
        
        print("Predicting exoplanet temperatures using HIGH spicy atmospheric conditions...")
        
        exoplanets = [
            ('GJ_1214b', 'exoplanet_data/GJ_1214b_HIGH_spicy.csv'),
            ('LHS_1140b', 'exoplanet_data/LHS_1140b_HIGH_spicy.csv'),
            ('ProximaCentauri_b', 'exoplanet_data/ProximaCentauri_b_HIGH_spicy.csv'),
            ('TRAPPIST-1e', 'exoplanet_data/TRAPPIST-1e_HIGH_spicy.csv')
        ]
        
        exoplanet_predictions = {}
        
        for planet_name, filepath in exoplanets:
            full_path = f'/Users/vasubansal/code/universal_atmospheric_model/{filepath}'
            try:
                # Load exoplanet atmospheric data
                planet_df = pd.read_csv(full_path)
                
                # Add planet column (treat as Mars analog for prediction)
                planet_df['planet'] = 'Mars'
                
                # Predict temperatures
                feature_columns = ['pressure', 'wind_speed', 'humidity', 'gravity', 'solar_constant', 'planet']
                X_exoplanet = self.preprocessor.transform(planet_df[feature_columns])
                predictions = self.model.predict(X_exoplanet)
                
                # Store results
                result_df = planet_df.copy()
                result_df['planet'] = planet_name  # Restore original planet name
                result_df['predicted_temperature'] = predictions
                
                # Save to file
                output_path = f'/Users/vasubansal/code/universal_atmospheric_model/{planet_name}_HIGH_spicy_predictions.csv'
                result_df.to_csv(output_path, index=False)
                
                exoplanet_predictions[planet_name] = {
                    'data': result_df,
                    'mean_temp': predictions.mean(),
                    'temp_range': [predictions.min(), predictions.max()]
                }
                
                print(f"{planet_name}: Mean temp = {predictions.mean():.1f}°C, Range = [{predictions.min():.1f}, {predictions.max():.1f}]°C")
                
            except Exception as e:
                print(f"Error predicting {planet_name}: {e}")
                continue
        
        print(f"Exoplanet predictions saved to individual CSV files")
        return exoplanet_predictions


def main():
    """
    Main function to run the complete planetary weather prediction pipeline.
    Trains on Earth/Mars data, then provides exoplanet prediction capability.
    """
    print("="*60)
    print("PLANETARY WEATHER PREDICTION MODEL")
    print("="*60)
    
    # Initialize the predictor
    predictor = PlanetaryWeatherPredictor()
    
    # File paths
    earth_data_path = '/Users/vasubansal/code/universal_atmospheric_model/data/earth.csv'
    mars_data_path = '/Users/vasubansal/code/universal_atmospheric_model/data/marsWeather_with_wind.csv'
    
    try:
        # 1. Load and process data
        earth_df = predictor.load_earth_data(earth_data_path)
        mars_df = predictor.load_mars_data(mars_data_path)
        
        # 2. Merge datasets
        combined_df = predictor.merge_data(earth_df, mars_df)
        
        # Save the merged dataset
        combined_df.to_csv('/Users/vasubansal/code/universal_atmospheric_model/merged_planetary_data.csv', index=False)
        print("Merged daily dataset saved to merged_planetary_data.csv")
        
        # 3. Prepare features
        X, y, feature_names = predictor.prepare_features(combined_df)
        
        # 4. Train model
        model, X_train, X_test, y_train, y_test = predictor.train_model(X, y)
        
        # 5. Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # 6. Create visualizations
        predictor.create_visualizations(combined_df, X_test, y_test)
        
        # 7. Save model
        predictor.save_model()
        
        # 8. Predict exoplanet temperatures using HIGH spicy data
        exoplanet_predictions = predictor.predict_exoplanets_high_spicy()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files created:")
        print("- merged_planetary_data.csv (cleaned and merged dataset)")
        print("- temperature_predictions.csv (actual vs predicted)")
        print("- planetary_temp_predictor.pkl (trained model)")
        print("- plots/ directory with all visualizations")
        print("\nModel trained on:")
        print("- Earth daily weather data (1462 records)")
        print("- Mars daily weather data (3037 records)")
        print("\nExoplanet prediction capability available using HIGH spicy conditions:")
        print("- GJ_1214b (Super-Earth with thick atmosphere)")
        print("- LHS_1140b (Rocky planet in habitable zone)")
        print("- ProximaCentauri_b (Closest potentially habitable exoplanet)")
        print("- TRAPPIST-1e (Earth-sized planet in habitable zone)")
        
        return predictor
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    predictor = main()
