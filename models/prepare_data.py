import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def prepare_enhanced_flood_dataset(n_samples=100000, 
                                   output_path='data/processed/flood_training_data.csv'):
    """
    Generate realistic flood prediction dataset with comprehensive features
    
    Features include:
    - Topographic: elevation, slope, aspect, curvature, flow accumulation
    - Hydrological: distance to water, watershed area, drainage density, soil permeability
    - Meteorological: rainfall, antecedent moisture, temperature
    - Land use: urban density, vegetation cover, impervious surface
    - Infrastructure: drainage capacity, flood controls
    - Temporal: season, month
    """
    
    print("="*80)
    print("GENERATING ENHANCED FLOOD DATASET")
    print("="*80)
    print(f"\n📊 Generating {n_samples:,} samples with 18 features...")
    
    np.random.seed(42)
    
    # ==================== GEOGRAPHIC FEATURES ====================
    print("\n[1/5] Geographic features...")
    
    # Elevation (0-500m, lower elevations more flood-prone)
    elevation = np.random.gamma(2, 50, n_samples).clip(0, 500)
    
    # Slope in degrees (0-45°, flatter = more flood risk)
    slope = np.random.gamma(2, 3, n_samples).clip(0, 45)
    
    # Aspect (0-360°, direction slope faces)
    aspect = np.random.uniform(0, 360, n_samples)
    
    # Terrain curvature (-5 to 5, negative=concave/water accumulates)
    curvature = np.random.normal(0, 1.5, n_samples).clip(-5, 5)
    
    # Flow accumulation (logarithmic, higher = more water flows through)
    flow_accumulation = np.random.lognormal(3, 2, n_samples).clip(1, 100000)
    
    # ==================== HYDROLOGICAL FEATURES ====================
    print("[2/5] Hydrological features...")
    
    # Distance to nearest water body (0-5000m)
    distance_to_water = np.random.gamma(3, 300, n_samples).clip(0, 5000)
    
    # Watershed area in km² (larger watersheds = more water)
    watershed_area = np.random.lognormal(2, 1.5, n_samples).clip(0.1, 1000)
    
    # Drainage density (km/km², higher = better drainage)
    drainage_density = np.random.gamma(5, 0.3, n_samples).clip(0, 5)
    
    # Soil permeability (mm/hr, 0=impermeable clay, 100=sandy soil)
    soil_permeability = np.random.gamma(3, 8, n_samples).clip(0.1, 100)
    
    # Topographic Wetness Index (5-20, higher = wetter)
    twi = np.random.normal(10, 3, n_samples).clip(5, 20)
    
    # ==================== METEOROLOGICAL FEATURES ====================
    print("[3/5] Meteorological features...")
    
    # 24-hour rainfall (0-300mm)
    rainfall_24h = np.random.gamma(2, 15, n_samples).clip(0, 300)
    
    # Antecedent rainfall - previous 7 days (0-200mm)
    antecedent_rainfall = np.random.gamma(2, 20, n_samples).clip(0, 200)
    
    # Soil moisture (0-100%, saturation level)
    soil_moisture = np.random.beta(2, 3, n_samples) * 100
    
    # Temperature (°C, affects evapotranspiration)
    temperature = np.random.normal(20, 8, n_samples).clip(-5, 40)
    
    # ==================== LAND USE & INFRASTRUCTURE ====================
    print("[4/5] Land use and infrastructure features...")
    
    # Urban density (0-100%, higher = more impervious surfaces)
    urban_density = np.random.beta(1.5, 4, n_samples) * 100
    
    # Vegetation cover (0-100%, NDVI-based)
    vegetation_cover = np.random.beta(3, 2, n_samples) * 100
    
    # Impervious surface percentage (0-100%)
    impervious_surface = urban_density * 0.7 + np.random.normal(0, 5, n_samples)
    impervious_surface = impervious_surface.clip(0, 100)
    
    # Drainage infrastructure capacity (0-1, 1=excellent)
    drainage_capacity = np.random.beta(3, 2, n_samples)
    
    # ==================== TEMPORAL FEATURES ====================
    # Season (1=winter, 2=spring, 3=summer, 4=fall)
    season = np.random.randint(1, 5, n_samples)
    
    # Month (1-12)
    month = np.random.randint(1, 13, n_samples)
    
    # ==================== CREATE FLOOD TARGET ====================
    print("[5/5] Generating flood events based on realistic physics...")
    
    # Flood probability based on multiple interacting factors
    flood_score = np.zeros(n_samples)
    
    # Topographic risk (lower elevation + flat slope + concave = higher risk)
    topo_risk = (500 - elevation) / 500 * 0.2
    topo_risk += (45 - slope) / 45 * 0.15
    topo_risk += np.maximum(-curvature, 0) / 5 * 0.1
    
    # Hydrological risk
    hydro_risk = np.log10(flow_accumulation + 1) / 5 * 0.15
    hydro_risk += (5000 - distance_to_water) / 5000 * 0.15
    hydro_risk += twi / 20 * 0.1
    hydro_risk += (100 - soil_permeability) / 100 * 0.1
    
    # Meteorological risk (rainfall + saturated soil = flooding)
    rainfall_risk = rainfall_24h / 300 * 0.25
    rainfall_risk += antecedent_rainfall / 200 * 0.15
    rainfall_risk += soil_moisture / 100 * 0.15
    
    # Land use risk (urban + impervious = poor drainage)
    land_risk = impervious_surface / 100 * 0.2
    land_risk += (100 - vegetation_cover) / 100 * 0.1
    land_risk += (1 - drainage_capacity) * 0.15
    
    # Combine all risk factors
    flood_score = topo_risk + hydro_risk + rainfall_risk + land_risk
    flood_score += np.random.normal(0, 0.1, n_samples)  # Add noise
    
    # Create binary flood target (threshold at ~30% flood rate)
    flood_threshold = np.percentile(flood_score, 70)
    flood_severity = (flood_score > flood_threshold).astype(int)
    
    # ==================== CREATE DATAFRAME ====================
    df = pd.DataFrame({
        # Topographic
        'elevation_m': elevation,
        'slope_deg': slope,
        'aspect_deg': aspect,
        'curvature': curvature,
        'flow_accumulation': flow_accumulation,
        
        # Hydrological  
        'distance_to_water_m': distance_to_water,
        'watershed_area_km2': watershed_area,
        'drainage_density': drainage_density,
        'soil_permeability_mm_hr': soil_permeability,
        'topographic_wetness_index': twi,
        
        # Meteorological
        'rainfall_24h_mm': rainfall_24h,
        'antecedent_rainfall_7d_mm': antecedent_rainfall,
        'soil_moisture_pct': soil_moisture,
        'temperature_c': temperature,
        
        # Land use & Infrastructure
        'urban_density_pct': urban_density,
        'vegetation_cover_pct': vegetation_cover,
        'impervious_surface_pct': impervious_surface,
        'drainage_capacity': drainage_capacity,
        
        # Temporal
        'season': season,
        'month': month,
        
        # Target
        'flood_severity': flood_severity
    })
    
    flood_count = flood_severity.sum()
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Flood events: {flood_count:,} ({flood_count/n_samples*100:.1f}%)")
    print(f"  Non-flood: {n_samples-flood_count:,} ({(n_samples-flood_count)/n_samples*100:.1f}%)")
    print(f"  Total features: {len(df.columns)-1}")
    
    return df
    
    # Slope (flatter in flood-prone areas)
    df['slope_deg'] = (
        5 + np.random.normal(0, 2, len(df)) -
        df['total_area_flooded_sq_km'] * 0.1
    ).clip(0.01, 30)
    
    # River proximity
    df['river_proximity'] = (df['total_area_flooded_sq_km'] > 5).astype(int)
    
    # Rainfall proxy (based on flood area)
    df['rainfall_mm'] = (
        df['total_area_flooded_sq_km'] * 20 + 
        np.random.normal(0, 50, len(df))
    ).clip(0, 1500)
    
    # Select final features
    print("\n[5/6] Selecting features...")
    final_features = ['elevation_m', 'rainfall_mm', 'river_proximity', 
                      'slope_deg', 'flood_severity']
    
    training_data = df[final_features].dropna()
    
    print(f"✓ Final dataset: {training_data.shape}")
    
    # Save
    print(f"\n[6/6] Saving to {output_path}...")
    training_data.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total samples: {len(training_data):,}")
    print(f"  Features: elevation_m, rainfall_mm, river_proximity, slope_deg")
    print(f"  Target: flood_severity (0=low, 1=high)")
    print(f"  High severity: {training_data['flood_severity'].sum():,} ({training_data['flood_severity'].mean()*100:.1f}%)")
    
    return training_data


if __name__ == "__main__":
    # Generate enhanced dataset
    df = prepare_enhanced_flood_dataset(n_samples=100000)
    
    # Save dataset
    output_path = 'data/processed/flood_training_data.csv'
    print(f"\n💾 Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*80)
    
    print("\n📊 Dataset Summary:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Features: {len(df.columns)-1}")
    print(f"  Target variable: flood_severity")
    
    print("\n📈 Sample Statistics:")
    print(df.describe().round(2))
    
    print("\n🎯 Feature Correlations with flood_severity:")
    correlations = df.corr()['flood_severity'].sort_values(ascending=False)
    for feature, corr in list(correlations.items())[:10]:
        if feature != 'flood_severity':
            print(f"  {feature:30s}: {corr:6.3f}")
    
    print("\n✨ Top 5 Most Important Features (by correlation):")
    top_features = correlations.drop('flood_severity').abs().sort_values(ascending=False).head(5)
    for i, (feature, corr) in enumerate(top_features.items(), 1):
        print(f"  {i}. {feature}")

