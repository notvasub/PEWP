import sys
import argparse
from pathlib import Path

def run_complete_analysis():
    """Run the complete unified weather analysis."""
    print("🌟 Starting Complete Weather Analysis...")
    
    try:
        from unified_weather_analysis import UnifiedWeatherAnalysis
        
        # Initialize and run analysis
        analyzer = UnifiedWeatherAnalysis()
        
        # Calculate all scores
        print("\n📊 Calculating weather scores for all planets...")
        unified_scores = analyzer.calculate_all_weather_scores()
        
        # Create visualizations
        print("\n🎨 Creating comprehensive visualizations...")
        analyzer.create_unified_visualizations()
        
        # Generate report
        print("\n📝 Generating detailed report...")
        analyzer.generate_comprehensive_report()
        
        print("\n✅ COMPLETE ANALYSIS FINISHED!")
        print("Check the 'data/predictions/' and 'plots/' folders for results.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def run_quick_analysis():
    """Run quick analysis without visualizations."""
    print("⚡ Starting Quick Weather Analysis...")
    
    try:
        from unified_weather_analysis import UnifiedWeatherAnalysis
        
        # Initialize and run core analysis only
        analyzer = UnifiedWeatherAnalysis()
        
        # Calculate scores only
        print("\n📊 Calculating weather scores...")
        unified_scores = analyzer.calculate_all_weather_scores()
        
        # Generate basic report
        analyzer.generate_comprehensive_report()
        
        print("\n✅ QUICK ANALYSIS FINISHED!")
        print("Weather scores calculated. Run without flags for full visualizations.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quick analysis: {e}")
        return False

def run_exoplanet_analysis():
    """Run exoplanet analysis and visualizations."""
    print("🪐 Starting Exoplanet Analysis...")
    
    try:
        from exoplanet_analysis_and_visualization import ExoplanetAnalyzer
        
        # Initialize analyzer
        analyzer = ExoplanetAnalyzer()
        
        # Run complete exoplanet analysis
        predictions, summary = analyzer.run_complete_analysis()
        
        print("\n✅ EXOPLANET ANALYSIS FINISHED!")
        print("Check the 'plots/' folder for exoplanet visualizations.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during exoplanet analysis: {e}")
        return False

def run_weather_analysis_system():
    """Run the advanced weather analysis system."""
    print("🌍 Starting Advanced Weather Analysis System...")
    
    try:
        from weather_analysis_system import WeatherAnalysisSystem
        
        # Initialize system
        analyzer = WeatherAnalysisSystem()
        
        # Calculate all weather scores
        unified_scores = analyzer.calculate_all_weather_scores()
        
        print("\n✅ WEATHER ANALYSIS SYSTEM FINISHED!")
        print("Advanced weather scores calculated.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during weather analysis: {e}")
        return False

def run_visualizations():
    """Generate all visualizations."""
    print("🎨 Starting Visualization Generation...")
    
    try:
        from visualization_generator import SeparateVisualizationSystem
        
        # Initialize visualization system
        viz = SeparateVisualizationSystem()
        
        # Create all visualizations
        success = viz.create_all_visualizations()
        
        if success:
            print("\n✅ VISUALIZATIONS GENERATED!")
            print("Check the 'plots/' folder for individual PNG files.")
        else:
            print("\n❌ Visualization generation failed.")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during visualization generation: {e}")
        return False

def run_academic_visualizations():
    """Generate academic-quality visualizations."""
    print("📚 Starting Academic Visualization Generation...")
    
    try:
        from academic_exoplanet_visualizations import AcademicExoplanetVisualizer
        
        # Initialize visualizer
        visualizer = AcademicExoplanetVisualizer()
        
        # Generate all academic visualizations
        summary = visualizer.generate_all_visualizations()
        
        print("\n✅ ACADEMIC VISUALIZATIONS GENERATED!")
        print("Check the 'plots/academic/' folder for publication-ready files.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during academic visualization generation: {e}")
        return False

def show_results_summary():
    """Show a summary of available results."""
    print("\n📁 AVAILABLE RESULTS:")
    print("="*50)
    
    results_files = {
        'data/predictions/unified_weather_scores.csv': 'Complete weather scores for all planets',
        'data/predictions/weather_rubric_scores.csv': 'Exoplanet weather rubric scores',
        'data/predictions/unified_weather_report.txt': 'Comprehensive analysis report',
        'plots/unified_weather_analysis.png': 'Complete visualization suite',
        'plots/weather_rubric_analysis.png': 'Exoplanet-focused visualizations',
        'plots/academic/': 'Academic publication-ready visualizations'
    }
    
    base_path = Path('/Users/vasubansal/code/universal_atmospheric_model')
    
    for file_path, description in results_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
            print(f"   {description}")
        else:
            print(f"❌ {file_path}")
            print(f"   {description} (not found)")
    
    print(f"\n💡 TIP: All files are relative to:")
    print(f"    {base_path}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Universal Atmospheric Model - Main Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Complete weather analysis
  python main.py --quick                   # Quick analysis without plots
  python main.py --exoplanets              # Exoplanet analysis only
  python main.py --weather-system          # Advanced weather analysis system
  python main.py --visualizations          # Generate all visualizations
  python main.py --academic                # Generate academic visualizations
  python main.py --summary                 # Show results summary
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis without visualizations')
    parser.add_argument('--exoplanets', action='store_true',
                       help='Run exoplanet analysis only')
    parser.add_argument('--weather-system', action='store_true',
                       help='Run advanced weather analysis system')
    parser.add_argument('--visualizations', action='store_true',
                       help='Generate all visualizations')
    parser.add_argument('--academic', action='store_true',
                       help='Generate academic visualizations')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of available results')
    
    args = parser.parse_args()
    
    print("🌍 UNIVERSAL ATMOSPHERIC MODEL")
    print("="*50)
    print("Advanced Weather Analysis for Earth, Mars, and Exoplanets")
    
    success = True
    
    if args.summary:
        show_results_summary()
    elif args.quick:
        success = run_quick_analysis()
    elif args.exoplanets:
        success = run_exoplanet_analysis()
    elif args.weather_system:
        success = run_weather_analysis_system()
    elif args.visualizations:
        success = run_visualizations()
    elif args.academic:
        success = run_academic_visualizations()
    else:
        # Default: complete analysis
        success = run_complete_analysis()
        
        # Show summary after completion
        if success:
            show_results_summary()
    
    if success:
        print(f"\n🎉 SUCCESS! Universal Atmospheric Model analysis complete.")
        print(f"💡 Run 'python main.py --help' to see all available options.")
    else:
        print(f"\n❌ Analysis failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
