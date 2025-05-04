from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import SimulationInputSerializer
from django.conf import settings
import os
import pandas as pd
from .simulation.simulator import run_simulation

class PredictionView(APIView):
    def post(self, request):
        """Handle POST requests for F1 race predictions"""
        serializer = SimulationInputSerializer(data=request.data)
        
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract validated data
        data = serializer.validated_data


        
        # Prepare simulation parameters
        sim_params = {
            'race_data_path': os.path.join(settings.BASE_DIR, 'data', 'laptimestest3.csv'),
            'total_laps': 58,  # Could be made configurable in the future
            'year': data['year'],
            'model_type': data['simulation_type'],  # 'default' or 'custom'
            'model': data['model']
        }

        if 'round' in data:
            # You'll need to pass this to create_selected_track function
            # This can be handled in run_simulation or directly here
            sim_params['selected_round'] = data['round']
        
        # Add custom parameters for custom simulation
        if data['simulation_type'] == 'custom':
            print(f'TRUE: data: {data}')
            if 'custom_safety_car_laps' in data:
                sim_params['custom_safety_car_laps'] = data['custom_safety_car_laps']
            
            if 'custom_pitstops' in data:
                sim_params['custom_pitstops'] = data['custom_pitstops']
                
            if 'custom_compounds' in data:
                sim_params['custom_compounds'] = data['custom_compounds']

            if 'qualifying_data' in data:
                print(f'found custom quali times')
                sim_params['custom_qualifying_times'] = data['qualifying_data']
        
        try:
            # Run simulation
            results = run_simulation(**sim_params)
            
            # Convert DataFrame results to a format suitable for JSON response
            # Assuming results is a pandas DataFrame with driver names as index
            serialized_results = {}
            for driver, lap_times in results.iterrows():
                serialized_results[driver] = [None if pd.isna(value) else float(value) for value in lap_times.tolist()]
            
            return Response({
                'success': True,
                'results': serialized_results,
                'metadata': {
                    'year': data['year'],
                    'selected_round' : data['round'],
                    'model': data['model'],
                    'simulation_type': data['simulation_type']
                }
            })
            
        except Exception as e:
            # Log the error for debugging
            import traceback
            traceback.print_exc()
            
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DriversListView(APIView):
    """API endpoint that provides driver information for a specific year"""
    
    def get(self, request, year, round=None):
        """Handle GET requests for driver information by year and optional round"""
        try:
            # Check if round is required for 2024
            if year == 2024 and round is None:
                return Response(
                    {'error': 'Round parameter is required for year 2024'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            drivers = []
            
            # Special handling for 2024
            if year == 2024:
                # Load drivers from laptimestest3.csv based on round
                laptimes_path = os.path.join(settings.BASE_DIR, 'data', 'laptimestest3.csv')
                drivers_path = os.path.join(settings.BASE_DIR, 'data', 'drivers.csv')
                constructors_path = os.path.join(settings.BASE_DIR, 'data', 'constructors.csv')
                
                if not all(os.path.exists(p) for p in [laptimes_path, drivers_path, constructors_path]):
                    return Response(
                        {'error': 'Required data files not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Load necessary data
                laptimes_df = pd.read_csv(laptimes_path)
                drivers_df = pd.read_csv(drivers_path)
                constructors_df = pd.read_csv(constructors_path)
                
                # Filter laptimes for the specific year and round
                filtered_laptimes = laptimes_df[(laptimes_df['year'] == 2024) & 
                                               (laptimes_df['round'] == int(round))]
                
                if filtered_laptimes.empty:
                    return Response(
                        {'error': f'No data available for year 2024, round {round}'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Get unique driver and constructor IDs
                unique_drivers = filtered_laptimes[['driverId', 'constructorId']].drop_duplicates()
                
                # Create driver list with lookup
                for _, row in unique_drivers.iterrows():
                    driver_id = int(row['driverId'])
                    constructor_id = int(row['constructorId'])
                    
                    # Look up driver name
                    driver_row = drivers_df[drivers_df['driverId'] == driver_id]
                    if not driver_row.empty:
                        driver_ref = driver_row.iloc[0]['driverRef']
                        driver_name = self.format_driver_name(driver_ref)
                    else:
                        driver_name = f"Driver {driver_id}"
                    
                    # Look up constructor name
                    constructor_row = constructors_df[constructors_df['constructorId'] == constructor_id]
                    if not constructor_row.empty:
                        team_name = constructor_row.iloc[0]['name']
                    else:
                        team_name = f"Team {constructor_id}"
                    
                    drivers.append({
                        'id': driver_id,
                        'name': driver_name,
                        'team': team_name
                    })
            else:
                # Original logic for other years
                driver_csv_path = os.path.join(settings.BASE_DIR, 'data', f'{year}_driver_info.csv')
                
                if not os.path.exists(driver_csv_path):
                    return Response(
                        {'error': f'No driver data available for {year}'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Read the CSV file
                df = pd.read_csv(driver_csv_path)
                
                # Transform data to required format
                for _, row in df.iterrows():
                    # Get team name from team code
                    team_name = self.get_team_name(row['team_code'])
                    
                    # Get driver's full name
                    driver_name = self.format_driver_name(row['driverRef'])
                    
                    drivers.append({
                        'id': int(row['driverId']),
                        'name': driver_name,
                        'team': team_name
                    })
            
            return Response(drivers)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def format_driver_name(self, driver_ref):
        """Convert driver_ref to a proper name format"""
        # Example: convert 'max_verstappen' to 'Max Verstappen'
        if '_' in driver_ref:
            parts = driver_ref.split('_')
            return ' '.join(part.capitalize() for part in parts)
        else:
            return driver_ref.capitalize()
    
    def get_team_name(self, team_code):
        """Convert team code to full team name"""
        team_map = {
            'MCL': 'McLaren',
            'RBR': 'Red Bull Racing',
            'MER': 'Mercedes',
            'FER': 'Ferrari',
            'HAA': 'Haas',
            'WIL': 'Williams',
            'AST': 'Aston Martin',
            'ALP': 'Alpine',
            'ATR': 'Racing Bulls',
            'ALF': 'Kick Sauber'
        }
        return team_map.get(team_code, team_code)