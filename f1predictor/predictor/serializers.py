from rest_framework import serializers

class SafetyCarLapsField(serializers.ListField):
    """Custom field to handle safety car laps input"""
    child = serializers.IntegerField(min_value=1)
    
    def to_internal_value(self, data):
        # Handle different input formats (string, list, etc.)
        if isinstance(data, str):
            # Convert comma-separated string to list of integers
            return [int(lap.strip()) for lap in data.split(',') if lap.strip().isdigit()]
        return super().to_internal_value(data)

class PitstopsField(serializers.DictField):
    """Custom field to handle pitstops data - driver ID to list of lap numbers mapping"""
    def to_internal_value(self, data):
        # Convert string keys to integers and ensure values are lists of integers
        processed_data = {}
        for driver_id, laps in data.items():
            # Convert driver ID to integer
            driver_id_int = int(driver_id)
            
            # Process laps based on format
            if isinstance(laps, str):
                # Handle comma-separated string
                laps_list = [int(lap.strip()) for lap in laps.split(',') if lap.strip().isdigit()]
            elif isinstance(laps, list):
                # Already a list - ensure all items are integers
                laps_list = [int(lap) for lap in laps]
            else:
                raise serializers.ValidationError(f"Pitstop laps must be a list or comma-separated string")
                
            processed_data[driver_id_int] = laps_list
            
        return processed_data

class CompoundsField(serializers.DictField):
    """Custom field to handle compound data - nested dict with driver ID -> lap -> compound"""
    def to_internal_value(self, data):
        # Convert to the nested dictionary structure with integer keys and values
        processed_data = {}
        for driver_id, lap_compounds in data.items():
            # Convert driver ID to integer
            driver_id_int = int(driver_id)
            
            # Process lap-compound mappings
            driver_compounds = {}
            for lap, compound in lap_compounds.items():
                lap_int = int(lap)
                compound_int = int(compound)
                driver_compounds[lap_int] = compound_int
                
            processed_data[driver_id_int] = driver_compounds
            
        return processed_data


class QualifyingTimeSerializer(serializers.Serializer):
    """Serializer for an individual driver's qualifying data"""
    q1milli = serializers.FloatField(required=False, default=100.0)
    q2milli = serializers.FloatField(required=False, default=100.0)
    q3milli = serializers.FloatField(required=False, default=100.0)
    grid = serializers.IntegerField(required=False, default=0)


class QualifyingDataField(serializers.DictField):
    """Custom field to handle qualifying data - driver ID to qualifying times mapping"""
    child = QualifyingTimeSerializer()
    
    def to_internal_value(self, data):
        """Convert string driver IDs to integers and validate qualifying data"""
        processed_data = {}
        
        for driver_id, quali_data in data.items():
            # Convert driver ID to integer
            driver_id_int = int(driver_id)
            
            # Validate qualifying data using the QualifyingTimeSerializer
            serializer = QualifyingTimeSerializer(data=quali_data)
            if not serializer.is_valid():
                raise serializers.ValidationError(f"Invalid qualifying data for driver {driver_id}: {serializer.errors}")
            
            # Store validated data
            processed_data[driver_id_int] = serializer.validated_data
            
        return processed_data
    
class SimulationInputSerializer(serializers.Serializer):
    """Main serializer for handling simulation input data"""
    year = serializers.IntegerField(default=2025)
    round = serializers.IntegerField(required=False, default=1)
    model = serializers.CharField(required=False, default="default")
    simulation_type = serializers.ChoiceField(choices=['default', 'custom'], default='default')
    
    # Custom parameters (only valid if simulation_type is 'custom')
    custom_safety_car_laps = SafetyCarLapsField(required=False)
    custom_pitstops = PitstopsField(required=False)
    custom_compounds = CompoundsField(required=False)
    qualifying_data = QualifyingDataField(required=False)
