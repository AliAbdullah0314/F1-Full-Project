import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import os

import torch.nn as nn

import math
def custom_scaler(df):
    df_scaled = df.copy()
    
    # Scale specific features
    scaling_factors = {
        'raceId': 100,
        'driverId': 10,
        'constructorId': 10,
        'year': 100,
        'q1milli': 10,
        'q2milli': 10,
        'q3milli': 10,
        'Driver_Season_Points': 10,
        'Races_before': 10,
        'milliseconds_y': 10000
    }
    
    # Apply scaling to main features
    for feature, factor in scaling_factors.items():
        if feature in df_scaled.columns:
            df_scaled[feature] = df_scaled[feature] / factor
    
    # # Apply scaling to P1-P20 prefixed features
    # for i in range(1, 21):
    #     prefix = f'P{i}_'
    #     for feature, factor in scaling_factors.items():
    #         prefixed_feature = f'{prefix}{feature}'
    #         if prefixed_feature in df_scaled.columns:
    #             df_scaled[prefixed_feature] = df_scaled[prefixed_feature] / factor
    
    return df_scaled

def pad_sequence_data_2024_round(sequences, sequence_lengths_2024):
    sequence_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_sequences = pad_sequence(sequence_tensors, batch_first=True)
    return padded_sequences, sequence_lengths_2024

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, d_model=256, nhead=8, 
                 num_encoder_layers=6, dropout=0.1):
        super().__init__()
        
        # Static variable processing
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Variable selection network
        self.var_select = nn.Sequential(
            nn.Linear(input_dim + d_model, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(input_size=input_dim, 
                          hidden_size=d_model, 
                          num_layers=2,
                          batch_first=True,
                          dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Gated residual connections
        self.gate = nn.Sequential(
            nn.Linear(d_model*2, d_model*2),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static, attention_mask=None):
        # Static context
        static_ctx = self.static_encoder(static)  # [batch_size, d_model]
        
        # Variable selection
        static_expanded = static_ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        var_weights = self.var_select(torch.cat([x, static_expanded], dim=-1))
        x_selected = x * var_weights
        
        # Temporal processing
        lstm_out, _ = self.lstm(x_selected)  # [batch_size, seq_len, d_model]
        
        # Static context fusion
        static_fused = torch.cat([
            lstm_out,
            static_ctx.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        ], dim=-1)
        
        # Transformer processing
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
            static_fused = static_fused.masked_fill(padding_mask.unsqueeze(-1), 0)
            
        transformer_out = self.transformer(static_fused, src_key_padding_mask=padding_mask)
        
        # Gated residual
        gate = self.gate(transformer_out)
        residual_out = gate * transformer_out + (1 - gate) * static_fused
        
        # Output projection
        output = self.output_layer(residual_out)
        return output.squeeze(-1)


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # Multiple LSTM layers with dropout
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Dropout layer for the output of the last LSTM layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch normalization layer
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM layers
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=87)
        
        # Apply batch normalization
        output = self.batch_norm(output.transpose(1, 2)).transpose(1, 2)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Apply the fully connected layer
        output = self.fc(output)
        
        # Squeeze the last dimension to match the shape of batch_y
        output = output.squeeze(-1)
        
        return output

class F1RaceSimulator:
    def __init__(self, safety_car_model, pitstop_model, compound_model, laptime_model, device,
                 use_custom_data=False, custom_safety_car_laps=None, custom_pitstops=None, custom_compounds=None, laptime_model_type="TFT"):
        """Initialize the race simulation pipeline with pre-trained models"""
        
        # Initialize existing models
        self.safety_car_model = safety_car_model
        self.pitstop_model = pitstop_model
        self.compound_model = compound_model
        self.laptime_model = laptime_model
        self.device = device
        self.laptime_model_type = laptime_model_type.upper()
        
        # Custom data flags and values
        self.use_custom_data = use_custom_data
        self.custom_safety_car_laps = custom_safety_car_laps or []
        self.custom_pitstops = custom_pitstops or {}
        self.custom_compounds = custom_compounds or {}
        
        # Set feature definitions for each model
        self.safety_car_features = ['raceId','circuitId','driverId','constructorId', 'grid', 'year', 'round', 'lap', 'isSafetyCarPrev']
                                   
        
        self.pitstop_features = ['raceId','circuitId','driverId','constructorId', 'grid', 'year', 'round', 'lap', 'q1milli', 'q2milli', 'q3milli', 
                                                                'Driver_Season_Points', 'driverwins', 'YOB', 'Races_before', 'Races_won', 'Podiums', 'isSafetyCar', 'isSafetyCarPrev', 
                                                                'tyre_age', 'tyre_compound', 'isVET', 'isZHO', 'isVER', 'isTSU', 'isSTR', 'isMSC', 'isSAR', 'isRIC', 
                                                                'isSAI', 'isRUS', 'isPIA', 'isPER', 'isOCO', 'isNOR', 'isMAG', 'isLEC', 'isLAW', 'isLAT', 'isHUL', 'isHAM', 
                                                                'isGAS', 'isDEV', 'isBOT', 'isBEA', 'isALO', 'isALB', 'isRBR', 'isFER', 'isMER', 'isALP', 'isMCL', 'isALF', 
                                                                'isAST', 'isHAA', 'isATR', 'isWIL']
       
        
        self.compound_features = [
            'raceId','circuitId','driverId','constructorId', 'year', 'round', 'lap', 'Driver_Season_Points', 'driverwins', 'YOB', 
            'Races_before', 'Races_won', 'Podiums', 'isSafetyCar', 'isSafetyCarPrev','tyre_age', 'tyre_compound',
            'isVET',	'isZHO',	'isVER',	'isTSU',	'isSTR',	'isMSC',	'isSAR',	'isRIC',	'isSAI',	'isRUS',	'isPIA',	'isPER',	'isOCO',	
            'isNOR',	'isMAG',	'isLEC',	'isLAW',	'isLAT',	'isHUL',	'isHAM',	'isGAS',	'isDEV',	'isBOT',	'isBEA',	'isALO',	'isALB',	
            'isRBR',	'isFER',	'isMER',	'isALP',	'isMCL',	'isALF',	'isAST',	'isHAA',	'isATR',	'isWIL'
        ] 
        
        
        self.temporal_features = ['lap', 'isSafetyCar', 'isSafetyCarPrev', 
                                 'isPitting', 'tyre_age', 'tyre_compound']
                                 
        self.static_features = ['raceId','circuitId','driverId','constructorId', 'grid', 'year', 'round', 'q1milli', 'q2milli', 'q3milli',
        'Driver_Season_Points', 'driverwins', 'YOB', 'Races_before', 'Races_won', 'Podiums', 'isVET', 'isZHO', 'isVER', 'isTSU', 'isSTR', 'isMSC', 'isSAR', 'isRIC',
        'isSAI', 'isRUS', 'isPIA', 'isPER', 'isOCO', 'isNOR', 'isMAG', 'isLEC', 'isLAW', 'isLAT', 'isHUL', 'isHAM',
        'isGAS', 'isDEV', 'isBOT', 'isBEA', 'isALO', 'isALB', 'isRBR', 'isFER', 'isMER', 'isALP', 'isMCL', 'isALF',
        'isAST', 'isHAA', 'isATR', 'isWIL']
        

    def simulate_race(self, initial_data, total_laps):
        """Run a complete race simulation with separated laptime prediction"""
        # Initialize race data
        race_data = initial_data.copy()
        
        # Create a list to store race states for each lap
        race_states = []
        
        # Add initial values if missing
        race_data['tyre_age'] = race_data.get('tyre_age', 1)
        race_data['isSafetyCar'] = 0
        race_data['isSafetyCarPrev'] = 0
        race_data['isPitting'] = 0
        
        # PHASE 1: Predict safety car, pitstops, and compounds for all laps
        print("Phase 1: Predicting race events (safety car, pitstops, and compounds)")
        
        for lap in range(1, total_laps + 1):
            print(f"Processing lap {lap}/{total_laps}")
            
            # Update lap number
            race_data['lap'] = lap
            
            # Determine safety car status for this lap
            if self.use_custom_data:
                # Use custom safety car data
                sc_deployed = 1 if lap in self.custom_safety_car_laps else 0
            else:
                # Use prediction model
                sc_predictions = self.predict_safety_car(race_data)
                sc_deployed = 1 if sc_predictions.mean() > 0.5 else 0
            
            race_data['isSafetyCar'] = sc_deployed
            
            # Determine pitstops for all drivers
            if self.use_custom_data:
                # Use custom pitstop data
                for idx, row in race_data.iterrows():
                    driver_id = row['driverId']
                    race_data.loc[idx, 'isPitting'] = 1 if (driver_id in self.custom_pitstops and 
                                                        lap in self.custom_pitstops[driver_id]) else 0
            else:
                # Use prediction model
                race_data['isPitting'] = self.predict_pitstops(race_data)
            
            # Determine compounds for pitting drivers
            for idx, row in race_data[race_data['isPitting'] == 1].iterrows():
                driver_id = row['driverId']
                if self.use_custom_data and driver_id in self.custom_compounds and lap in self.custom_compounds[driver_id]:
                    # Use custom compound data
                    race_data.loc[idx, 'next_compound'] = self.custom_compounds[driver_id][lap]
                else:
                    # Use prediction model
                    race_data.loc[idx, 'next_compound'] = self.predict_compound(row)
            
            # Store this lap's data (without laptimes)
            race_states.append(race_data.copy())
            
            # Update for next lap
            race_data['isSafetyCarPrev'] = race_data['isSafetyCar']
            
            # Update tire age and compound
            for idx in race_data.index:
                if race_data.loc[idx, 'isPitting'] == 1:
                    # Reset tire age for next lap
                    race_data.loc[idx, 'tyre_age'] = 1
                    
                    # Apply new compound if predicted
                    if pd.notna(race_data.loc[idx, 'next_compound']):
                        race_data.loc[idx, 'tyre_compound'] = race_data.loc[idx, 'next_compound']
                else:
                    # Increment tire age
                    race_data.loc[idx, 'tyre_age'] += 1
            
            # Clear pitting and next compound for next lap
            race_data['isPitting'] = 0
            race_data['next_compound'] = np.nan
        # ----------------------
        # NEW CODE: Create standardized DataFrame matching laptimestest3.csv format
        # ----------------------
        print("Creating standardized race data DataFrame...")
        
        # Combine all race states into a single DataFrame
        combined_race_data = pd.concat(race_states, ignore_index=True)
        
        # Ensure all required columns are present
        required_columns = [
            'raceId', 'circuitId', 'driverId', 'constructorId', 'grid', 'year', 'round', 
            'lap', 'milliseconds_y', 'q1milli', 'q2milli', 'q3milli', 
            'Driver_Season_Points', 'driverwins', 'YOB', 'Races_before', 'Races_won', 
            'Podiums', 'isSafetyCar', 'isSafetyCarPrev', 'isPitting', 'tyre_age', 'tyre_compound'
        ]
        
        # Get all driver indicator columns (isVER, isHAM, etc.)
        driver_indicators = [col for col in combined_race_data.columns if col.startswith('is') 
                            and not col in ['isSafetyCar', 'isSafetyCarPrev', 'isPitting']]
        
        # Combine all required columns
        all_required_columns = required_columns + driver_indicators
        
        # Initialize milliseconds_y as NaN (will be filled later)
        if 'milliseconds_y' not in combined_race_data.columns:
            combined_race_data['milliseconds_y'] = np.nan
        
        # Ensure all required columns exist
        for col in all_required_columns:
            if col not in combined_race_data.columns:
                # If a column is missing, check if it's in initial_data
                if col in initial_data.columns:
                    # Copy static data from initial data to all rows
                    driver_values = {}
                    for idx, row in initial_data.iterrows():
                        driver_values[row['driverId']] = row[col]
                    
                    # Apply the values to combined_race_data based on driverId
                    combined_race_data[col] = combined_race_data['driverId'].map(driver_values)
                else:
                    # If column is not in initial_data, fill with zeros or appropriate default
                    combined_race_data[col] = 0
        
        # Sort the DataFrame for a clean structure: first by lap, then by grid position
        standardized_race_data = combined_race_data.sort_values(by=['lap', 'grid']).reset_index(drop=True)
        
        # Ensure columns are in the right order
        column_order = all_required_columns.copy()
        # Add any remaining columns not in the required list
        for col in standardized_race_data.columns:
            if col not in column_order:
                column_order.append(col)
        
        # Reorder columns to match standard format
        standardized_race_data = standardized_race_data[column_order]

        # print(standardized_race_data.head)
        standardized_race_data.to_csv('standardized_race_data.csv')
        
        
        
        
        # # PHASE 2: Predict all lap times at once
        # print("Phase 2: Predicting lap times for all drivers and laps")
        combined_df_2024_round1 = standardized_race_data.sort_values(by=['raceId', 'driverId', 'lap'])
        print(f'combined_df_2024_round1:{combined_df_2024_round1.head}')


        # Select round 1 from the 2024 season
        # combined_df_2024_round1 = combined_df_2024[(combined_df_2024['year'] == 2024) & 
        #                                         (combined_df_2024['round'] == 1)]
        
        # Get driver IDs for later
        driver_ids = combined_df_2024_round1.groupby(['raceId', 'driverId'])\
                            .apply(lambda group: group['driverId'].iloc[0]).values

        # Scale the data
        combined_df_2024_round1 = custom_scaler(combined_df_2024_round1)
        

        if self.laptime_model_type == "TFT":
            # Form temporal and static sequences
            temporal_sequences = []
            static_data = []
            sequence_lengths_2024 = []

            temporal_features = [ 'lap', 'isSafetyCar', 'isSafetyCarPrev', 'isPitting', 'tyre_age',
                                                    'tyre_compound']
            static_features = ['raceId','circuitId','driverId','constructorId', 'grid', 'year', 'round', 'q1milli', 'q2milli', 'q3milli',
    'Driver_Season_Points', 'driverwins', 'YOB', 'Races_before', 'Races_won', 'Podiums', 'isVET', 'isZHO', 'isVER', 'isTSU', 'isSTR', 'isMSC', 'isSAR', 'isRIC',
    'isSAI', 'isRUS', 'isPIA', 'isPER', 'isOCO', 'isNOR', 'isMAG', 'isLEC', 'isLAW', 'isLAT', 'isHUL', 'isHAM',
    'isGAS', 'isDEV', 'isBOT', 'isBEA', 'isALO', 'isALB', 'isRBR', 'isFER', 'isMER', 'isALP', 'isMCL', 'isALF',
    'isAST', 'isHAA', 'isATR', 'isWIL']

            for (raceId, driverId), group in combined_df_2024_round1.groupby(['raceId', 'driverId']):
                group = group.sort_values(by='lap')
                
                # Extract temporal and static features separately
                temporal_seq = group[temporal_features].values
                static_seq = group[static_features].iloc[0].values
                
                temporal_sequences.append(temporal_seq)
                static_data.append(static_seq)
                sequence_lengths_2024.append(len(temporal_seq))

            # Convert to tensors and pad
            temporal_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in temporal_sequences]
            static_tensors = [torch.tensor(static, dtype=torch.float32) for static in static_data]

            # print(f'temporal_tensors: {temporal_tensors}')
            
            padded_temporal = pad_sequence(temporal_tensors, batch_first=True)
            static_tensor = torch.stack(static_tensors)
            
            # Create TFT-specific test dataset
            class TestTFTDatasetRound(Dataset):
                def __init__(self, temporal, static, lengths):
                    self.temporal = temporal
                    self.static = static
                    self.lengths = lengths
                    self.max_seq_length = temporal.size(1)
                    
                    self.attention_masks = torch.zeros((len(temporal), self.max_seq_length), dtype=torch.float32)
                    for idx, length in enumerate(lengths):
                        self.attention_masks[idx, :length] = 1

                def __len__(self):
                    return len(self.temporal)

                def __getitem__(self, i):
                    return self.temporal[i], self.static[i], self.lengths[i], self.attention_masks[i]
            
            test_dataset_2024 = TestTFTDatasetRound(padded_temporal, static_tensor, sequence_lengths_2024)
            
            # Generate predictions
            self.laptime_model.eval()
            all_predictions = []

            with torch.no_grad():
                for i in range(len(test_dataset_2024)):
                    temporal_feat, static_feat, length, attn_mask = test_dataset_2024[i]
                    
                    # Move to device and add batch dimension
                    temporal_feat = temporal_feat.to(self.device).unsqueeze(0)
                    static_feat = static_feat.to(self.device).unsqueeze(0)
                    attn_mask = attn_mask.to(self.device).unsqueeze(0)
                    
                    # Forward pass with both temporal and static features
                    outputs = self.laptime_model(temporal_feat, static_feat, attn_mask)
                    
                    # Extract valid predictions
                    valid_outputs = outputs[0, :length]
                    all_predictions.append(valid_outputs.cpu())
            
            # Process predictions
            all_predictions = [pred.numpy() for pred in all_predictions]
            masked_preds = []
            
            for i in range(len(test_dataset_2024)):
                sample_length = test_dataset_2024[i][2]
                if not isinstance(sample_length, int):
                    sample_length = sample_length.item()
                    
                masked_output = all_predictions[i][:sample_length]
                masked_preds.append(masked_output)

        else:
            print('PREDICTING WITH LSTM')
            features = ['raceId','circuitId','driverId','constructorId', 'grid', 'year', 'round', 'lap', 'q1milli', 'q2milli', 'q3milli', 
                                        'Driver_Season_Points', 'driverwins', 'YOB', 'Races_before', 'Races_won', 'Podiums', 'isSafetyCar', 'isSafetyCarPrev', 
                                        'isPitting','tyre_age', 'tyre_compound', 'isVET', 'isZHO', 'isVER', 'isTSU', 'isSTR', 'isMSC', 'isSAR', 'isRIC', 
                                        'isSAI', 'isRUS', 'isPIA', 'isPER', 'isOCO', 'isNOR', 'isMAG', 'isLEC', 'isLAW', 'isLAT', 'isHUL', 'isHAM', 
                                        'isGAS', 'isDEV', 'isBOT', 'isBEA', 'isALO', 'isALB', 'isRBR', 'isFER', 'isMER', 'isALP', 'isMCL', 'isALF', 
                                        'isAST', 'isHAA', 'isATR', 'isWIL']

            target = 'milliseconds_y'
            sequences_2024 = []
            sequence_lengths_2024 = []

            for (raceId, driverId), group in combined_df_2024_round1.groupby(['raceId', 'driverId']):
                group = group.sort_values(by='lap')  # sort by lap within the group
                seq_data = group[features].values  # extract feature columns
                sequences_2024.append(seq_data)
                sequence_lengths_2024.append(len(seq_data))  # Store the sequence lengths

            padded_sequences_2024, sequence_lengths_2024 = pad_sequence_data_2024_round(sequences_2024,sequence_lengths_2024)

            class TestRaceDatasetRound(Dataset):
                def __init__(self, X, lengths):
                    self.X = X
                    self.lengths = lengths

                def __len__(self):
                    return len(self.X)

                def __getitem__(self, i):
                    return self.X[i], self.lengths[i]
                
            test_dataset_2024 = TestRaceDatasetRound(padded_sequences_2024, sequence_lengths_2024)
            # test_data_loader_2024 = DataLoader(test_dataset_2024, batch_size=32, shuffle=False)

            self.laptime_model.eval()  # Set the model to evaluation mode

            all_predictions = []

            print("Outputs")
            with torch.no_grad():
                for i in range(len(test_dataset_2024)):
                    batch_X, length = test_dataset_2024[i]
                    length = torch.tensor([length], dtype=torch.int64).cpu()  # Convert to 1D tensor

                    batch_X = batch_X.to(self.device)
                    length = length.to(self.device)
                    outputs = self.laptime_model(batch_X.unsqueeze(0), length)  # Add batch dimension to X

                    # mask = torch.arange(outputs.size(1))[None, :] < length[:, None]

                    # masked_outputs = outputs[mask]
                    # print(f'O: {outputs}')
                    all_predictions.append(outputs)

            # Step 4: Post-process the predictions
            all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()  # Concatenate predictions

            all_predictions_original_scale = all_predictions

            prediction_features = []
            masked_preds = []
            for i in range(len(test_dataset_2024)):
                prediction_features.append(test_dataset_2024[i][0])
                
                sequence_length = test_dataset_2024[i][1] if isinstance(test_dataset_2024[i][1], torch.Tensor) else torch.tensor(test_dataset_2024[i][1])

                # Generate the mask for valid indices based on sequence length
                mask = torch.arange(len(all_predictions_original_scale[i])) < sequence_length

                # Apply the mask by slicing, since mask is now compatible with the 1D array
                masked_output = all_predictions_original_scale[i][mask.numpy()]

                masked_preds.append(masked_output)


        

        # Create DataFrame with driver names
        predictions_df = pd.DataFrame(masked_preds)
        
        driverinfo = pd.read_csv(os.path.join(settings.DATA_DIR, 'drivers.csv'))
        # print(f'driverinfo: {driverinfo.head}')
        driver_id_name = []
        for id in driver_ids:
            name_series = driverinfo.loc[driverinfo['driverId'] == int(id), 'driverRef']
            if name_series.empty:
                driver_id_name.append(None)
            else:
                driver_id_name.append(name_series.values[0])

        predictions_df.index = driver_id_name

        
        return predictions_df
        
        # # Prepare data structures for TFT model
        # driver_lap_data = {}
        
        # # Group race states by driver
        # for lap_idx, lap_state in enumerate(race_states):
        #     for idx, row in lap_state.iterrows():
        #         driver_id = row['driverId']
                
        #         if driver_id not in driver_lap_data:
        #             driver_lap_data[driver_id] = {
        #                 'temporal': [],
        #                 'static': None,
        #                 'lap_indices': []
        #             }
                
        #         # Add temporal features for this lap
        #         driver_lap_data[driver_id]['temporal'].append(
        #             row[self.temporal_features].values
        #         )
                
        #         # Store static features (same for all laps)
        #         if driver_lap_data[driver_id]['static'] is None:
        #             driver_lap_data[driver_id]['static'] = row[self.static_features].values
                
        #         # Remember which lap and race state this corresponds to
        #         driver_lap_data[driver_id]['lap_indices'].append((lap_idx, idx))
        
        # # Predict lap times for each driver
        # for driver_id, data in driver_lap_data.items():
        #     print(f"Predicting lap times for driver {driver_id}")
            
        #     # Convert to tensors
        #     temporal_tensor = torch.tensor(np.array(data['temporal']), dtype=torch.float32).unsqueeze(0).to(self.device)
        #     static_tensor = torch.tensor(np.array([data['static']]), dtype=torch.float32).to(self.device)
            
        #     # Create attention mask (all 1s for valid laps)
        #     attention_mask = torch.ones(1, len(data['temporal'])).to(self.device)
            
        #     # Get predictions from TFT model
        #     self.laptime_model.eval()
        #     with torch.no_grad():
        #         outputs = self.laptime_model(temporal_tensor, static_tensor, attention_mask)
        #         predictions = outputs.squeeze(0).cpu().numpy()
            
        #     # Update lap times in the race states
        #     for i, (lap_idx, row_idx) in enumerate(data['lap_indices']):
        #         race_states[lap_idx].loc[row_idx, 'milliseconds_y'] = predictions[i]
        
        # # Combine all laps into one DataFrame
        # return pd.concat(race_states, ignore_index=True)
    
    def predict_safety_car(self, lap_data):
        """Predict safety car for all drivers"""
        features = lap_data[self.safety_car_features].fillna(0)
        predictions = pd.Series(self.safety_car_model.predict(features), index=lap_data.index)
        return predictions
    
    def predict_pitstops(self, lap_data):
        """Predict pit stops for all drivers"""
        features = lap_data[self.pitstop_features].fillna(0)
        predictions = pd.Series(self.pitstop_model.predict(features), index=lap_data.index)
        return predictions
    
    def predict_compound(self, driver_data):
        """Predict tire compound for a pitting driver"""
        features = pd.DataFrame([driver_data[self.compound_features].fillna(0)])
        return self.compound_model.predict(features)[0]




import json

def build_initial_data(selected_drivers, selected_track, qualifying_data, year):
    """
    Create initial DataFrame for race simulation based on user selections
    """
    # Initialize empty DataFrame with required columns
    initial_data = pd.DataFrame()
    
    # Add race identifiers
    initial_data['raceId'] = [selected_track['raceId']] * len(selected_drivers)
    initial_data['circuitId'] = [selected_track['circuitId']] * len(selected_drivers)
    initial_data['year'] = [year] * len(selected_drivers)  # Current year
    initial_data['round'] = [selected_track['round']] * len(selected_drivers)
    
    print(f'qualifying_data: {qualifying_data}')
    # Process each selected driver
    for idx, driver in enumerate(selected_drivers):
        # Get qualifying position and time
        if driver['driverId'] in qualifying_data:
            
            q_data = qualifying_data[driver['driverId']]
        else:
            q_data = {'grid': idx+1, 'q1milli': 0, 'q2milli': 0, 'q3milli': 0}
            
        # Add driver information
        initial_data.loc[idx, 'driverId'] = driver['driverId']
        initial_data.loc[idx, 'code'] = driver['code']
        initial_data.loc[idx, 'constructorId'] = driver['constructorId']
        initial_data.loc[idx, 'grid'] = q_data['grid']
        initial_data.loc[idx, 'Driver_Season_Points'] = driver['seasonPoints']
        initial_data.loc[idx, 'driverwins'] = driver['careerWins']
        initial_data.loc[idx, 'YOB'] = driver['yearOfBirth']
        initial_data.loc[idx, 'Races_before'] = driver['raceCount']
        initial_data.loc[idx, 'Races_won'] = driver['careerWins']
        initial_data.loc[idx, 'Podiums'] = driver['careerPodiums']
        
        # Add qualifying data
        initial_data.loc[idx, 'q1milli'] = q_data['q1milli']
        initial_data.loc[idx, 'q2milli'] = q_data['q2milli']
        initial_data.loc[idx, 'q3milli'] = q_data['q3milli']
        
        # Initial race state
        initial_data.loc[idx, 'lap'] = 1
        # Set tire compound (either from user input or default based on track/position)
        # initial_data.loc[idx, 'tyre_compound'] = (
        #     custom_tire_compounds[driver['driverId']] 
        #     if custom_tire_compounds and driver['driverId'] in custom_tire_compounds
        #     else recommend_starting_compound(selected_track, q_data['grid'])
        # )
        initial_data.loc[idx, 'tyre_compound'] = 0

        initial_data.loc[idx, 'tyre_age'] = 1
        initial_data.loc[idx, 'isSafetyCar'] = 0
        initial_data.loc[idx, 'isSafetyCarPrev'] = 0
        initial_data.loc[idx, 'isPitting'] = 0
        
        # Add one-hot encoded driver indicators
        driver_indicator_columns = ['isVET', 'isZHO', 'isVER', 'isTSU', 'isSTR', 'isMSC', 'isSAR', 'isRIC',
'isSAI', 'isRUS', 'isPIA', 'isPER', 'isOCO', 'isNOR', 'isMAG', 'isLEC', 'isLAW', 'isLAT', 'isHUL', 'isHAM',
'isGAS', 'isDEV', 'isBOT', 'isBEA', 'isALO', 'isALB']
        # Fill all driver indicators with 0 first
        for driver_col in driver_indicator_columns:
            initial_data.loc[idx, driver_col] = 0
        # Set the relevant driver indicator to 1
        driver_col = f"is{driver['code']}"
        if driver_col in driver_indicator_columns:
            initial_data.loc[idx, driver_col] = 1
            
        # Add one-hot encoded constructor indicators
        constructor_indicator_columns = ['isRBR', 'isFER', 'isMER', 'isALP', 'isMCL', 'isALF', 
                                                                'isAST', 'isHAA', 'isATR', 'isWIL']
        # Fill all constructor indicators with 0 first
        for team_col in constructor_indicator_columns:
            initial_data.loc[idx, team_col] = 0
        # Set the relevant constructor indicator to 1
        team_col = f"is{driver['team_code']}"
        if team_col in constructor_indicator_columns:
            initial_data.loc[idx, team_col] = 1
    
    # # Apply scaling factors as used in training
    # initial_data = custom_scaler(initial_data)
    
    return initial_data


def create_selected_drivers(driver_csv_path, year=2025, laptimes_path=None, selected_round=1):
    """Create list of selected drivers based on data source appropriate for the year and round"""
    # Special case for 2024: extract from laptimestest3.csv
    if year == 2024 and laptimes_path:
        # Read laptimes data
        laptimes_df = pd.read_csv(laptimes_path)
        
        # Filter for 2024 data and the selected round
        laptimes_df = laptimes_df[(laptimes_df['year'] == 2024) & 
                                  (laptimes_df['round'] == selected_round)]
        
        # If no data found for this round, return empty list or raise error
        if laptimes_df.empty:
            print(f"No data found for year 2024, round {selected_round}")
            return []
        
        # Load reference data for driver codes
        drivers_df = pd.read_csv(os.path.join(settings.DATA_DIR, 'drivers.csv'))
        constructors_df = pd.read_csv(os.path.join(settings.DATA_DIR, 'constructors.csv'))
        
        # Get unique drivers for this round
        unique_drivers = laptimes_df[['driverId', 'constructorId']].drop_duplicates()
        
        selected_drivers = []
        for _, row in unique_drivers.iterrows():
            # Get driver data (first row for this driver)
            driver_data = laptimes_df[laptimes_df['driverId'] == row['driverId']].iloc[0]
            
            # Look up driver code from drivers reference table
            driver_row = drivers_df[drivers_df['driverId'] == row['driverId']]
            driver_code = driver_row.iloc[0]['code'] if not driver_row.empty else f"D{row['driverId']}"
            
            # Look up team code from constructors reference table
            team_row = constructors_df[constructors_df['constructorId'] == row['constructorId']]
            team_code = team_row.iloc[0].get('constructorRef', '')[:3].upper() if not team_row.empty else "UNK"
            
            # Create driver info dictionary
            driver_info = {
                'driverId': int(driver_data['driverId']),
                'constructorId': int(driver_data['constructorId']),
                'code': driver_code,
                'team_code': team_code,
                'seasonPoints': float(driver_data.get('Driver_Season_Points', 0)),
                'yearOfBirth': int(driver_data.get('YOB', 0)),
                'raceCount': int(driver_data.get('Races_before', 0)),
                'careerWins': int(driver_data.get('Races_won', 0)),
                'careerPodiums': int(driver_data.get('Podiums', 0))
            }
            selected_drivers.append(driver_info)
        
        return selected_drivers
    
    # Default case for other years: read from year-specific CSV
    df = pd.read_csv(driver_csv_path)
    
    selected_drivers = []
    for index, row in df.iterrows():
        driver_info = {
            'driverId': int(row['driverId']),
            'constructorId': int(row['constructorId']),
            'code': row['code'],
            'team_code': row['team_code'],
            'seasonPoints': float(row['seasonPoints']),
            'yearOfBirth': int(row['yearOfBirth']),
            'raceCount': int(row['raceCount']),
            'careerWins': int(row['careerWins']),
            'careerPodiums': int(row['careerPodiums'])
        }
        selected_drivers.append(driver_info)
    
    return selected_drivers

def create_selected_track(tracks_csv_path, selected_round=24, year=2025, laptimes_path=None):
    """Create track information dictionary based on year and round"""
    # Special case for 2024: extract from laptimestest3.csv
    if year == 2024 and laptimes_path:
        # Read laptimes data
        laptimes_df = pd.read_csv(laptimes_path)
        
        # Filter for 2024 and the selected round
        track_data = laptimes_df[(laptimes_df['year'] == 2024) & 
                                (laptimes_df['round'] == selected_round)]
        
        if track_data.empty:
            raise ValueError(f"No data found for 2024, round {selected_round}")
        
        # Get track info from first row
        track_row = track_data.iloc[0]
        
        selected_track = {
            'raceId': int(track_row['raceId']),
            'circuitId': int(track_row['circuitId']),
            'round': int(track_row['round'])
        }
        
        return selected_track
    
    # Default case for other years
    df = pd.read_csv(tracks_csv_path)
    track_row = df[df['round'] == selected_round].iloc[0]
    
    selected_track = {
        'raceId': int(track_row['raceId']),
        'circuitId': int(track_row['circuitId']),
        'round': int(track_row['round']),
    }
    
    return selected_track




def initialize_qualifying_data(laptimes_csv_path, selected_drivers, selected_track):
    """
    Extract qualifying data from laptimestest3.csv with proper F1 qualifying sorting logic
    
    Args:
        laptimes_csv_path: Path to laptimestest3.csv
        selected_drivers: List of driver dictionaries
        selected_track: Dictionary with track information
        
    Returns:
        Dictionary with qualifying data for each driver
    """
    # Read laptimes data
    laptimes_df = pd.read_csv(laptimes_csv_path)
    
    # Filter for 2024 version of the chosen race
    qualifying_df = laptimes_df[
        (laptimes_df['year'] == 2024) & 
        (laptimes_df['circuitId'] == selected_track['circuitId'])
    ]
    
    # Get first lap for each driver (contains qualifying info)
    if not qualifying_df.empty:
        qualifying_df = qualifying_df.groupby('driverId').first().reset_index()
    
    # Initialize qualifying_data dictionary
    qualifying_data = {}
    
    # Process each selected driver
    for driver in selected_drivers:
        driver_id = driver['driverId']
        
        # Check if driver has qualifying data
        driver_quali = qualifying_df[qualifying_df['driverId'] == driver_id] if not qualifying_df.empty else pd.DataFrame()
        
        if not driver_quali.empty:
            # Driver found in qualifying data
            row = driver_quali.iloc[0]
            qualifying_data[driver_id] = {
                'grid': 0,  # Will be determined later based on quali times
                'q1milli': float(row['q1milli']) if (row['q1milli']!=0) else 100,
                'q2milli': float(row['q2milli']) if (row['q2milli']!=0) else 100,
                'q3milli': float(row['q3milli']) if (row['q3milli']!=0) else 100
            }
        else:
            # Driver not found, use default values of 100 for all sessions
            qualifying_data[driver_id] = {
                'grid': 0,
                'q1milli': 100,
                'q2milli': 100,
                'q3milli': 100
            }
    
    # Position counter for grid assignment
    position = 1
    all_drivers = list(qualifying_data.items())
    
    # 1. First sort Q3 drivers (positions 1-10)
    # Get drivers with valid Q3 times (not 100)
    q3_drivers = [(k, v) for k, v in all_drivers if v['q3milli'] != 100]
    q3_sorted = sorted(q3_drivers, key=lambda x: x[1]['q3milli'])
    
    # Assign positions for Q3 drivers
    for driver_id, quali_data in q3_sorted:
        qualifying_data[driver_id]['grid'] = position
        position += 1
    
    # 2. If we haven't filled positions 1-10 yet, look for drivers with valid Q2 times
    if position <= 10:
        # Get drivers with valid Q2 times who haven't been assigned a grid position yet
        q2_for_q3 = [(k, v) for k, v in all_drivers 
                     if v['q2milli'] != 100 and v['grid'] == 0]
        q2_for_q3_sorted = sorted(q2_for_q3, key=lambda x: x[1]['q2milli'])
        
        # Fill remaining top-10 positions with best Q2 drivers
        while q2_for_q3_sorted and position <= 10:
            driver_id, quali_data = q2_for_q3_sorted.pop(0)
            qualifying_data[driver_id]['grid'] = position
            position += 1
    
    # 3. Now handle positions 11-15 with remaining Q2 drivers
    q2_drivers = [(k, v) for k, v in all_drivers 
                 if v['q2milli'] != 100 and v['grid'] == 0]
    q2_sorted = sorted(q2_drivers, key=lambda x: x[1]['q2milli'])
    
    # Assign positions for Q2 drivers
    for driver_id, quali_data in q2_sorted:
        qualifying_data[driver_id]['grid'] = position
        position += 1
    
    # 4. If we haven't filled positions up to 15, look for drivers with valid Q1 times
    if position <= 15:
        # Get drivers with valid Q1 times who haven't been assigned a grid position yet
        q1_for_q2 = [(k, v) for k, v in all_drivers 
                     if v['q1milli'] != 100 and v['grid'] == 0]
        q1_for_q2_sorted = sorted(q1_for_q2, key=lambda x: x[1]['q1milli'])
        
        # Fill remaining positions up to 15 with best Q1 drivers
        while q1_for_q2_sorted and position <= 15:
            driver_id, quali_data = q1_for_q2_sorted.pop(0)
            qualifying_data[driver_id]['grid'] = position
            position += 1
    
    # 5. Handle remaining positions with drivers who have valid Q1 times
    q1_drivers = [(k, v) for k, v in all_drivers 
                 if v['q1milli'] != 100 and v['grid'] == 0]
    q1_sorted = sorted(q1_drivers, key=lambda x: x[1]['q1milli'])
    
    # Assign positions for Q1 drivers
    for driver_id, quali_data in q1_sorted:
        qualifying_data[driver_id]['grid'] = position
        position += 1
    
    # 6. Finally, assign positions for drivers with no valid qualifying times
    no_time_drivers = [(k, v) for k, v in all_drivers if v['grid'] == 0]
    for driver_id, quali_data in no_time_drivers:
        qualifying_data[driver_id]['grid'] = position
        position += 1
    
    return qualifying_data






from django.conf import settings
import os

def run_simulation(
    race_data_path, 
    total_laps,
    year=2025,                  # New parameter for year selection
    model="TFT",
    model_type="default",       # New parameter for model type
    selected_round=1,  # Added round parameter with default
    custom_safety_car_laps=None,# For custom SC laps
    custom_pitstops=None,       # For custom pitstops
    custom_compounds=None,      # For custom compounds
    custom_qualifying_times=None# For custom qualifying times
):


    # custom_safety_car_laps = [5, 20, 35]  # Safety car deploys on laps 5, 20, and 35

    # custom_pitstops = {
    #     1: [15, 35],     # Driver ID 1 pits on laps 15 and 35
    #     33: [14, 38],    # Driver ID 33 pits on laps 14 and 38
    #     44: [18, 40]     # Driver ID 44 pits on laps 18 and 40
    # }

    #     custom_compounds = {
    #     1: {
    #         15: 2,  # Driver ID 1 switches to compound 2 on lap 15
    #         35: 1   # Driver ID 1 switches to compound 1 on lap 35
    #     },
    #     33: {
    #         14: 1,  # Driver ID 33 switches to compound 1 on lap 14
    #         38: 0   # Driver ID 33 switches to compound 0 on lap 38
    #     }
    # }

    # Set PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    model_paths = {
        'safety_car': os.path.join(settings.BASE_DIR, 'static', 'models', 'safety_car_pipeline.pkl'),
        'pitstop': os.path.join(settings.BASE_DIR, 'static', 'models', 'pitstop_pipeline.pkl'),
        'compound': os.path.join(settings.BASE_DIR, 'static', 'models', 'pitstop_compounds_pipeline.pkl'),
        'tft': os.path.join(settings.BASE_DIR, 'static', 'models', 'tft_b16_hs256_nh8_numenc4_lstm2_d0.1_w0.6_0.3_0.1.pth.pth'),
        'lstm': os.path.join(settings.BASE_DIR, 'static', 'models', 'lstm_model_b5_e150_lr0.0001_h512_l2_d0.1_w0.6_0.3_0.1.pth.pth')
    }
    
    # Define paths to data files
    driver_info_path = os.path.join(settings.DATA_DIR, f'{year}_driver_info.csv')
    track_info_path = os.path.join(settings.DATA_DIR, f'{year}_track_info.csv')
    laptimes_path = os.path.join(settings.DATA_DIR, 'laptimestest3.csv')



    try:
        safety_car_model = joblib.load(model_paths['safety_car'])
        pitstop_model = joblib.load(model_paths['pitstop'])
        compound_model = joblib.load(model_paths['compound'])

        model_choice = model.upper()
        if model_choice == "LSTM":
            try:
                # Load LSTM model
                laptime_model = torch.load(model_paths['lstm'], map_location=device, weights_only=False)
            except Exception as e:
                print(f"Error loading LSTM model: {e}, falling back to TFT")
                model_choice = "TFT"
                with torch.serialization.safe_globals([TemporalFusionTransformer]):
                    laptime_model = torch.load(model_paths['tft'], map_location=device, weights_only=False)
        else:  # Default to TFT
            with torch.serialization.safe_globals([TemporalFusionTransformer]):
                laptime_model = torch.load(model_paths['tft'], map_location=device, weights_only=False)
        
    except FileNotFoundError as e:
        raise Exception(f"Model file not found: {e}")
    
    
    # Load race data based on selected year
    selected_drivers = create_selected_drivers(
        driver_info_path, 
        year=year, 
        laptimes_path=laptimes_path if year == 2024 else None,
        selected_round = selected_round
    )
    
    selected_track = create_selected_track(
        track_info_path, 
        selected_round=selected_round,
        year=year,
        laptimes_path=laptimes_path if year == 2024 else None
    )
    
    # Handle qualifying data based on year and model type
    if model_type == "default":
        # For both years, get qualifying data from 2024 source
        qualifying_data = initialize_qualifying_data(laptimes_path, selected_drivers, selected_track)
    else:  # Custom model
        # Use custom qualifying times provided by user
        qualifying_data = custom_qualifying_times
    
    # Build initial data
    initial_data = build_initial_data(selected_drivers, selected_track, qualifying_data, year=year)
    initial_data.to_csv(f'initial_data.csv', index=False)

    # Create simulator with custom data flags
    simulator = F1RaceSimulator(
        safety_car_model=safety_car_model,
        pitstop_model=pitstop_model,
        compound_model=compound_model,
        laptime_model=laptime_model,
        device=device,
        use_custom_data=(model_type == "custom"),
        custom_safety_car_laps=custom_safety_car_laps,
        custom_pitstops=custom_pitstops,
        custom_compounds=custom_compounds,
        laptime_model_type=model
    )
    
    # Run simulation and return results
    race_results = simulator.simulate_race(initial_data, total_laps)
    race_results.to_csv(f'race_simulation_results_{year}_{model_type}.csv')
    
    return race_results

# run_simulation('../laptimestest3.csv',58,{'safety_car': '../dynamic_features/safety_car_pipeline.pkl',
#                                           'pitstop': '../dynamic_features/pitstop_pipeline.pkl', 
#                                           'compound': '../dynamic_features/pitstop_compounds_pipeline.pkl',
#                                           'laptime':'./tft_b16_hs256_nh8_numenc4_lstm2_d0.1_w0.6_0.3_0.1.pth.pth'})

