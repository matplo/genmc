#!/usr/bin/env python3
from __future__ import print_function
import os

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import argparse

# --- Configuration ---
parser = argparse.ArgumentParser(description='Convert Parquet files to HDF5 for OmniLearn.')
parser.add_argument('--parquet-events', type=str, required=True, help='Path to the event Parquet file.')
parser.add_argument('--parquet-particles', type=str, required=True, help='Path to the particle Parquet file.')
parser.add_argument('--h5output-filename', type=str, required=True, help='Path to the output HDF5 file.')
args = parser.parse_args()

collected_no_parts_warn = []

# --- Load Parquet files ---
# Load the data
try:
    events_df = pd.read_parquet(args.parquet_events)
    particles_df = pd.read_parquet(args.parquet_particles)
    # Check if the data is empty
    if events_df.empty or particles_df.empty:
        print("Error: One or both of the dataframes are empty.")
        exit(1)
    print("Data loaded successfully!")
    print(f"Events: {len(events_df)} events")
    print(f"Particles: {len(particles_df)} particles")
    print(f"Average particles per event: {len(particles_df)/len(events_df):.1f}")

    # Display basic info about the data
    print("\nEvent-level columns:", list(events_df.columns))
    print("Particle-level columns:", list(particles_df.columns))

    # --- Process data for HDF5 conversion ---
    particle_data = []  # particle level features
    jet_data = []  # event level

    print('Initializing processing ---> ')

    # Group particles by event
    particles_by_event = particles_df.groupby('event_id')

    for event_id, event_row in tqdm(events_df.iterrows(), desc="Processing events", total=len(events_df)):
        # Get particles for this event
        if event_id in particles_by_event.groups:
            event_particles = particles_by_event.get_group(event_id)

            # Extract particle features (assuming columns exist - adjust as needed)
            pt = event_particles['pT'].values
            eta = event_particles['eta'].values
            phi = event_particles['phi'].values

            # Compute derived features
            e_pT = np.mean(pt)
            eta_jet = np.mean(eta)
            pT_rel = np.log(pt / e_pT)
            # eta_centered = eta - eta_jet
            phi_rel = phi - np.mean(phi)
            eta_rel = eta - eta_jet
            e_phi = np.mean(phi)
            sum_pT = np.sum(pt)

            # Simple Q vector computation (adjust based on your needs) - we will use calculated values from parquet file
            # Qx = np.sum(pt * np.cos(phi))
            # Qy = np.sum(pt * np.sin(phi))
            # ep_angle = np.arctan2(Qy, Qx)  # Angle of the Q vector
            # Create particle features array
            particles_array = np.column_stack([
                # eta_centered.astype(np.float16),
                phi_rel.astype(np.float16),
                pT_rel.astype(np.float16),
                eta_rel.astype(np.float16),
                pt.astype(np.float16),
                eta.astype(np.float16),
                phi.astype(np.float16)
            ])

            # Create jet/event features array
            # Assuming centrality exists in events_df, otherwise set to 0
            # centrality = event_row.get('centrality', 0) if 'centrality' in events_df.columns else 0
            centrality = event_row.get('n_particles', 0) if 'n_particles' in events_df.columns else 0
            imp_par = event_row.get('impact_parameter', 0) if 'impact_parameter' in events_df.columns else 0
            n_coll = event_row.get('n_coll', 0) if 'n_coll' in events_df.columns else 0
            n_part = event_row.get('n_part', 0) if 'n_part' in events_df.columns else 0
            ep_angle = event_row.get('event_plane_angle', 0) if 'event_plane_angle' in events_df.columns else 0
            Qx = event_row.get('qx', 0) if 'qx' in events_df.columns else 0
            Qy = event_row.get('qy', 0) if 'qy' in events_df.columns else 0
            # jet_array = [e_pT, sum_pT, Qx, Qy, eta_jet, centrality, ep_angle, imp_par, n_coll, n_part, e_phi, len(event_particles)]
			jet_array = [e_pT, sum_pT, Qx, Qy, eta_jet, ep_angle, imp_par, n_coll, n_part, e_phi, len(event_particles)]

            particle_data.append(particles_array)
            jet_data.append(jet_array)
        else:
            # Handle events with no particles
            # print(f"Warning: Event {event_id} has no particles, skipping...")
            collected_no_parts_warn.append(event_id)
            continue

    print(f"Processed {len(particle_data)} events")
    if len(collected_no_parts_warn) > 0:
	    print(f'Warning: {len(collected_no_parts_warn)} events had no particles.')

    # --- HDF5 padding and saving ---
    if particle_data:
        max_particles = max(p.shape[0] for p in particle_data)
        print(f"Maximum particles per event: {max_particles}")

        # Pad particle data
        particles_padded = np.array([
            np.pad(p, ((0, max_particles - p.shape[0]), (0, 0)), mode="constant")
            for p in particle_data
        ], dtype=np.float16)

        jet_data_array = np.array(jet_data, dtype=np.float32)

        # Save to HDF5
        print(f"Saving to {args.h5output_filename}...")
        with h5py.File(args.h5output_filename, "w") as h5f:
            h5f.create_dataset("particle", data=particles_padded, dtype=np.float16)
            h5f.create_dataset("jet", data=jet_data_array, dtype=np.float16)

            # Add metadata
            h5f.attrs['n_events'] = len(particle_data)
            h5f.attrs['max_particles'] = max_particles
            h5f.attrs['particle_features'] = ['eta_centered', 'phi_rel', 'pT_rel', 'eta_rel', 'pT', 'eta', 'phi']
            h5f.attrs['jet_features'] = ['e_pT', 'sum_pT', 'Qx', 'Qy', 'eta_jet', 'centrality', 'ep_angle', 'b', 'n_coll', 'n_part', 'e_phi', 'n_particles']

        print("HDF5 file created successfully!")
        print(f"Particle dataset shape: {particles_padded.shape}")
        print(f"Jet dataset shape: {jet_data_array.shape}")
    else:
        print("No data to save!")

except FileNotFoundError as e:
    print(f"Error: Could not find data files. Make sure you have run the pythia_parquet.py script first.")
    print(f"Looking for: {args.parquet_events} and {args.parquet_particles}")
    print("You may need to adjust the file paths above.")
    exit(1)
except Exception as e:
    print(f"Error during processing: {e}")
    print("Please check your parquet files and column names.")
