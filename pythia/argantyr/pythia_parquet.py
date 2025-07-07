#!/usr/bin/env python3

from __future__ import print_function
import tqdm
import argparse
import os
import numpy as np
import sys
import yasp

import heppyy.util.fastjet_cppyy
import heppyy.util.pythia8_cppyy
import heppyy.util.heppyy_cppyy

from cppyy.gbl import fastjet as fj
from cppyy.gbl import Pythia8
from cppyy.gbl.std import vector

from heppyy.util.mputils import logbins
from heppyy.pythia_util import configuration as pyconf

import ROOT
import math
import array

from yasp import GenericObject
from heppyy.util.logger import Logger
import logging

import pandas as pd

def main():
	parser = argparse.ArgumentParser(description='pythia8 on the fly', prog=os.path.basename(__file__))
	# pyconf.add_standard_pythia_args(parser)
	parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
	parser.add_argument('-o','--output', help='root output filename', default='pythia_run_output.parquet', type=str)
	parser.add_argument('--etadet', help='detector eta', default=1.0, type=float)
	parser.add_argument('--nev', help='number of events to generate (overrides config file)', default=None, type=int)
	parser.add_argument('--cmnd', help='pythia command file', default='pythia.cmnd', type=str)
	parser.add_argument('--logfile', help='log file name', default=None, type=str)
	parser.add_argument('--seed', help='random seed', default=None, type=int)

	args = parser.parse_args()

	log = Logger(log_file=args.logfile, level=logging.INFO, console=True)

	# output
	# Create lists to hold event-level and particle-level data
	event_data = []
	particle_data = []

 
	pythia = Pythia8.Pythia()
	log.info("Reading configuration file:", args.cmnd)
	pythia.readFile(args.cmnd)
	if args.seed is not None:
		log.info(f"Setting random seed: {args.seed}")
		pythia.readString(f"Random:setSeed = on")
		pythia.readString(f"Random:seed = {args.seed}")
	log.info("Initializing PYTHIA...")
	if not pythia.init():
		log.error("[e] pythia initialization failed.")
		log.error("[e] Listing changed settings:")
		pythia.settings.listChanged()
		log.error("[e] Check the configuration file for errors.")
		return
	log.info("PYTHIA initialized successfully!")

	# Get number of events from config file or command line
	if args.nev is not None:
		# Command line override
		nev = args.nev
		log.info(f"Using command line event count: {nev}")
	else:
		# Use value from config file
		nev = pythia.mode("Main:numberOfEvents")
		log.info(f"Using config file event count: {nev}")
	
	if nev < 10:
		nev = 10
		log.info(f"Minimum event count enforced: {nev}")

	count_jets = 0
	pbar = tqdm.tqdm(total=nev)
	while pbar.n < nev:
		if not pythia.next():
			continue

		event_id = pbar.n
		_pythia_info = Pythia8.getInfo(pythia)
		# Collect event-level information (will be updated after particle loop)
		event_info = {
			'event_id': event_id,

			'impact_parameter': _pythia_info.hiInfo.b(),  # Impact parameter for heavy-ion collisions
			'n_part_proj': _pythia_info.hiInfo.nPartProj(),  # Number of participants
 			'n_part_targ': _pythia_info.hiInfo.nPartTarg(),  # Number of participants
			'n_part': _pythia_info.hiInfo.nPartProj() + _pythia_info.hiInfo.nPartTarg(),  # Number of participants
			'n_coll': _pythia_info.hiInfo.nCollTot(),  # Total number of collisions

			'hi_weight': _pythia_info.hiInfo.weight(),
			'hi_weightSum': _pythia_info.hiInfo.weightSum(),
   
			# from pythia 8.315 - also better to store post gen
			'hi_glauber_tot': _pythia_info.hiInfo.glauberTot(),  # Glauber total
			'hi_glauber_tot_err': _pythia_info.hiInfo.glauberTotErr(),  # Glauber total
			'hi_glauber_nd': _pythia_info.hiInfo.glauberND(),  # Glauber ND
			'hi_glauber_nd_err': _pythia_info.hiInfo.glauberNDErr(),  # Glauber ND error	
			'hi_glauber_inel': _pythia_info.hiInfo.glauberINEL(),  # Glauber INEL
			'hi_glauber_inel_err': _pythia_info.hiInfo.glauberINELErr(),  # Glauber INEL error
			'hi_glauber_el': _pythia_info.hiInfo.glauberEL(),  # Glauber EL
			'hi_glauber_el_err': _pythia_info.hiInfo.glauberELErr(),  # Glauber EL error

			'sigma_gen': _pythia_info.sigmaGen(),  # Inelastic cross-section
			'sigma_gen_err': _pythia_info.sigmaErr(),	# Total cross-section
			'weight': _pythia_info.weight(),
			'weightSum': _pythia_info.weightSum()
		}
		
		# Collect particle-level information for this event
		particles_in_event = []
		# Variables for event-level calculations
		sum_pT = 0.0
		sum_eta = 0.0
		sum_phi = 0.0
		qx = 0.0  # x-component of Q-vector for event plane
		qy = 0.0  # y-component of Q-vector for event plane
		
		for p in pythia.event:
			if p.isFinal() and abs(p.eta()) < args.etadet:
				pT = p.pT()
				eta = p.eta()
				phi = p.phi()
				
				particle_info = {
					'event_id': event_id,
					'particle_id': p.id(),
					'pT': pT,
					'eta': eta,
					'phi': phi,
					'charge': p.charge()
				}
				particles_in_event.append(particle_info)
				particle_data.append(particle_info)
				
				# Accumulate for event-level calculations
				sum_pT += pT
				sum_eta += eta
				sum_phi += phi
				
				# Calculate Q-vector components (pT-weighted for better resolution)
				qx += pT * math.cos(2.0 * phi)  # 2nd harmonic event plane
				qy += pT * math.sin(2.0 * phi)
		
		n_particles = len(particles_in_event)
		
		# Calculate event-level quantities
		if n_particles > 0:
			mean_pT = sum_pT / n_particles
			mean_eta = sum_eta / n_particles
			mean_phi = sum_phi / n_particles
			# Event plane angle (2nd harmonic)
			event_plane_angle = 0.5 * math.atan2(qy, qx)
		else:
			mean_pT = 0.0
			mean_eta = 0.0
			mean_phi = 0.0
			event_plane_angle = 0.0
		
		# Update event info with calculated quantities
		event_info.update({
			'n_particles': n_particles,
			'mean_pT': mean_pT,
			'mean_eta': mean_eta,
			'mean_phi': mean_phi,
			'event_plane_angle': event_plane_angle,
			'qx' : qx,
			'qy' : qy
		})
		event_data.append(event_info)
		
		pbar.update(1)
  
	pbar.close()
	log.info("Event generation completed.")

	pythia.stat()

	# Create separate DataFrames for event-level and particle-level data
	events_df = pd.DataFrame(event_data)
	particles_df = pd.DataFrame(particle_data)
	
	# Save both DataFrames to separate parquet files
	base_name = args.output.replace('.parquet', '')
	events_file = f"{base_name}_events.parquet"
	particles_file = f"{base_name}_particles.parquet"
	
	events_df.to_parquet(events_file, engine="pyarrow")
	particles_df.to_parquet(particles_file, engine="pyarrow")
 
	log.info(f"Event data written to {events_file}")
	log.info(f"Particle data written to {particles_file}")
	log.info(f"Total events: {len(events_df)}")
	log.info(f"Total particles: {len(particles_df)}")
	
if __name__ == '__main__':
	main()