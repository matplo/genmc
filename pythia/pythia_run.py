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

def main():
	parser = argparse.ArgumentParser(description='pythia8 on the fly', prog=os.path.basename(__file__))
	# pyconf.add_standard_pythia_args(parser)
	parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
	parser.add_argument('-o','--output', help='root output filename', default='pythia_run_output.root', type=str)
	parser.add_argument('--etadet', help='detector eta', default=2.5, type=float)
	parser.add_argument('--nev', help='number of events to generate (overrides config file)', default=None, type=int)
	parser.add_argument('--cmnd', help='pythia command file', default='pythia.cmnd', type=str)
	parser.add_argument('--logfile', help='log file name', default=None, type=str)

	args = parser.parse_args()

	log = Logger(log_file=args.logfile, level=logging.INFO, console=True)

	# output file
	from alian.io.root_io import SingleRootFile
	fout = SingleRootFile(args.output)
	fout.root_file.cd()
 
	# mycfg = []
	# pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
	# if not pythia:
	# 	print("[e] pythia initialization failed.")
	# 	return
 
	pythia = Pythia8.Pythia()
	log.info("Reading configuration file:", args.cmnd)
	pythia.readFile(args.cmnd)
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
			break
		# final_parts = vector[fj.PseudoJet]([fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in pythia.event if p.isFinal()])
		pbar.update(1)
  
	pbar.close()
	log.info("Event generation completed.")

	pythia.stat()

	fout.close()

if __name__ == '__main__':
	main()