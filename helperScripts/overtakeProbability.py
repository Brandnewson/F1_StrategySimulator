# This script estimates overtake density around the Spa lap using FastF1 data.

from pathlib import Path

import fastf1
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIAGRAMS = ROOT / "report" / "diagrams"
PROBABILITY_OUT = REPORT_DIAGRAMS / "OvertakeProbabilitySpa.png"
ABSOLUTE_OUT = REPORT_DIAGRAMS / "NumberOfOvertakesSpa.png"
MAPPING_OUT = REPORT_DIAGRAMS / "overtakingMapping.png"

def main():

	# Enable FastF1 cache for faster repeated runs
	fastf1.Cache.enable_cache(str(ROOT / 'fastf1_cache'))

	# Load a recent Spa race session (e.g., 2023)
	year = 2023
	gp = 'Belgian Grand Prix'
	session_type = 'R'  # Race
	session = fastf1.get_session(year, gp, session_type)
	print(f"Loading {year} {gp} {session_type}...")
	session.load(telemetry=True, weather=False)

	# Get laps data
	laps = session.laps
	drivers = laps['Driver'].unique()

	# We'll use position changes between laps to infer overtakes
	overtakes = []
	for drv in drivers:
		drv_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
		prev_pos = None
		for idx, row in drv_laps.iterrows():
			pos = row['Position']
			lapnum = row['LapNumber']
			if prev_pos is not None and pos < prev_pos:
				# Overtake detected (gained position)
				# We'll use the finish line location for this lap
				overtakes.append({
					'Driver': drv,
					'LapNumber': lapnum,
					'Position': pos,
					'Stint': row['Stint'],
					'TrackStatus': row['TrackStatus'],
					'LapTime': row['LapTime'],
					'Distance': row['Distance'] if 'Distance' in row else None,
					'LapStartTime': row['LapStartTime'],
					'LapEndTime': row['LapStartTime'] + row['LapTime'] if pd.notnull(row['LapTime']) else None
				})
			prev_pos = pos

	overtakes_df = pd.DataFrame(overtakes)
	print(f"Total overtakes detected: {len(overtakes_df)}")

	# Map overtakes to track positions using telemetry (approximate by lap start or sector times)
	# We'll use the LapStartTime to get the car's position at the start of the lap
	# For more accuracy, you could interpolate using sector times or telemetry, but this is a good start
	overtake_positions = []
	total_laps = laps['LapNumber'].max()
	for _, row in overtakes_df.iterrows():
		drv = row['Driver']
		lapnum = row['LapNumber']
		# Only consider overtakes in the first half of the race
		if lapnum > total_laps / 2:
			continue
		lap = laps[(laps['Driver'] == drv) & (laps['LapNumber'] == lapnum)]
		if lap.empty:
			continue
		lap_obj = lap.iloc[0] if not lap.empty else None
		if lap_obj is not None:
			try:
				tel = lap_obj.get_telemetry()
				if 'Distance' not in tel.columns:
					tel = fastf1.utils.add_distance(tel)
				# Find where DriverAhead changes (ignoring empty strings)
				driver_ahead = tel['DriverAhead'].replace('', np.nan).ffill()
				driver_ahead = driver_ahead.dropna()
				change_idx = driver_ahead[driver_ahead != driver_ahead.shift()].index
				# The first change (after the start) is likely the overtake
				if len(change_idx) > 1:
					# Take the first change after the start
					overtake_idx = change_idx[1]
					overtake_distance = tel.loc[overtake_idx, 'Distance']
					overtake_positions.append(overtake_distance)

			except Exception as e:
				print(f"Could not get telemetry for {drv} lap {lapnum}: {e}")

	# Bin overtakes by track distance
	if not overtake_positions:
		print("No overtake positions found. Exiting.")
		return
	overtake_positions = np.array(overtake_positions)
	# Get track length in meters from circuit info
	circuit_info = session.get_circuit_info()
	# Use the get_track_length() method as per FastF1 documentation
	try:
		track_length = circuit_info.get_track_length()
	except Exception:
		track_length = 7004  # Spa default length if not found
	bins = np.linspace(0, track_length, 50)
	hist, bin_edges = np.histogram(overtake_positions, bins=bins)
	overtake_prob = hist / hist.sum()

	REPORT_DIAGRAMS.mkdir(parents=True, exist_ok=True)

	# Plot 1: Normalised probability
	plt.figure(figsize=(12, 6))
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	plt.bar(bin_centers, overtake_prob, width=(track_length/50)*0.9, color='orange', edgecolor='k', alpha=0.7)
	plt.title(f"Overtaking Probability by Track Position\nSpa-Francorchamps {year} Race (First Half)")
	plt.xlabel("Track Distance (m)")
	plt.ylabel("Overtake Probability (normalised)")
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.savefig(PROBABILITY_OUT, dpi=220, bbox_inches='tight')
	plt.close()

	# Plot 2: Absolute overtakes
	plt.figure(figsize=(12, 6))
	plt.bar(bin_centers, hist, width=(track_length/50)*0.9, color='blue', edgecolor='k', alpha=0.7)
	plt.title(f"Absolute Overtakes by Track Position\nSpa-Francorchamps {year} Race (First Half)")
	plt.xlabel("Track Distance (m)")
	plt.ylabel("Number of Overtakes")
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.savefig(ABSOLUTE_OUT, dpi=220, bbox_inches='tight')
	plt.close()
	
	# Plot 3: Track map with overtakes as color gradient
	# Load Spa centre line coordinates
	centreline_df = pd.read_csv(ROOT / "datasets" / "SpaCentreLine.csv")
	x = centreline_df['X'].values
	y = centreline_df['Y'].values
	n_points = len(x)
	# Assign a distance to each point along the centreline
	# Assume uniform spacing for simplicity
	centreline_distances = np.linspace(0, track_length, n_points)

	# Map each centreline point to the relative overtake density of its distance bin.
	# This produces continuous coloured sections that match the bar-chart bins and
	# keeps the colour bar honest by using the observed peak density, not an implied 1.0 full scale.
	bin_indices = np.digitize(centreline_distances, bin_edges, right=False) - 1
	bin_indices = np.clip(bin_indices, 0, len(overtake_prob) - 1)
	relative_density = overtake_prob[bin_indices]
	peak_relative_density = float(relative_density.max()) if len(relative_density) else 0.0

	plt.figure(figsize=(14, 7))
	from matplotlib.collections import LineCollection
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	segment_colors = (relative_density[:-1] + relative_density[1:]) / 2
	norm = plt.Normalize(0, peak_relative_density if peak_relative_density > 0 else 1.0)
	lc = LineCollection(segments, cmap='turbo', norm=norm, zorder=2)
	lc.set_array(segment_colors)
	lc.set_linewidth(8)
	plt.gca().add_collection(lc)
	# Add a black outline for the track for contrast
	plt.plot(x, y, color='black', linewidth=10, alpha=0.5, zorder=1)
	plt.plot(x, y, color='gray', linewidth=1, alpha=0.3, zorder=0)
	colorbar = plt.colorbar(lc, label='Relative Overtake Density')
	if peak_relative_density > 0:
		ticks = np.linspace(0.0, peak_relative_density, 5)
		colorbar.set_ticks(ticks)
		colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
	plt.title(f"Relative Overtake Density Along Spa Centre Line\nSpa-Francorchamps {year} Race (First Half)")
	plt.axis('equal')
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	plt.tight_layout()
	plt.savefig(MAPPING_OUT, dpi=220, bbox_inches='tight')
	plt.close()
	print(f"Saved figures to {REPORT_DIAGRAMS}")
	

if __name__ == "__main__":
    main()

