import numpy as np
import random
import csv
import sys

# --- Configuration ---
NUM_INPUTS       = 256
NUM_WEIGHTS      = 256
WEIGHT_BITS      = 4
MAX_WEIGHT_VALUE = (1 << WEIGHT_BITS) - 1

FAULT_TYPE_SAF    = "SAF"
FAULT_TYPE_BRIDGE = "BRIDGE"

# --- Fault & Pattern Classes ---
class Fault:
    def __init__(self, fault_type, index=None, forced_value=None, i=None, j=None, bridge_type=None):
        self.fault_type     = fault_type
        self.index          = index            # for SAF
        self.forced_value   = forced_value     # for SAF
        self.i              = i               # for BRIDGE
        self.j              = j               # for BRIDGE
        self.bridge_type    = bridge_type      # for BRIDGE: 0..3
        self.detected       = False
        self.detect_pattern = None             # index of pattern that detected it
    def __repr__(self):
        if self.fault_type == FAULT_TYPE_SAF:
            return f"SAF(index={self.index}, forced={self.forced_value})"
        else:
            return f"BRIDGE(i={self.i}, j={self.j}, type={self.bridge_type})"

class Pattern:
    def __init__(self, bits):
        assert len(bits) == NUM_INPUTS
        self.bits = bits.copy()

# --- Simulator Functions ---
def simulate_row(pattern: Pattern, weights, fault: Fault=None, verbose=False):
    """
    Simulate one row accumulation (one stage) for a given pattern and optional fault.
    Returns the sum (integer).
    """
    in_faulty = pattern.bits.copy()
    w        = weights.copy()

    
    # SAF injection
    if fault and fault.fault_type == FAULT_TYPE_SAF:
        old_val = w[fault.index]
        w[fault.index] = fault.forced_value
        
    # Bridging injection
    if fault and fault.fault_type == FAULT_TYPE_BRIDGE:
        i = fault.i
        j = fault.j
        bt = fault.bridge_type
        a  = in_faulty[i]
        b  = in_faulty[j]
        if bt == 0:       # wired-AND
            new_val = a & b
            in_faulty[i] = new_val
            in_faulty[j] = new_val
        elif bt == 1:     # wired-OR
            new_val = a | b
            in_faulty[i] = new_val
            in_faulty[j] = new_val
        elif bt == 2:     # dominant (i dominates j)
            in_faulty[j] = in_faulty[i]
        elif bt == 3:     # dominant-AND
            in_faulty[j] = in_faulty[i] & in_faulty[j]
        else:
            raise ValueError(f"Unknown bridge_type {bt}")
        
    # Compute sum
    total = 0
    for k in range(NUM_INPUTS):
        if in_faulty[k]:
            total += w[k]
    return total

def detect_fault_with_two_rows(pattern: Pattern, w_row1, w_row2, fault: Fault, verbose=False):
    """
    Apply the pattern to both rows (row1 + row2).
    Return True if fault detected (i.e., sum deviation in either row), else False.
    """
    sum1_nom = simulate_row(pattern, w_row1, None, verbose=verbose)
    sum2_nom = simulate_row(pattern, w_row2, None, verbose=verbose)
    sum1_f   = simulate_row(pattern, w_row1, fault, verbose=verbose)
    sum2_f   = simulate_row(pattern, w_row2, fault, verbose=verbose)

    
    detected = (sum1_nom != sum1_f) or (sum2_nom != sum2_f)
    if detected:
        if detected:
            reason = []
            if sum1_nom != sum1_f:
                reason.append(f"sum1 changed : ({sum1_nom} -> {sum1_f})")
            if sum2_nom != sum2_f:
                reason.append(f"sum2 changed : ({sum2_nom} -> {sum2_f})")
            print(f"Fault detected: {fault}, Reason: {reason}")
        else:
            print()(f"Fault NOT detected")

    return detected

def run_simulation(fault_list, patterns, w_row1, w_row2):
    """
    Iterate over all faults and all patterns until detection.
    Returns fault coverage and list of undetected faults.
    """
    for f in fault_list:
        if f.detected:
            continue
        for p_idx, pat in enumerate(patterns):
            if detect_fault_with_two_rows(pat, w_row1, w_row2, f):
                f.detected = True
                f.detect_pattern = p_idx
                break

    num_detected = sum(1 for f in fault_list if f.detected)
    coverage     = num_detected / len(fault_list)
    undetected   = [f for f in fault_list if not f.detected]
    return coverage, undetected

# --- Pattern Generation (your assumed algorithm) ---
def generate_pattern_set():
    """
    Generate your pattern set automatically. Modify this function
    to implement your assumed testing algorithm.
    Here: simple walking-one + some random patterns.
    """
    patterns = []
    # Walking-oneqs
    for i in range(NUM_INPUTS):
        bits = np.zeros(NUM_INPUTS, dtype=int)
        bits[i] = 1
        patterns.append(Pattern(bits))
    # Add random patterns
    NUM_RANDOM = 500
    for _ in range(NUM_RANDOM):
        bits = np.random.randint(0,2, size=NUM_INPUTS).astype(int)
        patterns.append(Pattern(bits))
    return patterns

# --- Main ---
def main():
    # Define the two rows: stage1 weights, stage2 weights
    w_row1 = [0]*NUM_WEIGHTS
    w_row2 = [MAX_WEIGHT_VALUE]*NUM_WEIGHTS

    # Build fault list
    fault_list = []
    # SAF faults
    for idx in range(NUM_WEIGHTS):
        fault_list.append(Fault(FAULT_TYPE_SAF, index=idx, forced_value=MAX_WEIGHT_VALUE))
    # Bridging faults (adjacent inputs)
    for i in range(NUM_INPUTS-1):
        for bt in [0,1,2,3]:
            fault_list.append(Fault(FAULT_TYPE_BRIDGE, i=i, j=i+1, bridge_type=bt))

    print(f"Total faults: {len(fault_list)}")

    # Generate patterns (via your testing algorithm)
    patterns = generate_pattern_set()
    print(f"Total patterns generated: {len(patterns)}")

    # Run simulation
    coverage, undetected = run_simulation(fault_list, patterns, w_row1, w_row2)
    print(f"Fault coverage: {coverage*100:.2f}% ({len(fault_list)-len(undetected)}/{len(fault_list)})")
    print(f"Undetected faults count: {len(undetected)}")

    # Write results
    with open("fault_sim_results.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fault Type", "Location", "Detected?", "Pattern Index"])
        for f in fault_list:
            loc = (f.index,) if f.fault_type==FAULT_TYPE_SAF else (f.i, f.j)
            writer.writerow([f.fault_type, loc, f.detected, f.detect_pattern])

    print("Simulation complete. Results saved to fault_sim_results.csv")

if __name__ == "__main__":
    main()
