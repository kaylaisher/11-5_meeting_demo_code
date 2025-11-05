import numpy as np
import random
import csv
import sys

# --- Configuration ---
NUM_INPUTS       = 2
NUM_WEIGHTS      = 1
WEIGHT_BITS      = 4
MAX_WEIGHT_VALUE = (1 << WEIGHT_BITS) - 1

FAULT_TYPE_SAF    = "SAF"
FAULT_TYPE_BRIDGE = "BRIDGE"

def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

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
def simulate_row(pattern, weights, row_id, fault=None, verbose=False):
    """Simulate one row: multiply input bit * sum(weight bits)."""

    w_faulty = weights.copy()
    in_bits = pattern.bits  # [in[0], in[1]]

    
    # SAF injection
    if fault and fault.fault_type == FAULT_TYPE_SAF:
        f_row, f_bit = fault.index
        if f_row == row_id:
            old = w_faulty[f_bit]
            w_faulty[f_bit] = fault.forced_value
            if verbose:
                print(f"  -> SAF on Row{f_row} bit[{f_bit}] {old}->{fault.forced_value}")

        
    # Bridging injection
    if fault and fault.fault_type == FAULT_TYPE_BRIDGE:
        i, j = fault.i, fault.j
        a = in_bits[i]
        b = in_bits[j]
        
        if fault.bridge_type == 0:   # wired-AND
            new_val = a & b
            in_bits[i] = in_bits[j] = new_val

        elif fault.bridge_type == 1: # wired-OR
            new_val = a | b
            in_bits[i] = in_bits[j] = new_val

        elif fault.bridge_type == 2: # dominant (i dominates j)
            in_bits[j] = in_bits[i]

        elif fault.bridge_type == 3: # dominant-AND
            in_bits[j] = in_bits[i] & in_bits[j]

        if verbose:
            print(f"  -> BRIDGE inputs[{i}]↔[{j}] type={fault.bridge_type} "
                  f"→ {in_bits}")
        
    # Compute weighted-binary sum
    weight_value = (8 * w_faulty[3] +
                    4 * w_faulty[2] +
                    2 * w_faulty[1] +
                    1 * w_faulty[0])
    total = in_bits[row_id - 1] * weight_value

    if verbose:
        print(f"  Row{row_id}: in={in_bits[row_id - 1]}, weights={w_faulty}, "
              f"sum(bits)={sum(w_faulty)}, contrib={total}")
    return total

def detect_fault_with_two_rows(pattern: Pattern, w_row1, w_row2, fault: Fault, verbose=False):
    """
    Apply the pattern to both rows (row1 + row2).
    Return True if fault detected (i.e., sum deviation in either row), else False.
    """
    sum1_nom = simulate_row(pattern, w_row1, row_id=1, fault=None, verbose=verbose)
    sum2_nom = simulate_row(pattern, w_row2, row_id=2, fault=None, verbose=verbose)
    sum1_f   = simulate_row(pattern, w_row1, row_id=1, fault=fault, verbose=verbose)
    sum2_f   = simulate_row(pattern, w_row2, row_id=2, fault=fault, verbose=verbose)


    
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
    """Two-stage walking-one patterns for SAF detection."""
    patterns = []
    # Stage 1: SA0 detection (all ones except faulty bit = 0)
    for bit in range(WEIGHT_BITS):
        bits = [1, 1]  # both rows enabled
        # pattern metadata
        patterns.append(Pattern(bits))
    # Stage 2: SA1 detection (all zeros except faulty bit = 1)
    for bit in range(WEIGHT_BITS):
        bits = [1, 1]
        patterns.append(Pattern(bits))
    return patterns

def run_two_stage_saf_detection(fault_list, w_row1, w_row2, verbose=True):
    """
    Stage 1: all weights=1, detect SA0  (missing contribution)
    Stage 2: all weights=0, detect SA1  (extra contribution)
    """
    detected_faults = []

    # Stage 1 – detect SA0
    if verbose:
        print("\n=== Stage 1 : Walking-One for SA0 ===")
    w_all1 = [1,1,1,1]
    w_all0 = [0,0,0,0]

    for f in fault_list:
        if f.fault_type != FAULT_TYPE_SAF:
            continue
        f_row, f_bit = f.index
        if f.forced_value == 0:   # SA0 case
            # reference sum = 15
            sum_nom = simulate_row(Pattern([1,1]), [1,1,1,1], f_row, None, verbose=False)
            sum_faulty = simulate_row(Pattern([1,1]), [1,1,1,1], f_row, f, verbose=False)
            if sum_faulty != sum_nom:
                f.detected = True
                detected_faults.append(f)
                if verbose:
                    print(f"Detected SA0 at Row{f_row}.bit{f_bit}: {sum_nom}->{sum_faulty}")

    # Stage 2 – detect SA1
    if verbose:
        print("\n=== Stage 2 : Walking-One for SA1 ===")
    for f in fault_list:
        if f.fault_type != FAULT_TYPE_SAF:
            continue
        f_row, f_bit = f.index
        if f.forced_value == 1:   # SA1 case
            sum_nom = simulate_row(Pattern([1,1]), [0,0,0,0], f_row, None, verbose=False)
            sum_faulty = simulate_row(Pattern([1,1]), [0,0,0,0], f_row, f, verbose=False)
            if sum_faulty != sum_nom:
                f.detected = True
                detected_faults.append(f)
                if verbose:
                    print(f"Detected SA1 at Row{f_row}.bit{f_bit}: {sum_nom}->{sum_faulty}")

    # compute coverage
    num_detected = sum(1 for f in fault_list if f.detected)
    coverage = num_detected / len(fault_list)
    if verbose:
        print(f"\nSAF fault coverage: {coverage*100:.2f}% ({num_detected}/{len(fault_list)})")

    return coverage, [f for f in fault_list if not f.detected]

def main():
    # --- 1. Basic parameters ---
    base_weights = [1, 1, 1, 1]   # used for bridging tests (all ones)
    fault_list = []

    # --- 2. Generate faults ---
    # SAF faults: 2 rows × 4 bits × SA0/SA1
    for row_id in [1, 2]:
        for bit_idx in range(WEIGHT_BITS):
            fault_list.append(Fault(FAULT_TYPE_SAF, (row_id, bit_idx), 0))  # SA0
            fault_list.append(Fault(FAULT_TYPE_SAF, (row_id, bit_idx), 1))  # SA1

    # Bridging faults: inputs only (i=0, j=1)
    for bt in range(4):  # 0=AND,1=OR,2=dominant,3=dominant-AND
        fault_list.append(Fault(FAULT_TYPE_BRIDGE, i=0, j=1, bridge_type=bt))

    print(f"Total faults: {len(fault_list)} "
          f"(SAF={2*2*WEIGHT_BITS}, BRIDGE=4)")

    # =====================================================
    # === 3. Two-Stage SAF Detection Algorithm ============
    # =====================================================
    print("\n=== Stage 1 : Walking-One for SA0 (all weights = 1) ===")
    for f in fault_list:
        if f.fault_type == FAULT_TYPE_SAF and f.forced_value == 0:
            f_row, f_bit = f.index
            sum_nom = simulate_row(Pattern([1,1]), [1,1,1,1], f_row, None, verbose=False)
            sum_faulty = simulate_row(Pattern([1,1]), [1,1,1,1], f_row, f, verbose=False)
            if sum_faulty != sum_nom:
                f.detected = True
                print(f"Detected SA0 at Row{f_row}.bit{f_bit}: {sum_nom}->{sum_faulty}")

    print("\n=== Stage 2 : Walking-One for SA1 (all weights = 0) ===")
    for f in fault_list:
        if f.fault_type == FAULT_TYPE_SAF and f.forced_value == 1:
            f_row, f_bit = f.index
            sum_nom = simulate_row(Pattern([1,1]), [0,0,0,0], f_row, None, verbose=False)
            sum_faulty = simulate_row(Pattern([1,1]), [0,0,0,0], f_row, f, verbose=False)
            if sum_faulty != sum_nom:
                f.detected = True
                print(f"Detected SA1 at Row{f_row}.bit{f_bit}: {sum_nom}->{sum_faulty}")

    saf_detected = sum(1 for f in fault_list if f.fault_type == FAULT_TYPE_SAF and f.detected)
    saf_total = sum(1 for f in fault_list if f.fault_type == FAULT_TYPE_SAF)
    saf_cov = saf_detected / saf_total
    print(f"\nSAF coverage: {saf_cov*100:.2f}% ({saf_detected}/{saf_total})")

    # =====================================================
    # === 4. Bridging-Fault Detection =====================
    # =====================================================
    print("\n=== Bridging-Fault Detection ===")
    # Asymmetric row weights for better observability
    row1_weights = [1, 0, 1, 0]   # 0b1010 = 10
    row2_weights = [0, 1, 1, 1]   # 0b0111 = 7

    # Extended diagnostic input patterns
    patterns = [
        Pattern([0,0]),
        Pattern([0,1]),
        Pattern([1,0]),
        Pattern([1,1]),
        Pattern([0,1]),
        Pattern([1,0]),
        Pattern([1,1]),
        Pattern([0,0])
    ]

    bridge_faults = [f for f in fault_list if f.fault_type == FAULT_TYPE_BRIDGE]

    # Run detection using existing simulator
    bridge_cov, undetected_bridge = run_simulation(
        bridge_faults, patterns, row1_weights, row2_weights
    )

    print(f"Bridging coverage: {bridge_cov*100:.2f}% "
        f"({len(bridge_faults)-len(undetected_bridge)}/{len(bridge_faults)})")

    # =====================================================
    # === 5. Summary of Undetected Faults ================
    # =====================================================
    undetected = [f for f in fault_list if not f.detected]
    if undetected:
        print("\n=== Undetected Faults ===")
        for f in undetected:
            print("  ->", f)
    else:
        print("\n✅ All faults detected!")

    # =====================================================
    # === 6. Save Results ================================
    # =====================================================
    with open("fault_sim_results.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Type", "Location/Pair", "BridgeType", "Detected", "PatternIndex"])
        for f in fault_list:
            if f.fault_type == FAULT_TYPE_SAF:
                r, b = f.index
                writer.writerow(["SAF", f"(Row{r},bit{b})", "-", f.detected, f.detect_pattern])
            else:
                writer.writerow(["BRIDGE", f"inputs[{f.i}]<->inputs[{f.j}]",
                                 f.bridge_type, f.detected, f.detect_pattern])

    print("\nSimulation complete. Results saved to fault_sim_results.csv")

if __name__ == "__main__":
    main()
