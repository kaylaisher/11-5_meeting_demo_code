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
    """
    Simulate one row in the MAC slice.
    Each row contributes: in[row_id-1] * (8*w3 + 4*w2 + 2*w1 + 1*w0)
    Supports:
      • SAF (weight bit stuck-at)
      • 8 types of bridging faults between inputs[0] and [1]
    """

    w_faulty = weights.copy()
    in_bits = pattern.bits
    in_faulty = in_bits.copy()

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
        bt = fault.bridge_type
        
        if bt == 0:        # wired-AND
            a_faulty, b_faulty = a & b, a & b
        elif bt == 1:      # wired-OR
            a_faulty, b_faulty = a | b, a | b
        elif bt == 2:      # i dominant j
            a_faulty, b_faulty = a, a
        elif bt == 3:      # i dominant-AND j
            a_faulty, b_faulty = a, a & b
        elif bt == 4:      # j dominant i
            a_faulty, b_faulty = b, b
        elif bt == 5:      # j dominant-AND i
            a_faulty, b_faulty = a & b, b
        elif bt == 6:      # i dominant-OR j
            a_faulty, b_faulty = a, a | b
        elif bt == 7:      # j dominant-OR i
            a_faulty, b_faulty = a | b, b
        else:
            raise ValueError(f"Unknown bridge_type {bt}")
        

        in_faulty[i], in_faulty[j] = a_faulty, b_faulty

        if verbose:
            bridge_names = [
                "wired-AND", "wired-OR",
                "i dominant j", "i dominant-AND j",
                "j dominant i", "j dominant-AND i",
                "i dominant-OR j", "j dominant-OR i"
            ]
            print(f"  -> BRIDGE ({bridge_names[bt]}) between in[{i}] and in[{j}]")
            print(f"     Original inputs {in_bits} → Faulty inputs {in_faulty}")   

    # Compute weighted-binary sum
    weight_value = (8 * w_faulty[0] +
                    4 * w_faulty[1] +
                    2 * w_faulty[2] +
                    1 * w_faulty[3])
    total = in_faulty[row_id - 1] * weight_value

    if verbose:
        print(f"  Row{row_id}: in_faulty={in_faulty[row_id - 1]}, "
              f"weights={w_faulty}, weight_value={weight_value}, contrib={total}")
    
    return total

def detect_fault_with_two_rows(patterns, w_row1, w_row2, fault: Fault, verbose=True):
    """
    For each input pattern [00, 01, 10, 11], compute total sum = sum1 + sum2.
    Print the totals for nominal (no fault) and faulty cases side by side.
    Detects a fault if any total differs.
    """

    print(f"\n=== Testing {fault} ===")

    detected = False

    for pat in patterns:
        # nominal (no fault)
        sum1_nom = simulate_row(pat, w_row1, row_id=1, fault=None)
        sum2_nom = simulate_row(pat, w_row2, row_id=2, fault=None)
        total_nom = sum1_nom + sum2_nom

        # faulty
        sum1_f = simulate_row(pat, w_row1, row_id=1, fault=fault)
        sum2_f = simulate_row(pat, w_row2, row_id=2, fault=fault)
        total_f = sum1_f + sum2_f

        bits = ''.join(str(b) for b in pat.bits)
        print(f"Pattern {bits} → total_nom={total_nom:2d}, total_faulty={total_f:2d}")

        if total_nom != total_f:
            detected = True

    if detected:
        print(f"→ Fault detected: {fault}\n")
    else:
        print(f"→ Fault NOT detected: {fault}\n")

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
    for bt in range(8):  # 0=AND,1=OR,2=dominant,3=dominant-AND
        fault_list.append(Fault(FAULT_TYPE_BRIDGE, i=0, j=1, bridge_type=bt))

    print(f"Total faults: {len(fault_list)} "
          f"(SAF={2*2*WEIGHT_BITS}, BRIDGE=8)")

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

    patterns = [
        Pattern([0, 0]),
        Pattern([0, 1]),
        Pattern([1, 0]),
        Pattern([1, 1])
    ]

    # Asymmetric row weights for better observability
    row1_weights = [1, 0, 1, 0]   # 0b1010 = 10
    row2_weights = [0, 1, 1, 1]   # 0b0111 = 7


    bridge_faults = [f for f in fault_list if f.fault_type == FAULT_TYPE_BRIDGE]

    detected_count = 0
    for f in bridge_faults:
        if detect_fault_with_two_rows(patterns, row1_weights, row2_weights, f):
            detected_count += 1

    bridge_cov = detected_count / len(bridge_faults)

    print(f"Bridging coverage: {bridge_cov*100:.2f}% "
          f"({detected_count}/{len(bridge_faults)})")

    # =====================================================
    # === 5. Summary of Undetected Faults ================
    # =====================================================
    undetected = [f for f in fault_list if not f.detected]
    if undetected:
        print("\n=== Undetected Faults ===")
        for f in undetected:
            print("  ->", f)
    else:
        print("\n All faults detected!")

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
