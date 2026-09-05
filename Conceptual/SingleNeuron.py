# ----------------- IMPORTING REQUIRED MODULES ----------------------------
import numpy as np
# -------------------------------------------------------------------------

# =================== ReLu Activation function =============================

def ReLu(x):
    return max(0,x)

# ==========================================================================

def main():
    # ================== STEP 1 : DEFINE INPUT FEATURES IS X =================
    
    Border = "="*70

    print(Border)
    print(" STEP 1 : DEFINE INPUT FEATURES IS X ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    #                [x1  , x2 , x3]
    input = np.array([2.0,3.0,4.0])
    print(f"FEATURES X : {input}",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # ================== STEP 2 : DEFINE WEIGHTS i.e w =================

    print(Border)
    print(" STEP 2 : DEFINE WEIGHTS i.e w ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    #                  [ w1 , w2 , w3]
    weights = np.array([0.5,0.3,0.2])
    print(f"w : {weights}",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # ================== STEP 3 : DEFINE BIAS AS b =================

    print(Border)
    print(" STEP 3 : DEFINE BIAS AS b ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    bias = 1.0
    print(f"b : {bias}",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # ================== STEP 4 : CALCULATE WEIGHTED SUM i.e Z =================
    #
    # z = (x1*w1 + x2*w2 + x3*w3) + b (FORMULA)
    #

    print(Border)
    print(" STEP 4 : CALCULATE WEIGHTED SUM i.e Z ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    Z = np.dot(input,weights) + bias
    print(f"Z : {Z}",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # ================== STEP 5 : ACTIVATION FUNCTION ReLu() =================

    print(Border)
    print(" STEP 5 : ACTIVATION FUNCTION ReLu() ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    print("Successfully created ReLu() Activation function....",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # ================== STEP 6 : FINAL OUTPUT ==============================

    print(Border)
    print(" STEP 6 : FINAL OUTPUT ".center(70,"="))
    print(Border,end="\n\n")

    print("-"*70,end="\n")

    Y = ReLu(Z)
    print(f"Y : {Y}",end="\n")

    print("-"*70,end="\n\n")
    print(Border,end="\n\n")

    # =======================================================================

if __name__ == "__main__":
    main()