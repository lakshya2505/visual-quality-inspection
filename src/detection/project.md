Step 1 — Place and Set the DC Voltage Source (21V Bus)
Draw menu → Inputs and Sources → Voltage Source (DC)
Place it vertically on the left side of the canvas
Right click it → Edit → set voltage to 21V
This source has a positive terminal (top) and negative terminal (bottom)
The negative terminal is your ground rail — everything marked ground connects here
Step 2 — Place and Set the MOSFET (IRFZ44N)
Draw menu → Active Components → N-Channel MOSFET
Place it to the right of the voltage source
Right click → Edit → set:
Threshold voltage Vth = 4V
On resistance Ron = 0.028Ω
MOSFET in Falstad has exactly 3 terminals:
Gate — the horizontal stub coming out of the left side
Drain — the top vertical terminal
Source — the bottom vertical terminal

Connection:

Draw a wire from positive terminal of voltage source (top) → Drain of MOSFET (top terminal)
Step 3 — Place and Set the Inductor (100µH)
Draw menu → Passive Components → Inductor
Place it horizontally to the right of the MOSFET
Right click → Edit → set inductance = 0.0001 H (which is 100µH) and internal resistance = 0.5Ω
Inductor has a left terminal and a right terminal

Connection:

Draw a wire from Source of MOSFET (bottom terminal) → left terminal of Inductor
The junction where MOSFET Source meets inductor left terminal is your SW node — remember this point, you will attach a scope probe here later
Step 4 — Place and Set the Freewheeling Diode (1N5822)
Draw menu → Passive Components → Diode
Place it vertically below the SW node
Right click → Edit → set:
Forward voltage Vf = 0.45V
On resistance = 0.01Ω
Diode in Falstad has two terminals:
Anode — the flat base end (triangle base)
Cathode — the pointed tip end (with the line/stripe)

Connection:

Draw a wire from Cathode of diode (tip/stripe end) → SW node (same junction as MOSFET Source and inductor left terminal)
Draw a wire from Anode of diode (flat base end) → ground rail (negative terminal of voltage source)
Step 5 — Place and Set the Output Capacitor (1000µF)
Draw menu → Passive Components → Capacitor
Place it vertically to the right of the inductor, near the right terminal of inductor
Right click → Edit → set capacitance = 0.001 F (which is 1000µF)
Capacitor has a positive terminal (top) and negative terminal (bottom)

Connection:

Draw a wire from right terminal of Inductor → positive terminal of Capacitor (top)
The junction where inductor right terminal meets capacitor positive terminal is your Vout node — attach scope probe here later
Draw a wire from negative terminal of Capacitor (bottom) → ground rail
Step 6 — Place the Load Resistor (Motor Substitute — 10Ω)
Draw menu → Passive Components → Resistor
Place it vertically in parallel with the capacitor (same two nodes)
Right click → Edit → set resistance = 10Ω
Resistor has a top terminal and a bottom terminal

Connection:

Draw a wire from top terminal of Resistor → Vout node (same junction as inductor right terminal and capacitor positive terminal)
Draw a wire from bottom terminal of Resistor → ground rail
Step 7 — Place and Set the PWM Clock Source (Replaces IR2110 + Arduino)
Draw menu → Inputs and Sources → Clock
Place it to the left of the MOSFET Gate, horizontally
Right click → Edit → set:
High voltage = 10V
Low voltage = 0V
Frequency = 31000 Hz
Duty cycle = 50% (you will change this later to test different output voltages)
Clock source has a positive output terminal and a ground/reference terminal

Connection:

Draw a wire from positive output terminal of Clock source → Gate of MOSFET (horizontal stub on left side)
Draw a wire from ground terminal of Clock source → Source of MOSFET (bottom terminal) — this is critical, gate drive reference must be tied to MOSFET Source, not to circuit ground
Step 8 — Place Oscilloscope Probes
Draw menu → Outputs → Scope
You need 3 separate scope probes placed at 3 different nodes

Probe 1 — SW Node:

Click on the wire at the junction of MOSFET Source + Inductor left terminal + Diode Cathode
This shows the switching square wave

Probe 2 — Vout Node:

Click on the wire at the junction of Inductor right terminal + Capacitor positive terminal + Resistor top terminal
This shows the output DC voltage with ripple

Probe 3 — Across the Inductor:

Place one probe terminal on the left terminal of inductor (SW node side)
Place other probe terminal on the right terminal of inductor (Vout node side)
This shows the triangular inductor current waveform
Step 9 — Complete Ground Rail

Make sure these are all connected to the same ground rail (negative terminal of voltage source):

Negative terminal of voltage source → ground
Anode of freewheeling diode → ground
Negative terminal of output capacitor → ground
Bottom terminal of load resistor → ground
Ground terminal of Clock source → MOSFET Source (not circuit ground — this is the only exception)

Voltage Source (+) ──────────────── MOSFET Drain

MOSFET Source ──── SW NODE ──── Inductor (left terminal)
                      │
                  Diode Cathode
                      │
                  Diode Anode ──── Ground

Inductor (right terminal) ──── VOUT NODE ──── Capacitor (+)
                                    │
                                Resistor (top)
                                    │
                              Resistor (bottom) ──── Ground

Capacitor (−) ──── Ground

Clock Source (+) ──── MOSFET Gate
Clock Source (−) ──── MOSFET Source

Voltage Source (−) ──── Ground Rail

Step 10 — Run and Verify
Press the Play button (bottom left of Falstad)
Watch all three scope windows open automatically
Set simulation speed to about 50% using the speed slider for clean waveform visibility

You should see:

SW node scope — clean square wave flipping between 0V and 21V at 31kHz
Vout scope — flat DC line at approximately 10.5V with tiny ripple on top (at 50% duty cycle)
Inductor scope — triangular wave rising and falling at 31kHz

Then change duty cycle to verify buck formula:

Right click Clock source → Edit → change duty cycle to 30% → Vout should drop to approximately 6.3V
Change duty cycle to 70% → Vout should rise to approximately 14.7V

If all three readings match Vout = Duty% × 21V, your Falstad simulation is correct and verified.